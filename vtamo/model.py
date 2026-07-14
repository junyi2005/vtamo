"""
model.py — VTaMo: FlanT5 SLT with explicit multi-granularity vision-text alignment.

The model trains a LoRA-adapted Flan-T5 decoder on frozen CLIP-ViT features, jointly
with alignment objectives at three granularities:

1. Local alignment (vtamo.ot_sinkhorn.LocalAlignmentModule)
   Entropy-regularized OT (Sinkhorn) between frame windows and pseudo-gloss tokens,
   with a learnable NULL token absorbing transitional/co-articulation frames.
   L_local = beta_local * L_align + beta_tv * L_tv + beta_null * L_null
   Epsilon is annealed with the three-phase schedule (eps_high -> eps_mid -> eps_low).

2. Global alignment (vtamo.global_align)
   A learnable orthogonal transform T calibrates the sign embedding space against the
   text embedding space, supervised by an EMD objective over a FIFO memory queue of
   attention-pooled sentence vectors, and constrained by ||T^T T - I||_F^2.
   L_global = lambda_g(t) * L_EMD + beta_orth * L_orth
   T also shapes the local OT cost. The global loss updates only T.

3. Position-aligned contrastive learning
   Windows are reordered into target-token order using the transport plan and bound to
   their text-token embeddings with InfoNCE in a shared projected space.
   Gradients flow to the temporal encoder and projectors, not to the T5 LoRA params
   (text features are detached).

Reordering is applied during training only; at inference the target order is unknown,
so windows are consumed in signing order.

Key config parameters:
    local_align_enabled / local_align_iters / null_ratio_target
    beta_local / beta_tv / beta_null
    use_three_phase_eps / eps_high / eps_mid / eps_low / eps_phase2_epochs / eps_phase3_epochs
    global_emd_enabled / emd_eps / emd_iters / emd_ramp_steps / lambda_g_max / beta_orth
    global_queue_size / global_queue_min_size
    position_contrastive_enabled / position_contra_temp / beta_pos_contra
    position_contra_use_proj / position_contra_proj_hidden / position_contra_proj_dim
"""

import os
import torch
import torch.nn as nn
import random
import math
from typing import Dict, List, Optional, Tuple, Any

import torch.nn.functional as F

from torch.nn.utils.rnn import pad_sequence
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, T5ForConditionalGeneration
from transformers import BertConfig, BertModel
from peft import LoraConfig, get_peft_model, TaskType

from vtamo.tconv import TemporalConv, AttentionTemporalConv
from utils.helpers import create_mask, derangement
from vtamo.mm_projector import build_vision_projector
from utils.evaluate import evaluate_results
from vtamo.clip_loss import clip_loss
from vtamo.asb import AbstractSLT
from vtamo.ot_sinkhorn import (
    LocalAlignmentModule,
    reorder_by_window_alignment,
    compute_position_contrastive_loss,
    sinkhorn,
)
from vtamo.global_align import (
    attention_pool_sequence,
    FIFOMemoryQueue,
    compute_global_emd_loss_sentence,
    compute_orth_loss,
    compute_lambda_g_schedule,
    compute_eps_schedule,
    procrustes_init_from_ot_plan,
)
from transformers import get_cosine_schedule_with_warmup


os.environ["TOKENIZERS_PARALLELISM"] = "false"


torch.set_float32_matmul_precision('high')


class FlanT5SLT(AbstractSLT):
    """
    FlanT5-based Sign Language Translation model with multimodal capabilities.
    """
    def __init__(
        self, 
        tuning_type: str = 'lora', 
        model_name: Optional[str] = None, 
        frame_sample_rate: int = 1, 
        prompt: str = '',
        input_size: int = 1024,
        fusion_mode: str = 'joint',
        inter_hidden: int = 768,
        max_frame_len: int = 1024,
        max_txt_len: int = 64,
        cross_modal_align: bool = False,
        warm_up_steps: Optional[int] = None,
        combined_loss: bool = False,
        alpha: float = 0.1,
        use_resampler: bool = False,
        sampling_length: int = 64,
        cache_dir: str = "./cache",
        use_in_context: bool = False,
        num_in_context: int = 0,
        lora_r: int = 16,
        lora_alpha: int = 32,
        lora_dropout: float = 0.1,
        distill_enabled: bool = False,
        distill_alpha: float = 0.6,
        distill_temperature: float = 3.0,
        teacher_model_name: Optional[str] = None,
        teacher_ckpt: Optional[str] = None,
        teacher_tuning_type: str = 'lora',
        teacher_lora_r: int = 16,
        teacher_lora_alpha: int = 32,
        teacher_lora_dropout: float = 0.1,
        # Local alignment parameters
        local_align_enabled: bool = False,
        local_align_eps: float = 0.1,
        local_align_eps_start: float = 0.1,    # Starting eps for decay
        local_align_eps_end: float = 0.06,     # Ending eps for decay
        local_align_eps_decay_epochs: int = 40, # Epochs to decay over
        local_align_iters: int = 10,
        null_ratio_target: float = 0.2,  # Target: 20% of windows should have NULL as argmax
        beta_local: float = 1.0,
        beta_tv: float = 0.1,
        beta_null: float = 0.1,  # Weight for null ratio regularization
        # Attention Temporal Conv parameters (replace TemporalConv)
        use_attention_pool: bool = False,  # Whether to use AttentionTemporalConv
        attn_downsample_rate: int = 4,     # Downsample rate for attention pooling
        attn_num_heads: int = 4,           # Number of attention heads
        attn_window_size: int = 16,        # Window size for local attention
        attn_dropout: float = 0.1,         # Dropout rate
        attn_log_stats: bool = False,      # Log attention pooling stats
        # CONTRASTIVE version: Position-aligned contrastive learning parameters
        position_contrastive_enabled: bool = False,  # Enable position-aligned contrastive loss
        position_contra_temp: float = 0.07,          # Temperature for InfoNCE
        beta_pos_contra: float = 0.1,                # Weight for position contrastive loss
        position_contra_use_proj: bool = False,      # Use learnable projection heads for contrastive space
        position_contra_proj_hidden: Optional[int] = None,  # Hidden size for contrastive MLP
        position_contra_proj_dim: int = 768,         # Output dim of the contrastive projection
        lr_warmup_steps: Optional[int] = 2000,       # LR linear warm-up steps (None = 10% of training)
        # GLOBAL version: three-phase epsilon annealing for the local OT (paper Sec. 4)
        eps_high: float = 0.12,           # Phase 1 epsilon (soft)
        eps_mid: float = 0.10,            # Phase 2 epsilon
        eps_low: float = 0.03,            # Phase 3 epsilon (sharp)
        eps_phase2_epochs: int = 10,      # Epochs for phase 2 (eps_high -> eps_mid)
        eps_phase3_epochs: int = 80,      # Epochs for phase 3 (eps_mid -> eps_low)
        eps_schedule_type: str = "linear",
        use_three_phase_eps: bool = False,  # Use the paper's 3-phase schedule
        # GLOBAL version: global alignment (learnable orthogonal T + EMD over a FIFO queue)
        # Three-stage schedule:
        # - Stage 0 (0 ~ warm_up_steps): T exists and shapes the local OT, global loss = 0,
        #   only the orthogonality constraint acts on T
        # - Stage 1 (warm_up_steps ~ + emd_ramp_steps): lambda_g ramps 0 -> lambda_g_max
        # - Stage 2 (after): lambda_g held at lambda_g_max
        global_emd_enabled: bool = False,
        emd_eps: float = 0.05,            # Epsilon for the global Sinkhorn
        emd_iters: int = 20,              # Iterations for the global Sinkhorn
        emd_ramp_steps: int = 4000,       # Stage 1 duration
        lambda_g_max: float = 0.1,        # Maximum global loss weight
        beta_orth: float = 0.05,          # Orthogonality constraint weight
        emd_schedule_type: str = "linear",
        global_queue_size: int = 256,     # FIFO queue capacity for sentence vectors
        global_queue_min_size: int = 32,  # Min queue occupancy before computing the global OT
        **kwargs
    ):
        super().__init__(**kwargs)
        
        # Configuration parameters
        self.input_size = input_size
        self.prompt = prompt
        self.model_name = model_name
        self.frame_sample_rate = frame_sample_rate
        self.fusion_mode = fusion_mode
        self.inter_hidden = inter_hidden
        self.max_frame_len = max_frame_len
        self.max_txt_len = max_txt_len
        self.tuning_type = tuning_type
        self.cross_modal_align = cross_modal_align
        self.warm_up_steps = warm_up_steps
        self.combined_loss = combined_loss
        self.alpha = alpha
        self.use_resampler = use_resampler
        self.sampling_length = sampling_length
        self.cache_dir = cache_dir
        self.use_in_context = use_in_context
        self.num_in_context = num_in_context
        self.lora_r = lora_r
        self.lora_alpha = lora_alpha
        self.lora_dropout = lora_dropout
        self.distill_enabled = distill_enabled and teacher_ckpt is not None
        self.distill_alpha = distill_alpha
        self.distill_temperature = distill_temperature
        self.teacher_model_name = teacher_model_name or model_name
        self.teacher_ckpt = teacher_ckpt
        self.teacher_tuning_type = teacher_tuning_type
        self.teacher_lora_r = teacher_lora_r
        self.teacher_lora_alpha = teacher_lora_alpha
        self.teacher_lora_dropout = teacher_lora_dropout

        # Local alignment parameters
        self.local_align_enabled = local_align_enabled
        self.local_align_eps = local_align_eps
        self.local_align_eps_start = local_align_eps_start
        self.local_align_eps_end = local_align_eps_end
        self.local_align_eps_decay_epochs = local_align_eps_decay_epochs
        self.local_align_iters = local_align_iters
        self.null_ratio_target = null_ratio_target
        self.beta_local = beta_local
        self.beta_tv = beta_tv
        self.beta_null = beta_null

        # Attention Temporal Conv parameters
        self.use_attention_pool = use_attention_pool
        self.attn_downsample_rate = attn_downsample_rate
        self.attn_num_heads = attn_num_heads
        self.attn_window_size = attn_window_size
        self.attn_dropout = attn_dropout
        self.attn_log_stats = attn_log_stats

        # CONTRASTIVE version: Position-aligned contrastive learning parameters
        self.position_contrastive_enabled = position_contrastive_enabled
        self.position_contra_temp = position_contra_temp
        self.beta_pos_contra = beta_pos_contra
        self.position_contra_use_proj = position_contra_use_proj
        self.position_contra_proj_hidden = position_contra_proj_hidden
        self.position_contra_proj_dim = position_contra_proj_dim
        self.lr_warmup_steps = lr_warmup_steps

        # GLOBAL version: three-phase epsilon annealing
        self.eps_high = eps_high
        self.eps_mid = eps_mid
        self.eps_low = eps_low
        self.eps_phase2_epochs = eps_phase2_epochs
        self.eps_phase3_epochs = eps_phase3_epochs
        self.eps_schedule_type = eps_schedule_type
        self.use_three_phase_eps = use_three_phase_eps
        self._total_steps = None        # cached from the trainer on first train batch
        self._steps_per_epoch = None    # cached from the trainer on first train batch

        # GLOBAL version: global alignment parameters
        self.global_emd_enabled = global_emd_enabled
        self.emd_eps = emd_eps
        self.emd_iters = emd_iters
        self.emd_ramp_steps = emd_ramp_steps
        self.lambda_g_max = lambda_g_max
        self.beta_orth = beta_orth
        self.emd_schedule_type = emd_schedule_type
        self.global_queue_size = global_queue_size
        self.global_queue_min_size = global_queue_min_size
        self._procrustes_inited = False
        self._global_queue = None  # built in prepare_models(), needs the LM hidden size

        self.prepare_models(model_name)

        # Apply the selected tuning strategy
        if tuning_type == 'freeze':
            self._freeze_model()
        elif tuning_type == 'lora':
            self._apply_lora()

        self._teacher = None
        if self.distill_enabled:
            self._setup_teacher()

        self.set_container()
        
    # def load_pretrained_weights(self, checkpoint_path: str) -> None:
    #     """Load weights from a pretrained checkpoint."""
    #     checkpoint = torch.load(checkpoint_path, map_location=self.device, weights_only=True)
        
    #     # Get model's state dict
    #     model_state_dict = self.state_dict()
    #     checkpoint_state_dict = checkpoint['state_dict']
        
    #     # Filter out mismatched keys
    #     filtered_state_dict = {}
    #     for k, v in checkpoint_state_dict.items():
    #         if k in model_state_dict and v.size() == model_state_dict[k].size():
    #             filtered_state_dict[k] = v
        
    #     # Load the filtered state dict
    #     self.load_state_dict(filtered_state_dict)
    #     print(f'Checkpoint loaded from {checkpoint_path}. Loaded {len(filtered_state_dict)}/{len(checkpoint_state_dict)} parameters.')
    
    def load_pretrained_weights(self, checkpoint_path):
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        self.load_state_dict(checkpoint['state_dict'])
        print(f'Checkpoint is loaded from {checkpoint_path}.')

    def _apply_lora(self) -> None:
        """Apply LoRA adapter to the T5 model."""
        lora_config = LoraConfig(
            r=self.lora_r,
            lora_alpha=self.lora_alpha,
            target_modules=["q", "v"],
            lora_dropout=self.lora_dropout,
            bias="none",
            task_type=TaskType.SEQ_2_SEQ_LM
        )
        self.t5_model = get_peft_model(self.t5_model, lora_config)
        print("LoRA adapter applied to T5 model.")

    def _freeze_model(self) -> None:
        """Freeze the T5 model parameters."""
        self.t5_model.eval()
        for params in self.t5_model.parameters():
            params.requires_grad = False
        print("T5 model frozen.")

    def _setup_teacher(self) -> None:
        """Initialize and freeze teacher model for distillation."""
        teacher = FlanT5SLT(
            tuning_type=self.teacher_tuning_type,
            model_name=self.teacher_model_name,
            frame_sample_rate=self.frame_sample_rate,
            prompt=self.prompt,
            input_size=self.input_size,
            fusion_mode=self.fusion_mode,
            inter_hidden=self.inter_hidden,
            max_frame_len=self.max_frame_len,
            max_txt_len=self.max_txt_len,
            cross_modal_align=self.cross_modal_align,
            warm_up_steps=None,
            combined_loss=self.combined_loss,
            alpha=self.alpha,
            use_resampler=self.use_resampler,
            sampling_length=self.sampling_length,
            cache_dir=self.cache_dir,
            use_in_context=self.use_in_context,
            num_in_context=self.num_in_context,
            lora_r=self.teacher_lora_r,
            lora_alpha=self.teacher_lora_alpha,
            lora_dropout=self.teacher_lora_dropout,
            distill_enabled=False,
            teacher_ckpt=None,
            lr=self.lr,
            monitor=self.monitor,
            scheduler_config=self.scheduler_config,
            max_length=self.max_length,
            beam_size=self.beam_size,
        )
        teacher.load_pretrained_weights(self.teacher_ckpt)
        teacher.eval()
        for params in teacher.parameters():
            params.requires_grad = False

        # Keep teacher out of Lightning checkpoints and optimizer params.
        self.__dict__["_teacher"] = teacher

    def _ensure_teacher_on_device(self) -> None:
        if self._teacher is None:
            return
        teacher_device = next(self._teacher.parameters()).device
        if teacher_device != self.device:
            self._teacher.to(self.device)
        self._teacher.eval()

    def _compute_kd_loss(
        self,
        student_logits: torch.Tensor,
        teacher_logits: torch.Tensor,
        token_mask: Optional[torch.Tensor],
    ) -> torch.Tensor:
        temperature = self.distill_temperature
        log_probs = F.log_softmax(student_logits / temperature, dim=-1)
        probs = F.softmax(teacher_logits / temperature, dim=-1)
        kl = F.kl_div(log_probs, probs, reduction="none").sum(-1)
        if token_mask is not None:
            kl = kl * token_mask
            denom = token_mask.sum().clamp_min(1.0)
            kl = kl.sum() / denom
        else:
            kl = kl.mean()
        return kl * (temperature ** 2)

    def set_container(self) -> None:
        self.generated = []
        self.references = []
        self.train_losses = []
        self.val_losses = []
        # Detailed loss tracking for training
        self.train_loss_details = {
            't5_loss': [],
            'contra_loss': [],
            'vt_global_loss': [],
            'vt_local_loss': [],
            'tv_loss': [],
            'null_reg_loss': [],  # NULL ratio regularization loss
            'kd_loss': [],
            'combined_loss': [],
            'null_ratio': [],  # Actual ratio (argmax based)
            'soft_null_ratio': [],  # Soft ratio (probability based)
            'null_bias': [],  # Learned NULL bias
            # CONTRASTIVE version: Position-aligned contrastive loss
            'pos_contra_loss': [],
            'pos_contra_pos_sim': [],
            'pos_contra_neg_sim': [],
            'pos_contra_gap': [],
            'attn_peak': [],
            'attn_entropy': [],
            'attn_peak_l1': [],
            'attn_entropy_l1': [],
            'attn_peak_l2': [],
            'attn_entropy_l2': [],
        }

    def prepare_models(self, t5_model: str) -> None:
        """
        Prepare the textual and visual models.
        
        Args:
            t5_model: Name or path of the T5 model to use
        """
        
        # Load the textual model
        self.t5_model = T5ForConditionalGeneration.from_pretrained(
            t5_model, 
            cache_dir=self.cache_dir,
            torch_dtype=torch.bfloat16, 
        )
        
        # Load the tokenizer
        self.t5_tokenizer = AutoTokenizer.from_pretrained(
            t5_model, 
            cache_dir=self.cache_dir,
            max_length=self.max_txt_len,
        )

        # Load the vision projectors
        self.spatio_proj = build_vision_projector('linear', self.input_size, self.inter_hidden)
        self.spatiotemp_proj = build_vision_projector('linear', 1024, self.inter_hidden)
        self.fusion_proj = build_vision_projector('mlp2x_gelu', self.inter_hidden, self.t5_model.config.hidden_size)
        if self.position_contra_use_proj:
            hidden_size = self.t5_model.config.hidden_size
            proj_hidden = self.position_contra_proj_hidden or hidden_size
            # Both towers project into a shared contrastive space of
            # position_contra_proj_dim (paper: 768). InfoNCE is computed there.
            out_dim = self.position_contra_proj_dim or hidden_size
            self.pos_contra_vis_proj = nn.Sequential(
                nn.Linear(hidden_size, proj_hidden),
                nn.GELU(),
                nn.Linear(proj_hidden, out_dim),
                nn.LayerNorm(out_dim),
            )
            self.pos_contra_txt_proj = nn.Sequential(
                nn.Linear(hidden_size, proj_hidden),
                nn.GELU(),
                nn.Linear(proj_hidden, out_dim),
                nn.LayerNorm(out_dim),
            )

        # Load the temporal encoder
        # NEW: Choose between fixed TemporalConv or learnable AttentionTemporalConv
        if self.use_attention_pool:
            # AttentionTemporalConv: learnable attention-based downsampling
            # - Each output position attends to input frames with learnable weights
            # - Receptive field is learned, not fixed
            # - Gradients flow to all frames in the window (not just max)
            print(f"Using AttentionTemporalConv with downsample_rate={self.attn_downsample_rate}, "
                  f"window_size={self.attn_window_size}, num_heads={self.attn_num_heads}")
            self.temporal_encoder = AttentionTemporalConv(
                input_size=self.inter_hidden,
                hidden_size=self.inter_hidden,
                downsample_rate=self.attn_downsample_rate,
                num_heads=self.attn_num_heads,
                window_size=self.attn_window_size,
                dropout=self.attn_dropout,
            )
        else:
            # TemporalConv: fixed receptive field (K5+P2+K5+P2)
            # - Each output frame sees ~16 input frames (fixed)
            # - MaxPool is not learnable
            self.temporal_encoder = TemporalConv(self.inter_hidden, self.inter_hidden)

        # if self.cross_modal_align:
        self.logit_scale = nn.Parameter(torch.tensor(2.6592))

        # Local alignment module (always initialize, use controlled by local_align_enabled)
        # New semantics: null_ratio_target = target % of windows whose argmax is NULL
        self.local_align_module = LocalAlignmentModule(
            hidden_size=self.t5_model.config.hidden_size,
            eps=self.local_align_eps,
            n_iters=self.local_align_iters,
            null_ratio_target=self.null_ratio_target,
            beta_local=self.beta_local,
            beta_tv=self.beta_tv,
            beta_null=self.beta_null
        )

        # GLOBAL version: FIFO memory queue of pooled sentence vectors for the global OT
        self._global_queue = FIFOMemoryQueue(
            max_size=self.global_queue_size,
            feature_dim=self.t5_model.config.hidden_size
        )

    def prepare_inputs(
        self, 
        visual_outputs: torch.Tensor, 
        visual_mask: torch.Tensor, 
        samples: Dict, 
        split: str, 
        batch_idx: int
    ) -> Tuple[torch.Tensor, torch.Tensor, Any, torch.Tensor]:
        """
        Prepare combined inputs for the T5 model.
        
        Args:
            visual_outputs: Visual features
            visual_mask: Mask for visual features
            samples: Input samples
            split: Current split (train, val, test)
            batch_idx: Current batch index
            
        Returns:
            Tuple of (joint_outputs, joint_mask, output_tokens, targets)
        """
        bs = visual_outputs.shape[0]
        
        # Prepare the prompt with language information
        prompts = [f'{self.prompt}'] * bs
        prompts = [p.format(l) for p, l in zip(prompts, samples['lang'])]
        
        if self.use_in_context:
            prompts = [f"{p} {c}" for p, c in zip(prompts, samples['ex_lang_trans'])]
        
        # Tokenize prompts
        input_tokens = self.t5_tokenizer(
            prompts,
            padding="longest",
            truncation=True,
            return_tensors="pt",
        ).to(self.device)
        
        # Get lengths for visual and prompt sequences
        visual_lengths = visual_mask.sum(1).long()  # Ensure integer type
        prompt_lengths = input_tokens.attention_mask.sum(1).long()
        new_lengths = visual_lengths + prompt_lengths

        # Convert tokens to embeddings
        input_embeds = self.t5_model.encoder.embed_tokens(input_tokens.input_ids)

        # Concatenate visual and text embeddings
        joint_outputs = []
        for i in range(bs):
            vlen = visual_lengths[i].item()  # Convert to Python int for slicing
            plen = prompt_lengths[i].item()
            vis_out = visual_outputs[i, :vlen, :]
            prompt_embeds = input_embeds[i, :plen, :]
            concat_sample = torch.cat((vis_out, prompt_embeds), dim=0)
            joint_outputs.append(concat_sample)
        
        # Pad the combined embeddings
        joint_outputs = pad_sequence(joint_outputs, batch_first=True)
        joint_mask = create_mask(seq_lengths=new_lengths.tolist(), device=self.device)
        
        # Tokenize target texts
        output_tokens = self.t5_tokenizer(
            samples['text'],
            padding="longest",
            return_tensors="pt",
        ).to(self.device)
        
        # Prepare target labels (replace pad tokens with -100)
        targets = output_tokens.input_ids.masked_fill(
            output_tokens.input_ids == self.t5_tokenizer.pad_token_id, -100
        )
        
        return joint_outputs, joint_mask, output_tokens, targets

    def prepare_visual_inputs(self, samples: Dict) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Prepare visual inputs based on the fusion mode.
        
        Args:
            samples: Input samples containing visual features
            
        Returns:
            Tuple of (visual_outputs, visual_masks)
        """
        self._last_attn_stats = None
        # Determine which visual features to use based on fusion mode
        if self.fusion_mode in ['joint']:
            spatial = spatiotemporal = True
        else:
            spatial = self.fusion_mode == 'spatial'
            spatiotemporal = self.fusion_mode == 'spatiotemporal'

        # Process spatial features if needed
        if spatial:
            pixel_values = pad_sequence(samples['pixel_values'], batch_first=True)
            spatial_outputs = self.spatio_proj(pixel_values)
            spatial_mask = create_mask(seq_lengths=samples['num_frames'], device=self.device)
        
        # Process spatiotemporal features if needed
        if spatiotemporal:
            spatiotemporal_outputs = pad_sequence(samples['glor_values'], batch_first=True)
            spatiotemporal_outputs = self.spatiotemp_proj(spatiotemporal_outputs)
            spatiotemporal_mask = create_mask(seq_lengths=samples['glor_lengths'], device=self.device)
        
        # Combine features for joint mode
        if self.fusion_mode == 'joint':
            bs = spatial_outputs.shape[0]
            spatial_length = spatial_mask.sum(1).long()
            spatiotemporal_length = spatiotemporal_mask.sum(1).long()
            new_length = spatial_length + spatiotemporal_length

            # Concatenate spatial and spatiotemporal features for each sample
            joint_outputs = []
            for i in range(bs):
                slen = spatial_length[i].item()  # Convert to Python int for slicing
                stlen = spatiotemporal_length[i].item()
                valid_spatial_output = spatial_outputs[i, :slen, :]
                valid_spatiotemporal_output = spatiotemporal_outputs[i, :stlen, :]
                concat_sample = torch.cat((valid_spatial_output, valid_spatiotemporal_output), dim=0)
                joint_outputs.append(concat_sample)
            joint_outputs = pad_sequence(joint_outputs, batch_first=True)
            
            # Apply temporal encoder
            if self.attn_log_stats and self.use_attention_pool:
                visual_conv_outputs = self.temporal_encoder(
                    joint_outputs.permute(0,2,1),
                    torch.tensor(new_length.tolist(), device=self.device),
                    return_attn_stats=True,
                )
                self._last_attn_stats = {
                    "attn_peak": visual_conv_outputs.get("attn_peak_mean"),
                    "attn_entropy": visual_conv_outputs.get("attn_entropy_mean"),
                }
                layer_peaks = visual_conv_outputs.get("layer_attn_peak_mean")
                layer_entropy = visual_conv_outputs.get("layer_attn_entropy_mean")
                if layer_peaks:
                    if len(layer_peaks) > 0:
                        self._last_attn_stats["attn_peak_l1"] = layer_peaks[0]
                    if len(layer_peaks) > 1:
                        self._last_attn_stats["attn_peak_l2"] = layer_peaks[1]
                if layer_entropy:
                    if len(layer_entropy) > 0:
                        self._last_attn_stats["attn_entropy_l1"] = layer_entropy[0]
                    if len(layer_entropy) > 1:
                        self._last_attn_stats["attn_entropy_l2"] = layer_entropy[1]
            else:
                visual_conv_outputs = self.temporal_encoder(
                    joint_outputs.permute(0,2,1), torch.tensor(new_length.tolist(), device=self.device)
                )
            
            visual_outputs = visual_conv_outputs['visual_feat'].permute(1,0,2)
            visual_masks = create_mask(
                seq_lengths=visual_conv_outputs['feat_len'].to(torch.int).tolist(), 
                device=self.device
            ) 
        else:
            # Use single feature type
            if spatial:
                if self.attn_log_stats and self.use_attention_pool:
                    spatial_conv_outputs = self.temporal_encoder(
                        spatial_outputs.permute(0,2,1),
                        torch.tensor(samples['num_frames'], device=self.device),
                        return_attn_stats=True,
                    )
                    self._last_attn_stats = {
                        "attn_peak": spatial_conv_outputs.get("attn_peak_mean"),
                        "attn_entropy": spatial_conv_outputs.get("attn_entropy_mean"),
                    }
                    layer_peaks = spatial_conv_outputs.get("layer_attn_peak_mean")
                    layer_entropy = spatial_conv_outputs.get("layer_attn_entropy_mean")
                    if layer_peaks:
                        if len(layer_peaks) > 0:
                            self._last_attn_stats["attn_peak_l1"] = layer_peaks[0]
                        if len(layer_peaks) > 1:
                            self._last_attn_stats["attn_peak_l2"] = layer_peaks[1]
                    if layer_entropy:
                        if len(layer_entropy) > 0:
                            self._last_attn_stats["attn_entropy_l1"] = layer_entropy[0]
                        if len(layer_entropy) > 1:
                            self._last_attn_stats["attn_entropy_l2"] = layer_entropy[1]
                else:
                    spatial_conv_outputs = self.temporal_encoder(
                        spatial_outputs.permute(0,2,1), torch.tensor(samples['num_frames'], device=self.device)
                    )
                visual_outputs = spatial_conv_outputs['visual_feat'].permute(1,0,2)
                visual_masks = create_mask(
                    seq_lengths=spatial_conv_outputs['feat_len'].to(torch.int).tolist(), 
                    device=self.device
                )
            elif spatiotemporal:
                visual_outputs = spatiotemporal_outputs
                visual_masks = spatiotemporal_mask
            else:
                raise NotImplementedError("Invalid fusion mode")
        
        return visual_outputs, visual_masks

    def get_inputs(self, batch: List) -> Dict:
        """
        Process batch inputs into a structured dictionary.
        
        Args:
            batch: Raw batch from dataloader
            
        Returns:
            Processed inputs dictionary
        """
        pixel_values, glor_values, masks, ids = [], [], [], []
        texts, glosses = [], []
        num_frames, glor_lengths, langs = [], [], []
        ex_lang_translations = []
        
        max_frame_len = self.max_frame_len

        for sample in batch:
            if sample['pixel_value'].shape[0] != 0:
                # Calculate number of frames after sampling
                nframe = math.ceil(sample['num_frames'] / self.frame_sample_rate)
                pval = sample['pixel_value'][::self.frame_sample_rate]

                # Collect metadata
                ids.append(sample['id'])
                texts.append(sample['text'].lower())
                glosses.append(sample['gloss'])
                langs.append(sample['lang'])
                
                # Only collect in-context examples if enabled
                if self.use_in_context and self.num_in_context > 0:
                    _ex_lang_trans = [
                        f"{sample.get('en_text', '')}={sample['text']}",
                        f"{sample.get('fr_text', '')}={sample['text']}",
                        f"{sample.get('es_text', '')}={sample['text']}"
                    ]
                    _ex_lang_trans = _ex_lang_trans[:self.num_in_context]
                    ex_lang_translations.append(' '.join(_ex_lang_trans))
                else:
                    ex_lang_translations.append('')
                
                # Handle too long sequences with random cropping
                if nframe > max_frame_len:
                    nframe = max_frame_len
                    start_index = random.randint(0, pval.size(0) - max_frame_len)
                    pval = pval[start_index:start_index + max_frame_len]
                
                # Store processed visual features
                num_frames.append(nframe)
                pixel_values.append(pval)
                
                # Process glor values if available
                if sample['glor_value'] is not None:
                    if isinstance(sample['glor_value'], list):
                        glor_values.append(torch.cat(sample['glor_value'], dim=0))
                        glor_lengths.append(sum(len(g) for g in sample['glor_value']))
                    else:
                        glor_values.append(sample['glor_value'])
                        glor_lengths.append(len(sample['glor_value']))
        
        # Only apply derangement if in-context learning is enabled and we have valid examples
        if self.use_in_context and self.num_in_context > 0 and len(ex_lang_translations) > 1:
            if len(set(ex_lang_translations)) > 1:  # Check if there are different examples
                ex_lang_translations = derangement(ex_lang_translations)

        # Check for empty batch (all samples had missing features)
        if len(pixel_values) == 0:
            print(f"Warning: All samples in batch have missing features. Batch IDs: {[s.get('id', 'unknown') for s in batch]}")
            return None

        # Return structured dictionary
        return {
            'pixel_values': pixel_values,
            'glor_values': glor_values,
            'bool_mask_pos': masks,
            'ids': ids,
            'text': texts,
            'ex_lang_trans': ex_lang_translations,
            'gloss': glosses,
            'lang': langs,
            'num_frames': num_frames,
            'glor_lengths': glor_lengths,
        }

    def visual_textual_align(
        self,
        visual_outputs: torch.Tensor,
        visual_masks: torch.Tensor,
        samples: Dict,
        return_details: bool = False
    ) -> torch.Tensor:
        """
        Calculate visual-textual alignment loss and return alignment matrix for reordering.

        Args:
            visual_outputs: Visual features [B, M, D] - sequence of window features
            visual_masks: Mask for visual features [B, M]
            samples: Input samples containing 'text'
            return_details: If True, return (loss, info_dict) tuple

        Returns:
            If return_details=True: (loss, info_dict) where info_dict contains 'alignment' and 'text_mask'
            Otherwise: just loss
        """
        # Tokenize target texts
        output_tokens = self.t5_tokenizer(
            samples['text'],
            padding="longest",
            return_tensors="pt",
        ).to(self.device)

        # Get text embeddings from LLM embedding layer (not hidden states)
        # text_seq: [B, U, D] - token-level embeddings
        with torch.no_grad():
            text_seq = self.t5_model.encoder.embed_tokens(output_tokens.input_ids)
        text_mask = output_tokens.attention_mask  # [B, U]

        # Keep sign sequence for local alignment: [B, M, D]
        sign_seq = visual_outputs
        sign_mask = visual_masks

        # Initialize info dict for logging
        info_dict = {
            'text_mask': text_mask,  # Return text_mask for reordering
        }

        # ========== Global Alignment ==========
        #   L_global = lambda_g(t) * L_EMD + beta_orth * L_orth,   L_orth = ||T^T T - I||_F^2
        #
        # Three-stage schedule:
        # - Stage 0 (0 ~ warm_up_steps): T exists and shapes the local OT cost, but
        #   lambda_g = 0, so only the orthogonality constraint acts on T.
        # - Stage 1 (warm_up_steps ~ + emd_ramp_steps): lambda_g ramps 0 -> lambda_g_max.
        # - Stage 2 (after): lambda_g held at lambda_g_max.
        #
        # The global loss updates ONLY T: sign/text sequences are detached before pooling.
        T = self.local_align_module.T
        orth_loss = compute_orth_loss(T)

        warm_up_steps = self.warm_up_steps if self.warm_up_steps is not None else 0
        if self.global_emd_enabled and self.local_align_enabled:
            lambda_g = compute_lambda_g_schedule(
                current_step=self.global_step,
                warm_up_steps=warm_up_steps,
                emd_ramp_steps=self.emd_ramp_steps,
                lambda_g_max=self.lambda_g_max,
                schedule_type=self.emd_schedule_type,
            )
        else:
            lambda_g = 0.0

        # Sentence vectors via attention pooling, on DETACHED sequences so that the
        # global objective only ever updates T.
        sign_norm_detached = F.normalize(sign_seq.detach(), dim=-1)   # [B, M, D]
        text_norm_detached = F.normalize(text_seq.detach(), dim=-1)   # [B, U, D]

        sign_sentence_vecs = attention_pool_sequence(
            sign_norm_detached,
            mask=sign_mask.float() if sign_mask is not None else None
        )  # [B, D], L2 normalized
        text_sentence_vecs = attention_pool_sequence(
            text_norm_detached,
            mask=text_mask.float() if text_mask is not None else None
        )  # [B, D], L2 normalized

        if self.training and self.global_emd_enabled and self._global_queue is not None:
            self._global_queue.push(sign_sentence_vecs, text_sentence_vecs)

        # Procrustes initialization of T at the Stage0 -> Stage1 transition (one-time).
        if (self.global_emd_enabled and self.local_align_enabled and
                not self._procrustes_inited and
                warm_up_steps > 0 and
                int(self.global_step) == int(warm_up_steps)):

            with torch.no_grad():
                queue_sign, queue_text = self._global_queue.get_all(sign_seq.device)
                if queue_sign is not None and len(queue_sign) >= self.global_queue_min_size:
                    Xg_proc, Yg_proc = queue_sign, queue_text
                else:
                    Xg_proc, Yg_proc = sign_sentence_vecs, text_sentence_vecs

                N = Xg_proc.shape[0]
                sim_proc = torch.mm(Xg_proc, Yg_proc.t())          # [N, N]
                cost_proc = (1.0 - sim_proc).unsqueeze(0)          # [1, N, N]
                a_proc = torch.ones(1, N, device=Xg_proc.device, dtype=Xg_proc.dtype) / N
                b_proc = torch.ones(1, N, device=Xg_proc.device, dtype=Xg_proc.dtype) / N

                P_proc = sinkhorn(cost_proc, a=a_proc, b=b_proc,
                                  eps=self.emd_eps, n_iters=self.emd_iters).squeeze(0)
                T_init = procrustes_init_from_ot_plan(Xg_proc, Yg_proc, P_proc)
                self.local_align_module.T.data.copy_(T_init)

                print(f"[Procrustes Init] Step {self.global_step}: initialized T (N={N}), "
                      f"det={torch.det(T_init).item():.4f}, "
                      f"orth={compute_orth_loss(T_init).item():.6f}")

            self._procrustes_inited = True

        # Global EMD over the queue
        queue_size = len(self._global_queue) if self._global_queue is not None else 0
        info_dict['queue_size'] = queue_size

        emd_loss_raw = torch.tensor(0.0, device=sign_seq.device, dtype=sign_seq.dtype)
        if lambda_g > 0 and self.local_align_enabled:
            queue_sign, queue_text = self._global_queue.get_all(sign_seq.device)
            if queue_sign is not None and len(queue_sign) >= self.global_queue_min_size:
                # Cost: C_ij = 1 - cos((v_i @ T), s_j)
                emd_loss_raw, _ = compute_global_emd_loss_sentence(
                    sign_vecs=queue_sign,
                    text_vecs=queue_text,
                    T=T,
                    eps=self.emd_eps,
                    n_iters=self.emd_iters,
                )
                global_loss = lambda_g * emd_loss_raw + self.beta_orth * orth_loss
            else:
                global_loss = self.beta_orth * orth_loss
        else:
            # Stage 0: orthogonality constraint only
            global_loss = self.beta_orth * orth_loss

        info_dict['global_loss'] = global_loss.detach()
        info_dict['emd_loss_raw'] = emd_loss_raw.detach()
        info_dict['orth_loss'] = orth_loss.detach()
        info_dict['lambda_g'] = lambda_g

        # ========== Local Alignment (Token↔Window with NULL) ==========
        if self.local_align_enabled:
            # Compute local alignment loss using Sinkhorn
            local_loss, local_info = self.local_align_module(
                sign_seq=sign_seq,
                text_seq=text_seq,
                sign_mask=sign_mask.float() if sign_mask is not None else None,
                text_mask=text_mask.float(),
            )

            # Combine losses: global + local (local already includes TV + null_reg loss weighted)
            total_loss = global_loss + local_loss

            # Update info dict with new statistics
            info_dict['local_loss'] = local_info['local_loss']
            info_dict['tv_loss'] = local_info['tv_loss']
            info_dict['null_reg_loss'] = local_info['null_reg_loss']
            info_dict['null_ratio'] = local_info['null_ratio']  # Actual ratio (argmax based)
            info_dict['soft_null_ratio'] = local_info['soft_null_ratio']  # Soft ratio
            info_dict['null_bias'] = local_info['null_bias']  # Learned bias
            info_dict['total_vt_loss'] = total_loss.detach()
            info_dict['alignment'] = local_info['alignment']  # [B, M, K] alignment matrix for reordering
        else:
            total_loss = global_loss
            info_dict['total_vt_loss'] = total_loss.detach()
            info_dict['alignment'] = None  # No alignment if local align disabled

        if return_details:
            return total_loss, info_dict
        return total_loss

    def on_train_epoch_start(self) -> None:
        """
        Called at the start of each training epoch. Updates the OT entropy (eps).

        Two schedules are available:
        - use_three_phase_eps=True: the paper's piecewise schedule, eps_high held
          through phase 1, annealed eps_high -> eps_mid over eps_phase2_epochs, then
          eps_mid -> eps_low over eps_phase3_epochs.
        - use_three_phase_eps=False: the legacy single-phase linear decay from
          local_align_eps_start to local_align_eps_end.
        """
        if self.use_three_phase_eps:
            # Driven per-step in on_train_batch_start(); nothing to do per-epoch.
            return

        if self.local_align_enabled and hasattr(self, 'local_align_module'):
            epoch = self.current_epoch

            if epoch < self.local_align_eps_decay_epochs:
                # Linear interpolation
                progress = epoch / self.local_align_eps_decay_epochs
                current_eps = self.local_align_eps_start - progress * (self.local_align_eps_start - self.local_align_eps_end)
            else:
                current_eps = self.local_align_eps_end

            # Update the eps in LocalAlignmentModule
            self.local_align_module.eps = current_eps
            print(f"[Epoch {epoch}] OT eps updated: {current_eps:.4f}")

    def on_train_batch_start(self, batch, batch_idx) -> None:
        """
        Per-step driver for the paper's three-phase epsilon annealing.

        Phase 1 (0 ~ warm_up_steps):            eps = eps_high (soft, stable alignment)
        Phase 2 (next eps_phase2_epochs):       eps_high -> eps_mid
        Phase 3 (next eps_phase3_epochs):       eps_mid  -> eps_low
        Phase 4 (rest):                         eps = eps_low (sharp)

        Only active when use_three_phase_eps=True; otherwise the legacy per-epoch
        decay in on_train_epoch_start() applies.
        """
        if not (self.use_three_phase_eps and self.local_align_enabled
                and hasattr(self, 'local_align_module')):
            return

        # Cache the trainer-derived step counts once.
        if self._total_steps is None and self.trainer is not None:
            try:
                self._total_steps = self.trainer.estimated_stepping_batches
                accum = getattr(self.trainer, 'accumulate_grad_batches', 1) or 1
                if getattr(self.trainer, 'num_training_batches', None):
                    self._steps_per_epoch = self.trainer.num_training_batches // accum
                else:
                    max_epochs = self.trainer.max_epochs or 1
                    self._steps_per_epoch = self._total_steps // max(1, max_epochs)
            except Exception:
                return

        if not self._total_steps or not self._steps_per_epoch:
            return

        warm_up_steps = self.warm_up_steps if self.warm_up_steps is not None else 0
        self.local_align_module.eps = compute_eps_schedule(
            current_step=self.global_step,
            total_steps=self._total_steps,
            warm_up_steps=warm_up_steps,
            phase2_steps=self.eps_phase2_epochs * self._steps_per_epoch,
            phase3_steps=self.eps_phase3_epochs * self._steps_per_epoch,
            eps_high=self.eps_high,
            eps_mid=self.eps_mid,
            eps_low=self.eps_low,
            schedule_type=self.eps_schedule_type,
        )

    def shared_step(self, inputs: Dict, split: str, batch_idx: int) -> Tuple[torch.Tensor, Dict]:
        """
        Shared logic for training, validation and testing steps.

        Args:
            inputs: Input dictionary
            split: Current split (train, val, test)
            batch_idx: Current batch index

        Returns:
            Tuple of (loss, log_dict)
        """
        # Prepare visual inputs and project to match text embedding dimensions
        visual_outputs, visual_masks = self.prepare_visual_inputs(inputs)
        visual_outputs = self.fusion_proj(visual_outputs)
        
        # Initialize logging dictionary
        log_dict = {}
        if self.attn_log_stats and self._last_attn_stats is not None:
            for key, val in self._last_attn_stats.items():
                if val is not None:
                    log_dict[f"{split}/{key}"] = val
        
        # STEP 1: Determine training mode and prepare inputs accordingly
        if self.cross_modal_align:
            # For pure contrastive learning or warm-up phase
            if self.warm_up_steps is None and not self.combined_loss:
                # Pure contrastive learning mode
                with torch.no_grad():
                    input_embeds, input_masks, output_tokens, targets = self.prepare_inputs(
                        visual_outputs, visual_masks, inputs, split, batch_idx
                    )
                
                cont_loss, vt_info = self.visual_textual_align(visual_outputs, visual_masks, inputs, return_details=True)
                log_dict[f"{split}/contra_loss"] = cont_loss
                log_dict[f"{split}/vt_global_loss"] = vt_info['global_loss']
                log_dict[f"{split}/vt_total_loss"] = vt_info['total_vt_loss']
                if self.local_align_enabled:
                    log_dict[f"{split}/vt_local_loss"] = vt_info['local_loss']
                    log_dict[f"{split}/vt_tv_loss"] = vt_info['tv_loss']
                    log_dict[f"{split}/vt_null_reg_loss"] = vt_info['null_reg_loss']
                    log_dict[f"{split}/vt_null_ratio"] = vt_info['null_ratio']
                    log_dict[f"{split}/vt_soft_null_ratio"] = vt_info['soft_null_ratio']
                    log_dict[f"{split}/vt_null_bias"] = vt_info['null_bias']
                # Warm-up: enable position contrastive from the start
                pos_contra_loss = torch.tensor(0.0, device=self.device, dtype=visual_outputs.dtype)
                pos_contra_stats = None
                if self.local_align_enabled and vt_info['alignment'] is not None:
                    text_mask = vt_info['text_mask']
                    num_text_tokens = text_mask.shape[1]
                    reordered_outputs, _, _ = reorder_by_window_alignment(
                        visual_outputs,
                        vt_info['alignment'],
                        num_text_tokens,
                        sign_mask=visual_masks.float() if visual_masks is not None else None,
                        text_mask=text_mask.float(),
                    )
                    if self.position_contrastive_enabled and self.training:
                        output_tokens_for_contra = self.t5_tokenizer(
                            inputs['text'],
                            padding="longest",
                            return_tensors="pt",
                        ).to(self.device)
                        with torch.no_grad():
                            text_feats = self.t5_model.encoder.embed_tokens(output_tokens_for_contra.input_ids)
                        text_mask_for_contra = output_tokens_for_contra.attention_mask
                        contra_window_feats = reordered_outputs
                        contra_text_feats = text_feats.detach()
                        if self.position_contra_use_proj:
                            contra_window_feats = self.pos_contra_vis_proj(contra_window_feats)
                            contra_text_feats = self.pos_contra_txt_proj(contra_text_feats)
                        pos_contra_loss, pos_contra_stats = compute_position_contrastive_loss(
                            window_feats=contra_window_feats,
                            text_feats=contra_text_feats,
                            text_mask=text_mask_for_contra.float(),
                            temperature=self.position_contra_temp,
                            return_stats=True,
                        )
                log_dict[f"{split}/pos_contra_loss"] = pos_contra_loss
                if pos_contra_stats is not None:
                    log_dict[f"{split}/pos_contra_pos_sim"] = pos_contra_stats['pos_sim_mean']
                    log_dict[f"{split}/pos_contra_neg_sim"] = pos_contra_stats['neg_sim_mean']
                    log_dict[f"{split}/pos_contra_gap"] = pos_contra_stats['pos_neg_gap']
                loss = cont_loss + self.beta_pos_contra * pos_contra_loss

            elif self.warm_up_steps is not None and self.global_step <= self.warm_up_steps:
                # Warm-up phase: train everything EXCEPT T5 LoRA
                # - temporal_encoder: trains via contrastive + position contrastive gradients
                # - contrastive alignment (OT): trains
                # - position contrastive: trains (added in this version)
                # - T5 LoRA: frozen (torch.no_grad on prepare_inputs)
                with torch.no_grad():
                    input_embeds, input_masks, output_tokens, targets = self.prepare_inputs(
                        visual_outputs, visual_masks, inputs, split, batch_idx
                    )

                cont_loss, vt_info = self.visual_textual_align(visual_outputs, visual_masks, inputs, return_details=True)
                log_dict[f"{split}/contra_loss"] = cont_loss
                log_dict[f"{split}/vt_global_loss"] = vt_info['global_loss']
                log_dict[f"{split}/vt_total_loss"] = vt_info['total_vt_loss']
                if self.local_align_enabled:
                    log_dict[f"{split}/vt_local_loss"] = vt_info['local_loss']
                    log_dict[f"{split}/vt_tv_loss"] = vt_info['tv_loss']
                    log_dict[f"{split}/vt_null_reg_loss"] = vt_info['null_reg_loss']
                    log_dict[f"{split}/vt_null_ratio"] = vt_info['null_ratio']
                    log_dict[f"{split}/vt_soft_null_ratio"] = vt_info['soft_null_ratio']
                    log_dict[f"{split}/vt_null_bias"] = vt_info['null_bias']

                # Position contrastive during warmup (NEW: train from the start)
                pos_contra_loss = torch.tensor(0.0, device=self.device, dtype=visual_outputs.dtype)
                pos_contra_stats = None
                if self.local_align_enabled and vt_info['alignment'] is not None:
                    text_mask = vt_info['text_mask']
                    num_text_tokens = text_mask.shape[1]
                    reordered_outputs, _, _ = reorder_by_window_alignment(
                        visual_outputs,
                        vt_info['alignment'],
                        num_text_tokens,
                        sign_mask=visual_masks.float() if visual_masks is not None else None,
                        text_mask=text_mask.float(),
                    )
                    if self.position_contrastive_enabled and self.training:
                        output_tokens_for_contra = self.t5_tokenizer(
                            inputs['text'],
                            padding="longest",
                            return_tensors="pt",
                        ).to(self.device)
                        with torch.no_grad():
                            text_feats = self.t5_model.encoder.embed_tokens(output_tokens_for_contra.input_ids)
                        text_mask_for_contra = output_tokens_for_contra.attention_mask
                        contra_window_feats = reordered_outputs
                        contra_text_feats = text_feats.detach()
                        if self.position_contra_use_proj:
                            contra_window_feats = self.pos_contra_vis_proj(contra_window_feats)
                            contra_text_feats = self.pos_contra_txt_proj(contra_text_feats)
                        pos_contra_loss, pos_contra_stats = compute_position_contrastive_loss(
                            window_feats=contra_window_feats,
                            text_feats=contra_text_feats,
                            text_mask=text_mask_for_contra.float(),
                            temperature=self.position_contra_temp,
                            return_stats=True,
                        )
                log_dict[f"{split}/pos_contra_loss"] = pos_contra_loss
                if pos_contra_stats is not None:
                    log_dict[f"{split}/pos_contra_pos_sim"] = pos_contra_stats['pos_sim_mean']
                    log_dict[f"{split}/pos_contra_neg_sim"] = pos_contra_stats['neg_sim_mean']
                    log_dict[f"{split}/pos_contra_gap"] = pos_contra_stats['pos_neg_gap']
                loss = cont_loss + self.beta_pos_contra * pos_contra_loss

            else:
                # Combined loss mode (regular training + contrastive)
                # STEP 1: Compute OT alignment at FRAME level (same as _order)
                cont_loss, vt_info = self.visual_textual_align(visual_outputs, visual_masks, inputs, return_details=True)

                # STEP 2: REORDER using WINDOW-based aggregation (KEY CHANGE in _windownomean!)
                # - Alignment is still at frame level (for loss computation)
                # - But reordering groups frames into windows and reorders at window level
                if self.local_align_enabled and vt_info['alignment'] is not None:
                    # Get number of text tokens from text_mask
                    text_mask = vt_info['text_mask']
                    # num_text_tokens is the sequence length (U)
                    num_text_tokens = text_mask.shape[1]

                    # Use window-based reordering (NO mean pooling)
                    # 1. Group frames into U+2 windows
                    # 2. Sum frame probabilities within each window
                    # 3. Assign each window to best token
                    # 4. Reorder at window level
                    reordered_outputs, reorder_mask, window_info = reorder_by_window_alignment(
                        visual_outputs,           # Frame features [B, M, D]
                        vt_info['alignment'],     # Frame-level alignment [B, M, K]
                        num_text_tokens,          # U (for computing num_windows = U+2)
                        sign_mask=visual_masks.float() if visual_masks is not None else None,
                        text_mask=text_mask.float(),
                    )
                    # Use reordered features for LLM input
                    llm_visual_outputs = reordered_outputs
                    llm_visual_masks = reorder_mask

                    # === CONTRASTIVE version: Position-Aligned Contrastive Loss ===
                    pos_contra_loss = torch.tensor(0.0, device=self.device, dtype=visual_outputs.dtype)
                    pos_contra_stats = None

                    if self.position_contrastive_enabled and self.training:
                        # Get text embeddings (detached to prevent gradient flow to T5 LoRA)
                        output_tokens_for_contra = self.t5_tokenizer(
                            inputs['text'],
                            padding="longest",
                            return_tensors="pt",
                        ).to(self.device)
                        with torch.no_grad():
                            text_feats = self.t5_model.encoder.embed_tokens(output_tokens_for_contra.input_ids)
                        text_mask_for_contra = output_tokens_for_contra.attention_mask

                        # Compute position-aligned contrastive loss
                        contra_window_feats = reordered_outputs
                        contra_text_feats = text_feats.detach()
                        if self.position_contra_use_proj:
                            contra_window_feats = self.pos_contra_vis_proj(contra_window_feats)
                            contra_text_feats = self.pos_contra_txt_proj(contra_text_feats)
                        pos_contra_loss, pos_contra_stats = compute_position_contrastive_loss(
                            window_feats=contra_window_feats,       # Has gradient
                            text_feats=contra_text_feats,          # Explicitly detached before proj
                            text_mask=text_mask_for_contra.float(),
                            temperature=self.position_contra_temp,
                            return_stats=True,
                        )
                else:
                    # Fallback to original order if no alignment
                    llm_visual_outputs = visual_outputs
                    llm_visual_masks = visual_masks
                    pos_contra_loss = torch.tensor(0.0, device=self.device, dtype=visual_outputs.dtype)
                    pos_contra_stats = None

                # STEP 3: Prepare inputs with REORDERED features
                input_embeds, input_masks, output_tokens, targets = self.prepare_inputs(
                    llm_visual_outputs, llm_visual_masks, inputs, split, batch_idx
                )

                # Forward pass through T5 model
                outputs = self.t5_model(
                    inputs_embeds=input_embeds,
                    attention_mask=input_masks,
                    decoder_attention_mask=output_tokens.attention_mask,
                    labels=targets,
                    output_hidden_states=True,
                    return_dict=True
                )

                t5_loss = outputs.loss
                main_loss = t5_loss
                log_dict[f"{split}/loss"] = t5_loss

                if self.distill_enabled and split == "train":
                    self._ensure_teacher_on_device()
                    with torch.no_grad():
                        teacher_visual_outputs, teacher_visual_masks = self._teacher.prepare_visual_inputs(inputs)
                        teacher_visual_outputs = self._teacher.fusion_proj(teacher_visual_outputs)
                        teacher_input_embeds, teacher_input_masks, _, _ = self._teacher.prepare_inputs(
                            teacher_visual_outputs, teacher_visual_masks, inputs, split, batch_idx
                        )
                        teacher_outputs = self._teacher.t5_model(
                            inputs_embeds=teacher_input_embeds,
                            attention_mask=teacher_input_masks,
                            decoder_attention_mask=output_tokens.attention_mask,
                            labels=targets,
                            output_hidden_states=False,
                            return_dict=True,
                        )
                    kd_loss = self._compute_kd_loss(
                        outputs.logits,
                        teacher_outputs.logits,
                        output_tokens.attention_mask
                    )
                    main_loss = (1.0 - self.distill_alpha) * t5_loss + self.distill_alpha * kd_loss
                    log_dict[f"{split}/kd_loss"] = kd_loss
                    log_dict[f"{split}/distill_loss"] = main_loss

                # Combine losses (including position contrastive loss)
                loss = main_loss + self.alpha * cont_loss + self.beta_pos_contra * pos_contra_loss

                log_dict[f"{split}/contra_loss"] = cont_loss
                log_dict[f"{split}/vt_global_loss"] = vt_info['global_loss']
                log_dict[f"{split}/vt_total_loss"] = vt_info['total_vt_loss']
                # CONTRASTIVE version: Log position contrastive loss
                log_dict[f"{split}/pos_contra_loss"] = pos_contra_loss
                if pos_contra_stats is not None:
                    log_dict[f"{split}/pos_contra_pos_sim"] = pos_contra_stats['pos_sim_mean']
                    log_dict[f"{split}/pos_contra_neg_sim"] = pos_contra_stats['neg_sim_mean']
                    log_dict[f"{split}/pos_contra_gap"] = pos_contra_stats['pos_neg_gap']
                if self.local_align_enabled:
                    log_dict[f"{split}/vt_local_loss"] = vt_info['local_loss']
                    log_dict[f"{split}/vt_tv_loss"] = vt_info['tv_loss']
                    log_dict[f"{split}/vt_null_reg_loss"] = vt_info['null_reg_loss']
                    log_dict[f"{split}/vt_null_ratio"] = vt_info['null_ratio']
                    log_dict[f"{split}/vt_soft_null_ratio"] = vt_info['soft_null_ratio']
                    log_dict[f"{split}/vt_null_bias"] = vt_info['null_bias']
                log_dict[f"{split}/combined_loss"] = loss
        else:
            # Standard training without contrastive learning
            input_embeds, input_masks, output_tokens, targets = self.prepare_inputs(
                visual_outputs, visual_masks, inputs, split, batch_idx
            )
            
            # Forward pass through T5 model
            outputs = self.t5_model(
                inputs_embeds=input_embeds,
                attention_mask=input_masks,
                decoder_attention_mask=output_tokens.attention_mask,
                labels=targets,
                output_hidden_states=True,
                return_dict=True
            )
            
            t5_loss = outputs.loss
            loss = t5_loss
            log_dict[f"{split}/loss"] = t5_loss

            if self.distill_enabled and split == "train":
                self._ensure_teacher_on_device()
                with torch.no_grad():
                    teacher_visual_outputs, teacher_visual_masks = self._teacher.prepare_visual_inputs(inputs)
                    teacher_visual_outputs = self._teacher.fusion_proj(teacher_visual_outputs)
                    teacher_input_embeds, teacher_input_masks, _, _ = self._teacher.prepare_inputs(
                        teacher_visual_outputs, teacher_visual_masks, inputs, split, batch_idx
                    )
                    teacher_outputs = self._teacher.t5_model(
                        inputs_embeds=teacher_input_embeds,
                        attention_mask=teacher_input_masks,
                        decoder_attention_mask=output_tokens.attention_mask,
                        labels=targets,
                        output_hidden_states=False,
                        return_dict=True,
                    )
                kd_loss = self._compute_kd_loss(
                    outputs.logits,
                    teacher_outputs.logits,
                    output_tokens.attention_mask
                )
                loss = (1.0 - self.distill_alpha) * t5_loss + self.distill_alpha * kd_loss
                log_dict[f"{split}/kd_loss"] = kd_loss
                log_dict[f"{split}/distill_loss"] = loss

        # STEP 2: Handle evaluation phase (validation/testing)
        if split != "train":
            # ============================================================
            # Use GT text to compute alignment and reorder features for val
            # ============================================================
            # We use GT text to compute OT alignment matrix and reorder
            # visual features, then evaluate BLEU4 on reordered features.
            # This allows us to measure how well the reordering helps.
            # ============================================================
            _, eval_vt_info = self.visual_textual_align(visual_outputs, visual_masks, inputs, return_details=True)

            if self.local_align_enabled and eval_vt_info['alignment'] is not None:
                # Get number of text tokens from text_mask
                text_mask = eval_vt_info['text_mask']
                num_text_tokens = text_mask.shape[1]

                # Use window-based reordering (same as training)
                eval_reordered_outputs, eval_reorder_mask, _ = reorder_by_window_alignment(
                    visual_outputs,           # Frame features [B, M, D]
                    eval_vt_info['alignment'],     # Frame-level alignment [B, M, K]
                    num_text_tokens,          # U (for computing num_windows = U+2)
                    sign_mask=visual_masks.float() if visual_masks is not None else None,
                    text_mask=text_mask.float(),
                )
                eval_visual_outputs = eval_reordered_outputs
                eval_visual_masks = eval_reorder_mask
            else:
                # Fallback to original if no alignment
                eval_visual_outputs = visual_outputs
                eval_visual_masks = visual_masks

            # Prepare inputs for text generation with reordered features
            input_embeds, input_masks, _, _ = self.prepare_inputs(
                eval_visual_outputs, eval_visual_masks, inputs, split, batch_idx
            )

            # Generate translations
            generated = self.t5_model.generate(
                inputs_embeds=input_embeds,
                attention_mask=input_masks,
                num_beams=5,
                max_length=self.max_txt_len,
                top_p=0.9,
                do_sample=True,
                no_repeat_ngram_size=3,  # Prevent 3-gram repetition
                repetition_penalty=1.2,  # Penalize repeated tokens
            )
            
            # Decode generated outputs and references
            generated_strings = self.t5_tokenizer.batch_decode(generated, skip_special_tokens=True)
            generated_strings = [gen.lower() for gen in generated_strings]
            
            reference_strings = self.t5_tokenizer.batch_decode(output_tokens.input_ids, skip_special_tokens=True)
            reference_strings = [ref.lower() for ref in reference_strings]

            self.generated.extend(generated_strings)
            self.references.extend(reference_strings)
            # Accumulate validation loss
            self.val_losses.append(loss.detach())

        # Accumulate training loss and details
        if split == "train":
            self.train_losses.append(loss.detach())
            # Accumulate detailed losses from log_dict
            if f"{split}/loss" in log_dict:
                self.train_loss_details['t5_loss'].append(log_dict[f"{split}/loss"].detach())
            if f"{split}/contra_loss" in log_dict:
                self.train_loss_details['contra_loss'].append(log_dict[f"{split}/contra_loss"].detach())
            if f"{split}/vt_global_loss" in log_dict:
                self.train_loss_details['vt_global_loss'].append(log_dict[f"{split}/vt_global_loss"].detach())
            if f"{split}/vt_local_loss" in log_dict:
                self.train_loss_details['vt_local_loss'].append(log_dict[f"{split}/vt_local_loss"].detach())
            if f"{split}/vt_tv_loss" in log_dict:
                self.train_loss_details['tv_loss'].append(log_dict[f"{split}/vt_tv_loss"].detach())
            if f"{split}/vt_null_reg_loss" in log_dict:
                self.train_loss_details['null_reg_loss'].append(log_dict[f"{split}/vt_null_reg_loss"].detach())
            if f"{split}/kd_loss" in log_dict:
                self.train_loss_details['kd_loss'].append(log_dict[f"{split}/kd_loss"].detach())
            if f"{split}/combined_loss" in log_dict:
                self.train_loss_details['combined_loss'].append(log_dict[f"{split}/combined_loss"].detach())
            if f"{split}/vt_null_ratio" in log_dict:
                self.train_loss_details['null_ratio'].append(log_dict[f"{split}/vt_null_ratio"])
            if f"{split}/vt_soft_null_ratio" in log_dict:
                self.train_loss_details['soft_null_ratio'].append(log_dict[f"{split}/vt_soft_null_ratio"])
            if f"{split}/vt_null_bias" in log_dict:
                self.train_loss_details['null_bias'].append(log_dict[f"{split}/vt_null_bias"])
            # CONTRASTIVE version: Collect position contrastive loss
            if f"{split}/pos_contra_loss" in log_dict:
                val = log_dict[f"{split}/pos_contra_loss"]
                self.train_loss_details['pos_contra_loss'].append(val.detach() if torch.is_tensor(val) else val)
            if f"{split}/pos_contra_pos_sim" in log_dict:
                val = log_dict[f"{split}/pos_contra_pos_sim"]
                self.train_loss_details['pos_contra_pos_sim'].append(val.detach() if torch.is_tensor(val) else val)
            if f"{split}/pos_contra_neg_sim" in log_dict:
                val = log_dict[f"{split}/pos_contra_neg_sim"]
                self.train_loss_details['pos_contra_neg_sim'].append(val.detach() if torch.is_tensor(val) else val)
            if f"{split}/pos_contra_gap" in log_dict:
                val = log_dict[f"{split}/pos_contra_gap"]
                self.train_loss_details['pos_contra_gap'].append(val.detach() if torch.is_tensor(val) else val)
            if f"{split}/attn_peak" in log_dict:
                val = log_dict[f"{split}/attn_peak"]
                self.train_loss_details['attn_peak'].append(val.detach() if torch.is_tensor(val) else val)
            if f"{split}/attn_entropy" in log_dict:
                val = log_dict[f"{split}/attn_entropy"]
                self.train_loss_details['attn_entropy'].append(val.detach() if torch.is_tensor(val) else val)
            if f"{split}/attn_peak_l1" in log_dict:
                val = log_dict[f"{split}/attn_peak_l1"]
                self.train_loss_details['attn_peak_l1'].append(val.detach() if torch.is_tensor(val) else val)
            if f"{split}/attn_entropy_l1" in log_dict:
                val = log_dict[f"{split}/attn_entropy_l1"]
                self.train_loss_details['attn_entropy_l1'].append(val.detach() if torch.is_tensor(val) else val)
            if f"{split}/attn_peak_l2" in log_dict:
                val = log_dict[f"{split}/attn_peak_l2"]
                self.train_loss_details['attn_peak_l2'].append(val.detach() if torch.is_tensor(val) else val)
            if f"{split}/attn_entropy_l2" in log_dict:
                val = log_dict[f"{split}/attn_entropy_l2"]
                self.train_loss_details['attn_entropy_l2'].append(val.detach() if torch.is_tensor(val) else val)

        return loss, log_dict

    def on_train_epoch_end(self) -> None:
        # Print training loss summary for this epoch
        if len(self.train_losses) > 0:
            import torch
            avg_train_loss = torch.stack(self.train_losses).mean().item()

            print("\n" + "=" * 60)
            print(f"===== Training Summary (Epoch {self.current_epoch}) =====")
            print("=" * 60)
            print(f"  Total Loss: {avg_train_loss:.4f}")
            print("-" * 40)

            # Print detailed losses
            details = self.train_loss_details

            # T5 Loss (seq2seq)
            if len(details['t5_loss']) > 0:
                avg_t5 = torch.stack(details['t5_loss']).mean().item()
                print(f"  T5 Loss (seq2seq):     {avg_t5:.4f}")

            # Contrastive Loss
            if len(details['contra_loss']) > 0:
                avg_contra = torch.stack(details['contra_loss']).mean().item()
                print(f"  Contrastive Loss:      {avg_contra:.4f}")

            # VT Global Loss
            if len(details['vt_global_loss']) > 0:
                avg_global = torch.stack(details['vt_global_loss']).mean().item()
                print(f"  VT Global Loss:        {avg_global:.4f}")

            # VT Local Loss (if local alignment enabled)
            if len(details['vt_local_loss']) > 0:
                avg_local = torch.stack(details['vt_local_loss']).mean().item()
                print(f"  VT Local Loss (OT):    {avg_local:.4f}")

            # TV Loss (temporal smoothness)
            if len(details['tv_loss']) > 0:
                avg_tv = torch.stack(details['tv_loss']).mean().item()
                print(f"  TV Loss (smoothness):  {avg_tv:.4f}")

            # NULL Reg Loss
            if len(details['null_reg_loss']) > 0:
                avg_null_reg = torch.stack(details['null_reg_loss']).mean().item()
                print(f"  NULL Reg Loss:         {avg_null_reg:.4f}")

            # KD Loss (knowledge distillation)
            if len(details['kd_loss']) > 0:
                avg_kd = torch.stack(details['kd_loss']).mean().item()
                print(f"  KD Loss (distill):     {avg_kd:.4f}")

            # Combined Loss
            if len(details['combined_loss']) > 0:
                avg_combined = torch.stack(details['combined_loss']).mean().item()
                print(f"  Combined Loss:         {avg_combined:.4f}")

            # NULL statistics (if local alignment enabled)
            if len(details['null_ratio']) > 0:
                avg_null = sum(details['null_ratio']) / len(details['null_ratio'])
                print("-" * 40)
                print(f"  NULL Ratio (argmax):   {avg_null:.4f}  <- Target: {self.null_ratio_target:.2f}")

            if len(details['soft_null_ratio']) > 0:
                avg_soft_null = sum(details['soft_null_ratio']) / len(details['soft_null_ratio'])
                print(f"  Soft NULL Ratio:       {avg_soft_null:.4f}")

            if len(details['null_bias']) > 0:
                avg_null_bias = sum(details['null_bias']) / len(details['null_bias'])
                print(f"  NULL Bias (learned):   {avg_null_bias:.4f}")

            # CONTRASTIVE version: Print position contrastive loss
            if len(details['pos_contra_loss']) > 0:
                pos_vals = [v.item() if torch.is_tensor(v) else v for v in details['pos_contra_loss']]
                avg_pos = sum(pos_vals) / len(pos_vals)
                print("-" * 40)
                print(f"  [Position Contrastive]")
                print(f"  Pos Contra Loss:       {avg_pos:.4f}")
                if len(details['pos_contra_pos_sim']) > 0:
                    pos_sim_vals = [v.item() if torch.is_tensor(v) else v for v in details['pos_contra_pos_sim']]
                    avg_pos_sim = sum(pos_sim_vals) / len(pos_sim_vals)
                    print(f"  Pos Sim (mean):        {avg_pos_sim:.4f}")
                if len(details['pos_contra_neg_sim']) > 0:
                    neg_sim_vals = [v.item() if torch.is_tensor(v) else v for v in details['pos_contra_neg_sim']]
                    avg_neg_sim = sum(neg_sim_vals) / len(neg_sim_vals)
                    print(f"  Neg Sim (mean):        {avg_neg_sim:.4f}")
                if len(details['pos_contra_gap']) > 0:
                    gap_vals = [v.item() if torch.is_tensor(v) else v for v in details['pos_contra_gap']]
                    avg_gap = sum(gap_vals) / len(gap_vals)
                    print(f"  Pos-Neg Gap:           {avg_gap:.4f}")
                if len(details['attn_peak']) > 0:
                    # FIX: Convert bf16 tensors to Python floats before summing
                    attn_peak_vals = [v.item() if torch.is_tensor(v) else v for v in details['attn_peak']]
                    avg_attn_peak = sum(attn_peak_vals) / len(attn_peak_vals)
                    print(f"  Attn Peak (mean):      {avg_attn_peak:.4f}")
                if len(details['attn_entropy']) > 0:
                    attn_entropy_vals = [v.item() if torch.is_tensor(v) else v for v in details['attn_entropy']]
                    avg_attn_entropy = sum(attn_entropy_vals) / len(attn_entropy_vals)
                    print(f"  Attn Entropy (mean):   {avg_attn_entropy:.4f}")
                if len(details['attn_peak_l1']) > 0:
                    attn_peak_l1_vals = [v.item() if torch.is_tensor(v) else v for v in details['attn_peak_l1']]
                    avg_attn_peak_l1 = sum(attn_peak_l1_vals) / len(attn_peak_l1_vals)
                    print(f"  Attn Peak L1 (mean):   {avg_attn_peak_l1:.4f}")
                if len(details['attn_entropy_l1']) > 0:
                    attn_entropy_l1_vals = [v.item() if torch.is_tensor(v) else v for v in details['attn_entropy_l1']]
                    avg_attn_entropy_l1 = sum(attn_entropy_l1_vals) / len(attn_entropy_l1_vals)
                    print(f"  Attn Entropy L1:       {avg_attn_entropy_l1:.4f}")
                if len(details['attn_peak_l2']) > 0:
                    attn_peak_l2_vals = [v.item() if torch.is_tensor(v) else v for v in details['attn_peak_l2']]
                    avg_attn_peak_l2 = sum(attn_peak_l2_vals) / len(attn_peak_l2_vals)
                    print(f"  Attn Peak L2 (mean):   {avg_attn_peak_l2:.4f}")
                if len(details['attn_entropy_l2']) > 0:
                    attn_entropy_l2_vals = [v.item() if torch.is_tensor(v) else v for v in details['attn_entropy_l2']]
                    avg_attn_entropy_l2 = sum(attn_entropy_l2_vals) / len(attn_entropy_l2_vals)
                    print(f"  Attn Entropy L2:       {avg_attn_entropy_l2:.4f}")

            # Get current learning rate
            if self.trainer and self.trainer.optimizers:
                current_lr = self.trainer.optimizers[0].param_groups[0]['lr']
                print("-" * 40)
                print(f"  Learning Rate:         {current_lr:.2e}")

            print("=" * 60 + "\n")

            # Clear training losses for next epoch
            self.train_losses = []
            # Reset detailed loss tracking
            for key in self.train_loss_details:
                self.train_loss_details[key] = []

    def on_validation_epoch_end(self) -> None:
        # Print some examples of generated translations and references with colors
        # Changed: Print 10 RANDOM examples instead of first 5 (to avoid always seeing the same samples)
        import random
        print("\n" + "=" * 60)
        print("===== Validation Examples (10 random samples) =====")
        print("=" * 60)
        num_examples = min(10, len(self.generated))
        if len(self.generated) > 10:
            # Use fixed seed so we always see the same 10 examples for comparison across epochs
            rng = random.Random(42)
            sample_indices = rng.sample(range(len(self.generated)), num_examples)
        else:
            sample_indices = list(range(num_examples))
        for idx, i in enumerate(sample_indices):
            print(f"[{idx+1}/{num_examples}] Sample #{i}")
            print(f"\033[94mReference: {self.references[i]}\033[0m")  # Blue color for references
            print(f"\033[92mGenerated: {self.generated[i]}\033[0m")    # Green color for generated
            print("-" * 50)

        # Calculate evaluation metrics
        eval_res = evaluate_results(
            predictions=self.generated,
            references=self.references,
            split='val',
            device=self.device
        )

        # Calculate average validation loss
        import torch
        avg_val_loss = torch.stack(self.val_losses).mean().item() if len(self.val_losses) > 0 else 0.0

        # Print detailed evaluation metrics to terminal
        print("\n" + "=" * 60)
        print(f"===== Validation Metrics (Epoch {self.current_epoch}) =====")
        print("=" * 60)
        print(f"  Validation Loss: {avg_val_loss:.4f}")
        print("-" * 30)
        print(f"  BLEU-1: {eval_res.get('val/bleu1', 0):.2f}")
        print(f"  BLEU-2: {eval_res.get('val/bleu2', 0):.2f}")
        print(f"  BLEU-3: {eval_res.get('val/bleu3', 0):.2f}")
        print(f"  BLEU-4: {eval_res.get('val/bleu4', 0):.2f}")
        print("-" * 30)
        print(f"  ROUGE-L Precision: {eval_res.get('val/rougeL_precision', 0):.4f}")
        print(f"  ROUGE-L Recall:    {eval_res.get('val/rougeL_recall', 0):.4f}")
        print(f"  ROUGE-L F1:        {eval_res.get('val/rougeL_f1', 0):.4f}")
        print("=" * 60 + "\n")

        # Also log validation loss
        eval_res['val/loss'] = avg_val_loss

        # Clear validation losses for next epoch
        self.val_losses = []

        self.log_dict(eval_res, sync_dist=True)

        self.set_container()

    def on_test_epoch_end(self) -> None:
        # Print some examples of generated translations and references with colors
        # Changed: Print 10 RANDOM examples instead of first 5 (to avoid always seeing the same samples)
        import random
        print("\n" + "=" * 60)
        print("===== Test Examples (10 random samples) =====")
        print("=" * 60)
        num_examples = min(10, len(self.generated))
        if len(self.generated) > 10:
            # Use fixed seed so we always see the same 10 examples for comparison
            rng = random.Random(42)
            sample_indices = rng.sample(range(len(self.generated)), num_examples)
        else:
            sample_indices = list(range(num_examples))
        for idx, i in enumerate(sample_indices):
            print(f"[{idx+1}/{num_examples}] Sample #{i}")
            print(f"\033[94mReference: {self.references[i]}\033[0m")  # Blue color for references
            print(f"\033[92mGenerated: {self.generated[i]}\033[0m")    # Green color for generated
            print("-" * 50)

        # Calculate evaluation metrics
        eval_res = evaluate_results(
            predictions=self.generated,
            references=self.references,
            split='test',
            device=self.device
        )

        # Print detailed test metrics to terminal
        print("\n" + "=" * 60)
        print("===== Test Metrics =====")
        print("=" * 60)
        print(f"  BLEU-1: {eval_res.get('test/bleu1', 0):.2f}")
        print(f"  BLEU-2: {eval_res.get('test/bleu2', 0):.2f}")
        print(f"  BLEU-3: {eval_res.get('test/bleu3', 0):.2f}")
        print(f"  BLEU-4: {eval_res.get('test/bleu4', 0):.2f}")
        print("-" * 30)
        print(f"  ROUGE-L Precision: {eval_res.get('test/rougeL_precision', 0):.4f}")
        print(f"  ROUGE-L Recall:    {eval_res.get('test/rougeL_recall', 0):.4f}")
        print(f"  ROUGE-L F1:        {eval_res.get('test/rougeL_f1', 0):.4f}")
        print("=" * 60 + "\n")

        self.log_dict(eval_res, sync_dist=True)
        self.set_container()

    def configure_optimizers(self):
        trainable_params = [p for p in self.parameters() if p.requires_grad]
        optimizer = torch.optim.AdamW(
            trainable_params, 
            lr=self.lr, 
            eps=1e-8, 
            weight_decay=0.01, 
            betas=(0.9, 0.98)
        )
        
        # Calculate total steps based on PyTorch Lightning trainer settings
        if hasattr(self.trainer, 'estimated_stepping_batches'):
            total_steps = self.trainer.estimated_stepping_batches
        else:
            # Fallback calculation if the attribute doesn't exist
            max_epochs = self.trainer.max_epochs
            train_dataloader = self.trainer.train_dataloader
            if hasattr(train_dataloader, 'dataloader'):
                train_dataloader = train_dataloader.dataloader
            
            batches_per_epoch = len(train_dataloader)
            total_steps = batches_per_epoch * max_epochs
            
            # Account for gradient accumulation if used
            if hasattr(self.trainer, 'accumulate_grad_batches'):
                total_steps = total_steps // self.trainer.accumulate_grad_batches
        
        # Paper: linear warm-up over the first `lr_warmup_steps` (2,000) optimizer steps.
        # Set lr_warmup_steps=null to fall back to the legacy 10%-of-training heuristic.
        if self.lr_warmup_steps is not None:
            warmup_steps = min(int(self.lr_warmup_steps), max(1, total_steps - 1))
        else:
            warmup_steps = int(total_steps * 0.1)

        scheduler = get_cosine_schedule_with_warmup(
            optimizer=optimizer,
            num_warmup_steps=warmup_steps,
            num_training_steps=total_steps,
        )

        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "interval": "step",
                "frequency": 1,
            },
        }
