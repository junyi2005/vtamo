# Copyright 2024 SpaMo Authors. All rights reserved.
#
# Sinkhorn algorithm for optimal transport / soft alignment
# Modified version: null_ratio means "percentage of windows whose argmax is NULL"
# e.g., null_ratio_target=0.2 means 20% of windows should have NULL as their best match
#
# ORDER version: Added reorder_by_alignment() to reorder sign features according to
# text token order based on OT alignment matrix. This fixes the issue where
# sign language videos are not in natural language order.
#
# WINDOWNOMEAN version: Frame-level alignment + Window-based reordering (NO mean pooling)
# Key changes:
# 1. Alignment is still computed at FRAME level (same as _order)
# 2. After alignment, frames are grouped into windows (U+2 windows, overlapping)
# 3. Window's token = argmax of SUM of frame probabilities within the window
# 4. Reordering is done at WINDOW level (each window mapped to one token)
# 5. NO mean pooling - window features are computed by weighted sum based on alignment
#
# CONTRASTIVE version: Position-aligned contrastive learning on reordered window features
# Key changes:
# - Added compute_position_contrastive_loss() function for position-aligned InfoNCE loss
# - Each window (after reordering) is aligned with its corresponding text token (positive pair)
# - All other text tokens in the batch serve as negative samples
# - Gradient flows to: temporal_encoder, projectors, fusion_proj
# - Gradient does NOT flow to: T5 LoRA parameters (text_feats are detached)
# - Enabled after warm_up_steps to allow OT alignment to stabilize first

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple, List


# ==================== POSITION-ALIGNED CONTRASTIVE LOSS ====================

def compute_position_contrastive_loss(
    window_feats: torch.Tensor,  # [B, U, D] - 重排后的window特征
    text_feats: torch.Tensor,    # [B, U, D] - text token embeddings (detached)
    text_mask: torch.Tensor,     # [B, U] - 有效token掩码
    temperature: float = 0.07,
    return_stats: bool = False,
) -> torch.Tensor:
    """
    位置对齐的对比学习损失 (InfoNCE)。

    每个window和对应位置的text token为正样本，
    和batch内所有其他text tokens为负样本。

    L = -log(exp(sim(w_i, t_i)/τ) / Σ_{j∈batch} exp(sim(w_i, t_j)/τ))

    Gradient flow:
    - window_feats: has gradient -> flows to temporal_encoder, projectors, fusion_proj
    - text_feats: should be detached -> no gradient to T5 LoRA

    Args:
        window_feats: 重排后的window特征 [B, U, D] (has gradient)
        text_feats: text token embeddings [B, U, D] (should be detached)
        text_mask: 有效token掩码 [B, U], 1=有效, 0=padding
        temperature: 温度参数 (default: 0.07)

    Returns:
        loss: 标量损失值
        (loss, stats) if return_stats=True
    """
    B, U, D = window_feats.shape
    device = window_feats.device
    dtype = window_feats.dtype

    # L2归一化
    window_norm = F.normalize(window_feats, dim=-1)  # [B, U, D]
    text_norm = F.normalize(text_feats, dim=-1)      # [B, U, D]

    # 展平为 [B*U, D]
    window_flat = window_norm.view(-1, D)  # [B*U, D]
    text_flat = text_norm.view(-1, D)      # [B*U, D]
    mask_flat = text_mask.view(-1)          # [B*U]

    # 计算所有window和所有text tokens的相似度矩阵
    # logits[i, j] = <window_i, text_j> / τ
    logits = torch.mm(window_flat, text_flat.t()) / temperature  # [B*U, B*U]

    # 正样本标签：对角线元素 (第i个window对应第i个text token)
    labels = torch.arange(B * U, device=device)  # [B*U]

    # 创建有效样本掩码 (只计算有效位置的loss)
    valid_mask = mask_flat.float()  # [B*U]

    # mask掉padding位置的负样本
    padding_mask = (mask_flat < 0.5)  # [B*U], True for padding
    logits_masked = logits.clone()
    logits_masked[:, padding_mask] = float('-inf')

    # 保持对角线（正样本位置）
    diag_indices = torch.arange(B * U, device=device)
    diag_backup = logits[diag_indices, diag_indices].clone()
    restore_mask = padding_mask
    logits_masked[diag_indices[restore_mask], diag_indices[restore_mask]] = diag_backup[restore_mask]

    # Cross-entropy loss
    loss_per_sample = F.cross_entropy(logits_masked, labels, reduction='none')  # [B*U]

    # 只对有效位置求平均
    num_valid = valid_mask.sum().clamp(min=1.0)
    loss = (loss_per_sample * valid_mask).sum() / num_valid

    if return_stats:
        with torch.no_grad():
            sim = logits * temperature  # cosine similarity matrix
            valid = mask_flat > 0.5
            if valid.any():
                diag = sim.diag()
                pos_vals = diag[valid]
                pos_mean = pos_vals.mean()
                if valid.any():
                    sim_valid = sim[valid][:, valid]
                    sum_all = sim_valid.sum()
                    count_all = sim_valid.numel()
                    sum_pos = pos_vals.sum()
                    count_pos = pos_vals.numel()
                    count_neg = count_all - count_pos
                    if count_neg > 0:
                        neg_mean = (sum_all - sum_pos) / count_neg
                    else:
                        neg_mean = torch.tensor(0.0, device=device, dtype=dtype)
                else:
                    neg_mean = torch.tensor(0.0, device=device, dtype=dtype)
                gap = pos_mean - neg_mean
            else:
                pos_mean = torch.tensor(0.0, device=device, dtype=dtype)
                neg_mean = torch.tensor(0.0, device=device, dtype=dtype)
                gap = torch.tensor(0.0, device=device, dtype=dtype)

            stats = {
                'pos_sim_mean': pos_mean,
                'neg_sim_mean': neg_mean,
                'pos_neg_gap': gap,
            }
        return loss, stats

    return loss


# ==================== SINKHORN ALGORITHM ====================

def sinkhorn(
    cost: torch.Tensor,
    a: Optional[torch.Tensor] = None,
    b: Optional[torch.Tensor] = None,
    eps: float = 0.1,
    n_iters: int = 10,
    return_marginals: bool = False
) -> torch.Tensor:
    """
    Batched Sinkhorn algorithm for optimal transport.

    Computes approximate solution to the entropy-regularized optimal transport problem:
        min_P <P, C> - eps * H(P)
        s.t. P @ 1 = a, P.T @ 1 = b

    Args:
        cost: Cost matrix of shape [B, M, K] where M = num windows, K = num tokens (including NULL)
        a: Row marginals (source distribution) of shape [B, M]. Default: uniform
        b: Column marginals (target distribution) of shape [B, K]. Default: uniform
        eps: Entropy regularization coefficient (lower = sharper assignment)
        n_iters: Number of Sinkhorn iterations
        return_marginals: If True, also return the marginals (a, b) for verification

    Returns:
        P: Transport plan (soft assignment matrix) of shape [B, M, K]
           P[b, m, k] = transport mass from window m to token k
        If return_marginals=True, returns (P, a, b)
    """
    B, M, K = cost.shape
    device = cost.device
    dtype = cost.dtype

    # Default marginals: uniform distribution
    if a is None:
        a = torch.ones(B, M, device=device, dtype=dtype) / M
    if b is None:
        # Uniform distribution over all tokens (including NULL)
        b = torch.ones(B, K, device=device, dtype=dtype) / K

    # Ensure marginals sum to 1
    a = a / a.sum(dim=-1, keepdim=True).clamp(min=1e-8)
    b = b / b.sum(dim=-1, keepdim=True).clamp(min=1e-8)

    # Convert cost to log-domain kernel: K_ij = exp(-C_ij / eps)
    log_K = -cost / eps  # [B, M, K]

    # Initialize dual variables (in log domain for numerical stability)
    log_u = torch.zeros(B, M, device=device, dtype=dtype)
    log_v = torch.zeros(B, K, device=device, dtype=dtype)

    log_a = torch.log(a.clamp(min=1e-8))
    log_b = torch.log(b.clamp(min=1e-8))

    # Sinkhorn iterations
    for _ in range(n_iters):
        # Update log_u: u = a / (K @ v)
        log_u = log_a - torch.logsumexp(log_K + log_v.unsqueeze(1), dim=-1)

        # Update log_v: v = b / (K.T @ u)
        log_v = log_b - torch.logsumexp(log_K.transpose(-1, -2) + log_u.unsqueeze(1), dim=-1)

    # Compute transport plan: P = diag(u) @ K @ diag(v)
    log_P = log_u.unsqueeze(-1) + log_K + log_v.unsqueeze(1)
    P = torch.exp(log_P)

    if return_marginals:
        return P, a, b

    return P


def compute_tv_loss(
    A: torch.Tensor,
    exclude_null: bool = True,
    mask: Optional[torch.Tensor] = None,
    text_mask: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """
    Compute Total Variation loss for temporal continuity.

    Penalizes rapid changes in assignment over time, encouraging each token
    to form continuous temporal segments.

    Args:
        A: Alignment matrix of shape [B, M, K] where M is temporal dimension
        exclude_null: If True, exclude NULL token (k=0) from TV computation
        mask: Optional mask of shape [B, M] indicating valid windows

    Returns:
        tv_loss: Scalar TV loss value
    """
    if exclude_null and A.shape[-1] > 1:
        # Exclude NULL token (column 0)
        A_real = A[:, :, 1:]
    else:
        A_real = A

    # Compute temporal difference: |A[t] - A[t-1]|
    diff = (A_real[:, 1:, :] - A_real[:, :-1, :]).abs()  # [B, M-1, K-1]

    if mask is not None:
        # Create diff mask (valid if both t and t-1 are valid)
        diff_mask = mask[:, 1:] * mask[:, :-1]  # [B, M-1]
        diff = diff * diff_mask.unsqueeze(-1)
        tv_loss = diff.sum() / (diff_mask.sum().clamp(min=1) * A_real.shape[-1])
    else:
        tv_loss = diff.mean()

    return tv_loss


def compute_local_align_loss(
    A: torch.Tensor,
    cost: torch.Tensor,
) -> torch.Tensor:
    """
    Compute local alignment loss as total transport cost.

    Args:
        A: Transport plan matrix of shape [B, M, K]
        cost: Cost matrix of shape [B, M, K]

    Returns:
        local_loss: Scalar local alignment loss (mean over batch)
    """
    local_loss = (A * cost).sum(dim=[1, 2]).mean()
    return local_loss


def reorder_by_alignment(
    sign_seq: torch.Tensor,
    A: torch.Tensor,
    text_mask: Optional[torch.Tensor] = None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Reorder sign features according to text token order using soft alignment.

    For each text token k, compute a weighted combination of sign windows
    based on the alignment matrix A. This produces features ordered by
    text token sequence (natural language order).

    Args:
        sign_seq: Sign/video features of shape [B, M, D] (M = num windows)
        A: Alignment matrix of shape [B, M, K] where K = U+1 (includes NULL at index 0)
        text_mask: Optional mask for text tokens [B, U] (excluding NULL)

    Returns:
        reordered_seq: Reordered features [B, U, D] in text token order
        reorder_mask: Mask for reordered sequence [B, U]
    """
    B, M, D = sign_seq.shape
    _, _, K = A.shape
    U = K - 1  # Number of real text tokens (excluding NULL)
    device = sign_seq.device
    dtype = sign_seq.dtype

    # Extract alignment weights for real tokens (exclude NULL at column 0)
    # A_real[b, m, u] = probability that window m aligns to token u
    A_real = A[:, :, 1:]  # [B, M, U]

    # Transpose to get: A_T[b, u, m] = probability that token u gets from window m
    A_T = A_real.transpose(1, 2)  # [B, U, M]

    # Normalize weights for each token (so they sum to 1 over windows)
    # This ensures each token gets a proper weighted combination
    weights = A_T / A_T.sum(dim=-1, keepdim=True).clamp(min=1e-8)  # [B, U, M]

    # Compute weighted combination: reordered[b, u, :] = sum_m weights[b, u, m] * sign_seq[b, m, :]
    # weights: [B, U, M], sign_seq: [B, M, D]
    reordered_seq = torch.bmm(weights, sign_seq)  # [B, U, D]

    # Create mask for reordered sequence
    if text_mask is not None:
        reorder_mask = text_mask  # [B, U]
    else:
        reorder_mask = torch.ones(B, U, device=device, dtype=dtype)

    return reordered_seq, reorder_mask


def compute_window_boundaries(
    num_frames: int,
    num_windows: int,
) -> List[Tuple[int, int]]:
    """
    Compute window boundaries for overlapping windows.

    Windows are designed such that:
    - Total num_windows windows are created
    - Each window covers consecutive frames
    - Adjacent windows overlap at one frame (last frame of window i = first frame of window i+1)

    Args:
        num_frames: Total number of frames in the video
        num_windows: Number of windows to create (= num_tokens + 2)

    Returns:
        List of (start, end) tuples for each window (inclusive start, exclusive end)
    """
    if num_frames <= 0 or num_windows <= 0:
        return []

    boundaries = []
    step = (num_frames - 1) / num_windows

    for i in range(num_windows):
        start = int(round(i * step))
        end = int(round((i + 1) * step)) + 1
        end = min(end, num_frames)
        if end <= start:
            end = start + 1
        boundaries.append((start, end))

    return boundaries


def reorder_by_window_alignment(
    sign_seq: torch.Tensor,
    A: torch.Tensor,
    num_text_tokens: int,
    sign_mask: Optional[torch.Tensor] = None,
    text_mask: Optional[torch.Tensor] = None,
) -> Tuple[torch.Tensor, torch.Tensor, dict]:
    """
    Reorder sign features using WINDOW-based aggregation of frame-level alignment.

    This function:
    1. Takes frame-level alignment matrix A [B, M, K]
    2. Groups frames into windows (num_windows = num_text_tokens + 2)
    3. For each window, SUM the alignment probabilities of all frames in the window
    4. Each window is assigned to the token with highest summed probability
    5. Reorder windows (not frames!) to match text token order

    NO mean pooling is used - each window's feature is the weighted sum of its frames
    based on their alignment probabilities.

    Args:
        sign_seq: Frame features of shape [B, M, D] (M = num frames after TemporalConv)
        A: Frame-level alignment matrix [B, M, K] where K = U+1 (includes NULL)
        num_text_tokens: Number of text tokens U (used to compute num_windows = U + 2)
        sign_mask: Optional mask for valid frames [B, M]
        text_mask: Optional mask for text tokens [B, U]

    Returns:
        reordered_seq: Reordered features [B, U, D] in text token order
        reorder_mask: Mask for reordered sequence [B, U]
        info: Dictionary with window statistics
    """
    B, M, D = sign_seq.shape
    _, _, K = A.shape
    U = K - 1  # Number of real text tokens (excluding NULL)
    device = sign_seq.device
    dtype = sign_seq.dtype

    num_windows = num_text_tokens + 2  # Same as _window version

    # Determine valid frame counts for each sample
    if sign_mask is not None:
        frame_lengths = sign_mask.sum(dim=1).long()  # [B]
    else:
        frame_lengths = torch.full((B,), M, device=device, dtype=torch.long)

    # Initialize outputs
    # We'll create window features and then reorder them
    reordered_seq = torch.zeros(B, U, D, device=device, dtype=dtype)
    if text_mask is not None:
        reorder_mask = text_mask.clone()
    else:
        reorder_mask = torch.ones(B, U, device=device, dtype=dtype)

    window_token_assignments = []  # For logging

    for b in range(B):
        num_frames_b = frame_lengths[b].item()

        if num_frames_b == 0:
            continue

        # Step 1: Compute window boundaries
        actual_num_windows = min(num_windows, num_frames_b)
        boundaries = compute_window_boundaries(num_frames_b, actual_num_windows)

        # Step 2: For each window, aggregate frame alignment probabilities
        # A[b] is [M, K], we need to sum probabilities within each window
        window_probs = torch.zeros(actual_num_windows, K, device=device, dtype=dtype)
        window_features = torch.zeros(actual_num_windows, D, device=device, dtype=dtype)

        for w, (start, end) in enumerate(boundaries):
            # Sum alignment probabilities for frames in this window
            window_probs[w] = A[b, start:end, :].sum(dim=0)  # [K]

            # Compute window feature as weighted sum of frames based on their alignment
            # For each frame, use its total alignment weight (sum over all tokens)
            frame_weights = A[b, start:end, :].sum(dim=-1)  # [end-start]
            frame_weights = frame_weights / frame_weights.sum().clamp(min=1e-8)
            window_features[w] = (sign_seq[b, start:end, :] * frame_weights.unsqueeze(-1)).sum(dim=0)

        # Step 3: Assign each window to a token based on summed probabilities
        # Exclude NULL (column 0) for determining the best real token
        window_probs_real = window_probs[:, 1:]  # [num_windows, U]

        # For each window, find which token has highest aggregated probability
        window_to_token = window_probs_real.argmax(dim=-1)  # [num_windows], values in [0, U-1]

        # Also track which windows prefer NULL
        window_best_overall = window_probs.argmax(dim=-1)  # [num_windows], values in [0, K-1]
        is_null_window = (window_best_overall == 0)

        window_token_assignments.append(window_to_token.tolist())

        # Step 4: Reorder windows to match text token order
        # For each text token u, find the window(s) assigned to it
        # If multiple windows -> use the one with highest probability
        # If no window -> use weighted average of all windows based on their prob for this token

        for u in range(U):
            # Check if text token u is valid
            if text_mask is not None and text_mask[b, u] < 0.5:
                continue

            # Find windows assigned to this token (excluding NULL-preferring windows)
            assigned_mask = (window_to_token == u) & (~is_null_window)

            if assigned_mask.any():
                # Use the window with highest probability for this token
                assigned_indices = torch.where(assigned_mask)[0]
                probs_for_u = window_probs_real[assigned_indices, u]
                best_idx = assigned_indices[probs_for_u.argmax()]
                reordered_seq[b, u] = window_features[best_idx]
            else:
                # No window strongly assigned to this token
                # Use weighted average of all windows based on their prob for this token
                weights = window_probs_real[:, u]
                weights = weights / weights.sum().clamp(min=1e-8)
                reordered_seq[b, u] = (window_features * weights.unsqueeze(-1)).sum(dim=0)

    info = {
        'num_windows': num_windows,
        'window_token_assignments': window_token_assignments,
    }

    return reordered_seq, reorder_mask, info


class LocalAlignmentModule(nn.Module):
    """
    Module for computing local token-to-window alignment with NULL token support.

    New semantics for null_ratio:
    - null_ratio_target: Target percentage of windows whose argmax should be NULL
    - e.g., null_ratio_target=0.2 means 20% of windows should have NULL as best match
    - Learnable null_bias to control which windows prefer NULL
    """

    def __init__(
        self,
        hidden_size: int,
        eps: float = 0.1,
        n_iters: int = 10,
        null_ratio_target: float = 0.2,
        beta_local: float = 1.0,
        beta_tv: float = 0.1,
        beta_null: float = 0.1,
    ):
        """
        Args:
            hidden_size: Dimension of embeddings (D)
            eps: Sinkhorn entropy regularization
            n_iters: Number of Sinkhorn iterations
            null_ratio_target: Target ratio of windows whose argmax is NULL (e.g., 0.2 = 20%)
            beta_local: Weight for local alignment loss
            beta_tv: Weight for TV loss
            beta_null: Weight for null ratio regularization loss
        """
        super().__init__()
        self.hidden_size = hidden_size
        self.eps = eps
        self.n_iters = n_iters
        self.null_ratio_target = null_ratio_target
        self.beta_local = beta_local
        self.beta_tv = beta_tv
        self.beta_null = beta_null

        # Learnable NULL token embedding
        self.null_token = nn.Parameter(torch.randn(hidden_size) * 0.02)

        # Learnable bias for NULL token (controls how easily NULL wins)
        # Positive bias makes NULL more attractive (lower cost)
        self.null_bias = nn.Parameter(torch.tensor(0.0))

        # ========== GLOBAL version: learnable mapping matrix T ==========
        # Calibrates the sign embedding space against the text embedding space.
        # Initialized near identity and constrained towards orthogonality by
        # L_orth = ||T^T T - I||_F^2 (see vtamo.global_align.compute_orth_loss).
        # T is used here (local OT cost) and supervised by the global EMD loss.
        self.T = nn.Parameter(torch.eye(hidden_size) + 0.01 * torch.randn(hidden_size, hidden_size))

    def forward(
        self,
        sign_seq: torch.Tensor,
        text_seq: torch.Tensor,
        sign_mask: Optional[torch.Tensor] = None,
        text_mask: Optional[torch.Tensor] = None,
        return_alignment: bool = False
    ) -> Tuple[torch.Tensor, dict]:
        """
        Compute local alignment loss between sign sequence and text sequence.

        Args:
            sign_seq: Sign/video features of shape [B, M, D] (M = num windows)
            text_seq: Text token embeddings of shape [B, U, D] (U = num tokens)
            sign_mask: Optional mask for sign sequence [B, M]
            text_mask: Optional mask for text sequence [B, U]
            return_alignment: If True, include alignment matrix in output dict

        Returns:
            total_loss: Combined local + TV + null_ratio loss
            info_dict: Dictionary with individual losses and statistics
        """
        B, M, D = sign_seq.shape
        _, U, _ = text_seq.shape
        device = sign_seq.device
        dtype = sign_seq.dtype

        # Prepend NULL token to text sequence: [B, U+1, D]
        null_expanded = self.null_token.unsqueeze(0).unsqueeze(0).expand(B, 1, D)
        text_seq_with_null = torch.cat([null_expanded, text_seq], dim=1)  # [B, U+1, D]
        K = U + 1  # Total tokens including NULL

        # ========== GLOBAL version: Apply mapping matrix T to sign embeddings ==========
        # Transform sign embeddings: x' = x @ T
        # This allows global calibration of sign embedding space
        sign_transformed = torch.matmul(sign_seq, self.T)  # [B, M, D]

        # L2 normalize for cosine similarity
        sign_norm = F.normalize(sign_transformed, dim=-1)  # [B, M, D]
        text_norm = F.normalize(text_seq_with_null, dim=-1)  # [B, K, D]

        # Compute similarity and cost matrices
        sim = torch.bmm(sign_norm, text_norm.transpose(-1, -2))  # [B, M, K]
        cost = 1.0 - sim  # Cosine distance as cost

        # Apply learnable bias to NULL column (column 0)
        # Subtract bias from NULL cost (positive bias = lower cost = more attractive)
        cost_with_bias = cost.clone()
        cost_with_bias[:, :, 0] = cost[:, :, 0] - self.null_bias
        # Mask padding columns: set cost to large value so Sinkhorn won't assign mass
        if text_mask is not None:
            padding_mask = (1 - text_mask.float()).bool()  # [B, U], True for padding
            cost_with_bias[:, :, 1:].masked_fill_(padding_mask.unsqueeze(1), 1e6)


        # Prepare marginals (uniform)
        if sign_mask is not None:
            a = sign_mask.float()
            a = a / a.sum(dim=-1, keepdim=True).clamp(min=1e-8)
        else:
            a = None

        if text_mask is not None:
            # Create uniform marginal over valid tokens + NULL
            b = torch.cat([
                torch.ones(B, 1, device=device, dtype=dtype),  # NULL always valid
                text_mask.float()
            ], dim=1)
            b = b / b.sum(dim=-1, keepdim=True).clamp(min=1e-8)
        else:
            b = None

        # Compute soft alignment via Sinkhorn
        A, a_used, b_used = sinkhorn(
            cost_with_bias, a=a, b=b, eps=self.eps, n_iters=self.n_iters, return_marginals=True
        )  # [B, M, K]

        # ========== Compute losses ==========
        # Local alignment loss (use original cost, not biased)
        local_loss = compute_local_align_loss(A, cost)

        # TV loss for temporal smoothness
        tv_loss = compute_tv_loss(A, exclude_null=True, mask=sign_mask, text_mask=text_mask)

        # ========== Compute null_ratio based on argmax ==========
        # For each window, check if argmax is NULL (index 0)
        argmax_indices = A.argmax(dim=-1)  # [B, M]
        is_null = (argmax_indices == 0)  # [B, M] boolean

        if sign_mask is not None:
            # Only count valid windows
            valid_null = is_null.float() * sign_mask.float()
            null_ratio = valid_null.sum() / sign_mask.sum().clamp(min=1)
        else:
            null_ratio = is_null.float().mean()

        # ========== Soft null ratio for gradient flow ==========
        # IMPORTANT: A[:, :, 0] is transport mass, NOT probability!
        # Since row marginal a ≈ 1/M, A[:, :, 0] is O(1/M), not suitable for null_ratio_target=0.2
        #
        # Instead, use softmax over -cost_with_bias to get "window-level winner probability"
        # This approximates argmax in a differentiable way
        tau = 0.1  # Temperature: smaller = closer to argmax
        logits = -cost_with_bias / tau  # [B, M, K], lower cost = higher logit
        p_winner = torch.softmax(logits, dim=-1)  # [B, M, K], probability each token wins
        p_null = p_winner[:, :, 0]  # [B, M], probability NULL wins for each window

        if sign_mask is not None:
            soft_null_ratio = (p_null * sign_mask.float()).sum() / sign_mask.sum().clamp(min=1)
        else:
            soft_null_ratio = p_null.mean()

        # Null ratio regularization: "at most X%" semantics
        # Only penalize when soft_null_ratio EXCEEDS the target (hinge loss)
        # This allows 0~target range without penalty, only penalizes > target
        null_reg_loss = torch.relu(soft_null_ratio - self.null_ratio_target) ** 2

        # Total loss
        total_loss = (
            self.beta_local * local_loss +
            self.beta_tv * tv_loss +
            self.beta_null * null_reg_loss
        )

        # ========== Compute statistics for logging ==========
        info_dict = {
            'local_loss': local_loss.detach(),
            'tv_loss': tv_loss.detach(),
            'null_reg_loss': null_reg_loss.detach(),
            'null_ratio': null_ratio.item(),  # Actual ratio (argmax based) - this is what you want!
            'soft_null_ratio': soft_null_ratio.item(),  # Soft ratio (probability based)
            'null_bias': self.null_bias.item(),  # Current learned bias
            'alignment': A,  # Always return alignment matrix (with gradient) for reordering
        }

        return total_loss, info_dict
