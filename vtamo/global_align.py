# Copyright 2024 SpaMo Authors. All rights reserved.
#
# VTaMo — global alignment (paper Sec. "Global alignment").
#
# Calibrates the frozen visual and textual embedding spaces with a learnable
# orthogonal transformation T, supervised by an Earth Mover's Distance objective
# computed over a FIFO memory queue of pooled sentence vectors.
#
#   L_global = lambda_g(t) * L_EMD + beta_orth * L_orth
#   L_orth   = ||T^T T - I||_F^2
#
# T itself lives on ``LocalAlignmentModule`` (see the ot_sinkhorn module), where it
# also transforms the sign embeddings before the local OT cost is computed.
#
# The global loss updates only T: the sign/text sequences are detached before pooling.

from collections import deque
from typing import Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from vtamo.ot_sinkhorn import sinkhorn


def attention_pool_sequence(
    X: torch.Tensor,
    mask: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """
    Parameter-free, data-dependent 1-query attention pooling.

    Converts a sequence X ∈ R^{L×D} into a single vector ∈ R^D.
    No learnable parameters - query is derived from masked mean of X.

    Steps:
    1. q = masked_mean(X) along sequence dimension
    2. alpha = softmax((X @ q) / sqrt(D))  → attention weights [L]
    3. pooled = sum(alpha_i * X_i)  → [D]
    4. L2 normalize the result

    Args:
        X: Input sequence [L, D] or [B, L, D]
        mask: Optional validity mask [L] or [B, L], 1=valid, 0=invalid

    Returns:
        pooled: Pooled vector [D] or [B, D], L2 normalized
    """
    # Handle both batched and unbatched input
    if X.dim() == 2:
        # Unbatched: [L, D]
        L, D = X.shape

        # Step 1: Compute query as masked mean
        # NOTE: cast the mask to X.dtype (not .float()), otherwise a fp32 mask
        # promotes bf16/fp16 activations to fp32 and the matmul below fails on a
        # dtype mismatch outside autocast.
        if mask is not None:
            mask_float = mask.to(X.dtype).unsqueeze(-1)  # [L, 1]
            q = (X * mask_float).sum(dim=0) / mask_float.sum(dim=0).clamp(min=1e-8)  # [D]
        else:
            q = X.mean(dim=0)  # [D]

        # Step 2: Compute attention weights
        # scores = (X @ q) / sqrt(D)
        scores = torch.mv(X, q) / (D ** 0.5)  # [L]

        if mask is not None:
            # Mask out invalid positions with -inf
            scores = scores.masked_fill(mask < 0.5, float('-inf'))

        alpha = torch.softmax(scores, dim=0)  # [L]

        # Step 3: Weighted sum
        pooled = (alpha.unsqueeze(-1) * X).sum(dim=0)  # [D]

        # Step 4: L2 normalize
        pooled = F.normalize(pooled, dim=-1)

        return pooled

    else:
        # Batched: [B, L, D]
        B, L, D = X.shape

        # Step 1: Compute query as masked mean
        # NOTE: cast the mask to X.dtype (not .float()) — see the unbatched branch.
        if mask is not None:
            mask_float = mask.to(X.dtype).unsqueeze(-1)  # [B, L, 1]
            q = (X * mask_float).sum(dim=1) / mask_float.sum(dim=1).clamp(min=1e-8)  # [B, D]
        else:
            q = X.mean(dim=1)  # [B, D]

        # Step 2: Compute attention weights
        # scores[b, l] = (X[b, l, :] @ q[b, :]) / sqrt(D)
        scores = torch.bmm(X, q.unsqueeze(-1)).squeeze(-1) / (D ** 0.5)  # [B, L]

        if mask is not None:
            # Mask out invalid positions with -inf
            scores = scores.masked_fill(mask < 0.5, float('-inf'))

        alpha = torch.softmax(scores, dim=1)  # [B, L]

        # Step 3: Weighted sum
        pooled = (alpha.unsqueeze(-1) * X).sum(dim=1)  # [B, D]

        # Step 4: L2 normalize
        pooled = F.normalize(pooled, dim=-1)

        return pooled



class FIFOMemoryQueue:
    """
    FIFO memory queue for storing sentence-level vectors across batches.

    Stores (sign_vec, text_vec) pairs. When full, oldest entries are removed.
    Used to accumulate sentence vectors for large-batch global OT.

    Note: This is a simple single-process implementation. Does not sync across
    distributed processes (each rank maintains its own local queue).
    """

    def __init__(self, max_size: int = 256, feature_dim: int = 2048):
        """
        Args:
            max_size: Maximum number of (sign, text) pairs to store
            feature_dim: Dimension of feature vectors
        """
        self.max_size = max_size
        self.feature_dim = feature_dim
        self.sign_queue = deque(maxlen=max_size)
        self.text_queue = deque(maxlen=max_size)

    def push(self, sign_vecs: torch.Tensor, text_vecs: torch.Tensor) -> None:
        """
        Push batch of (sign, text) vector pairs into the queue.

        Args:
            sign_vecs: Sign sentence vectors [B, D], should be detached and normalized
            text_vecs: Text sentence vectors [B, D], should be detached and normalized
        """
        # Detach and move to CPU for storage (avoid GPU memory accumulation)
        sign_vecs = sign_vecs.detach().cpu()
        text_vecs = text_vecs.detach().cpu()

        B = sign_vecs.shape[0]
        for i in range(B):
            self.sign_queue.append(sign_vecs[i])
            self.text_queue.append(text_vecs[i])

    def get_all(self, device: torch.device) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Get all stored vectors as tensors.

        Args:
            device: Device to move tensors to

        Returns:
            sign_vecs: [N, D] all sign vectors in queue
            text_vecs: [N, D] all text vectors in queue
        """
        if len(self.sign_queue) == 0:
            return None, None

        sign_vecs = torch.stack(list(self.sign_queue), dim=0).to(device)
        text_vecs = torch.stack(list(self.text_queue), dim=0).to(device)
        return sign_vecs, text_vecs

    def __len__(self) -> int:
        return len(self.sign_queue)

    def clear(self) -> None:
        self.sign_queue.clear()
        self.text_queue.clear()



def compute_global_emd_loss_sentence(
    sign_vecs: torch.Tensor,
    text_vecs: torch.Tensor,
    T: torch.Tensor,
    eps: float = 0.05,
    n_iters: int = 20,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Compute global EMD loss between sentence-level vectors using Sinkhorn OT.

    This version operates on sentence vectors (one per sample) instead of
    point clouds (multiple points per sample).

    Cost matrix: C_ij = 1 - cos((v_i @ T), s_j)
    where v_i is sign sentence vector, s_j is text sentence vector.

    Args:
        sign_vecs: Sign sentence vectors [N, D] (already L2 normalized, but NOT transformed by T yet)
        text_vecs: Text sentence vectors [M, D] (already L2 normalized)
        T: Learnable mapping matrix [D, D]
        eps: Entropy regularization coefficient
        n_iters: Number of Sinkhorn iterations

    Returns:
        emd_loss: Scalar EMD loss (transport cost)
        transport_plan: [N, M] transport plan matrix
    """
    N, D = sign_vecs.shape
    M, _ = text_vecs.shape
    device = sign_vecs.device

    # The sign and text towers can reach here in different dtypes (e.g. the queue
    # holds bf16 text embeddings from the LM but fp32 pooled visual vectors).
    # Compute the whole global objective in T's dtype: T is the only parameter
    # being updated here, and keeping it in fp32 also keeps the orthogonality
    # constraint and the Sinkhorn iterations numerically stable.
    dtype = T.dtype
    sign_vecs = sign_vecs.to(dtype)
    text_vecs = text_vecs.to(dtype)

    # Transform sign vectors: v' = normalize(v @ T)
    # T has gradient, sign_vecs should be detached before calling this
    sign_transformed = torch.mm(sign_vecs, T)  # [N, D]
    sign_transformed = F.normalize(sign_transformed, dim=-1)  # [N, D]

    # Compute cosine similarity and cost
    # sim[i, j] = <sign_transformed[i], text_vecs[j]>
    sim = torch.mm(sign_transformed, text_vecs.t())  # [N, M]
    cost = 1.0 - sim  # Cosine distance

    # Add batch dimension for sinkhorn: [1, N, M]
    cost = cost.unsqueeze(0)

    # Uniform marginals
    a = torch.ones(1, N, device=device, dtype=dtype) / N
    b = torch.ones(1, M, device=device, dtype=dtype) / M

    # Run Sinkhorn
    P = sinkhorn(cost, a=a, b=b, eps=eps, n_iters=n_iters)  # [1, N, M]
    P = P.squeeze(0)  # [N, M]

    # Compute transport cost: <P, C>
    emd_loss = (P * cost.squeeze(0)).sum()

    return emd_loss, P



def compute_eps_schedule(
    current_step: int,
    total_steps: int,
    warm_up_steps: int,
    phase2_steps: int = 0,
    phase3_steps: int = 0,
    eps_high: float = 0.12,
    eps_mid: float = 0.10,
    eps_low: float = 0.03,
    schedule_type: str = "linear"
) -> float:
    """
    Compute current epsilon value based on training progress.

    Three-phase schedule (aligned with LLM training):
    - Phase 1 (warm_up, only local loss): eps = eps_high (constant, stable alignment)
    - Phase 2 (LLM first N epochs): eps_high -> eps_mid (gentle transition)
    - Phase 3 (M epochs): eps_mid -> eps_low (sharpen alignment)
    - Phase 4 (rest): eps = eps_low (maintain sharp alignment)

    Args:
        current_step: Current training step
        total_steps: Total training steps
        warm_up_steps: Number of warmup steps (Phase 1 ends here)
        phase2_steps: Number of steps for Phase 2 (e.g., 10 epochs worth of steps)
        phase3_steps: Number of steps for Phase 3 (e.g., 80 epochs worth of steps)
                      If 0, Phase 3 continues until end of training
        eps_high: Epsilon during Phase 1 (soft, stable) - default 0.12
        eps_mid: Epsilon at end of Phase 2 / start of Phase 3 - default 0.10
        eps_low: Final epsilon (sharp alignment) - default 0.03
        schedule_type: "linear" or "cosine" or "exponential"

    Returns:
        Current epsilon value
    """
    if total_steps <= 0:
        return eps_high

    # Phase 1: Warmup (only local loss) - keep eps_high
    if current_step <= warm_up_steps:
        return eps_high

    # Phase 2: LLM first N steps - decay from eps_high to eps_mid
    phase2_end = warm_up_steps + phase2_steps
    if current_step <= phase2_end and phase2_steps > 0:
        phase2_progress = (current_step - warm_up_steps) / phase2_steps
        phase2_progress = min(1.0, max(0.0, phase2_progress))

        if schedule_type == "linear":
            eps = eps_high - (eps_high - eps_mid) * phase2_progress
        elif schedule_type == "cosine":
            eps = eps_mid + (eps_high - eps_mid) * 0.5 * (1 + np.cos(np.pi * phase2_progress))
        else:
            eps = eps_high - (eps_high - eps_mid) * phase2_progress

        return max(eps_mid, min(eps_high, eps))

    # Phase 3: Decay from eps_mid to eps_low over phase3_steps
    # If phase3_steps is 0, use remaining steps (old behavior)
    if phase3_steps <= 0:
        phase3_steps = total_steps - phase2_end

    phase3_end = phase2_end + phase3_steps

    if current_step <= phase3_end:
        phase3_progress = (current_step - phase2_end) / phase3_steps
        phase3_progress = min(1.0, max(0.0, phase3_progress))

        if schedule_type == "linear":
            eps = eps_mid - (eps_mid - eps_low) * phase3_progress
        elif schedule_type == "cosine":
            eps = eps_low + (eps_mid - eps_low) * 0.5 * (1 + np.cos(np.pi * phase3_progress))
        elif schedule_type == "exponential":
            ratio = eps_low / max(eps_mid, 1e-8)
            eps = eps_mid * (ratio ** phase3_progress)
        else:
            eps = eps_mid - (eps_mid - eps_low) * phase3_progress

        return max(eps_low, min(eps_mid, eps))

    # Phase 4: Maintain eps_low
    return eps_low



def compute_orth_loss(T: torch.Tensor) -> torch.Tensor:
    """
    Compute orthogonality constraint loss for mapping matrix T.

    L_orth = ||T^T @ T - I||_F^2

    This encourages T to be an orthogonal (rotation) matrix,
    preserving the geometry of the embedding space.

    Args:
        T: Mapping matrix [D, D]

    Returns:
        orth_loss: Scalar orthogonality loss
    """
    D = T.shape[0]
    I = torch.eye(D, device=T.device, dtype=T.dtype)
    TtT = torch.mm(T.t(), T)
    orth_loss = torch.norm(TtT - I, p='fro') ** 2
    return orth_loss



def procrustes_init_from_ot_plan(
    X: torch.Tensor,
    Y: torch.Tensor,
    P: torch.Tensor,
) -> torch.Tensor:
    """
    Compute orthogonal Procrustes initialization from OT transport plan.

    Given point clouds X (sign) and Y (text) with OT plan P as soft correspondences,
    find orthogonal matrix T such that X @ T ≈ Y in the weighted least-squares sense.

    Weighted Procrustes problem:
        min_{T ∈ O(d)} sum_{ij} P_ij ||x_i @ T - y_j||^2

    Solution via SVD:
        1. Compute weighted centroids: μ_x = (P @ 1)^T X / mass, μ_y = (P^T @ 1)^T Y / mass
        2. Center: Xc = X - μ_x, Yc = Y - μ_y
        3. Cross-covariance: M = Xc^T @ P @ Yc
        4. SVD: M = U @ Σ @ V^T
        5. T = U @ V^T (with reflection correction if det(T) < 0)

    Args:
        X: Source point cloud [N, D] (sign embeddings, already normalized)
        Y: Target point cloud [M, D] (text embeddings, already normalized)
        P: Transport plan [N, M] from Sinkhorn

    Returns:
        T: Orthogonal matrix [D, D] for initialization
    """
    device = X.device
    dtype = X.dtype
    D = X.shape[1]

    # Compute marginals and total mass
    wx = P.sum(dim=1)  # [N] - row marginals
    wy = P.sum(dim=0)  # [M] - column marginals
    mass = P.sum()     # scalar - total mass

    # Compute weighted centroids
    # μ_x = sum_i wx_i * x_i / mass = (wx^T @ X) / mass
    mu_x = (wx.unsqueeze(0) @ X).squeeze(0) / mass.clamp(min=1e-8)  # [D]
    mu_y = (wy.unsqueeze(0) @ Y).squeeze(0) / mass.clamp(min=1e-8)  # [D]

    # Center the point clouds
    Xc = X - mu_x.unsqueeze(0)  # [N, D]
    Yc = Y - mu_y.unsqueeze(0)  # [M, D]

    # Compute weighted cross-covariance matrix: M = Xc^T @ P @ Yc
    # M[i,j] = sum_{n,m} Xc[n,i] * P[n,m] * Yc[m,j]
    M = Xc.t() @ P @ Yc  # [D, D]

    # ========== SVD in float32 for numerical stability ==========
    # bf16 SVD can be unstable or unsupported in some environments
    M_f32 = M.float()

    # SVD: M = U @ S @ Vh
    U, S, Vh = torch.linalg.svd(M_f32)

    # Compute T = U @ Vh (keep in float32 for det computation)
    T = (U @ Vh).float()

    # Reflection correction: if det(T) < 0, flip sign of last column of U
    # Note: torch.det doesn't support BFloat16, so T must be float32
    det_T = torch.det(T)
    if det_T < 0:
        U_corrected = U.clone()
        U_corrected[:, -1] = -U_corrected[:, -1]
        T = (U_corrected @ Vh).float()

    # Convert back to original dtype
    return T.to(dtype)



def compute_lambda_g_schedule(
    current_step: int,
    warm_up_steps: int,
    emd_ramp_steps: int,
    lambda_g_max: float,
    schedule_type: str = "linear"
) -> float:
    """
    Compute current lambda_g value for global EMD loss.

    Three-stage schedule:
    - Stage 0 (0 ~ warm_up_steps): lambda_g = 0 (global loss disabled)
    - Stage 1 (warm_up_steps ~ warm_up_steps + emd_ramp_steps): lambda_g ramps 0 -> lambda_g_max
    - Stage 2 (after): lambda_g = lambda_g_max (maintain)

    Args:
        current_step: Current training step
        warm_up_steps: End of Stage 0
        emd_ramp_steps: Duration of Stage 1 (ramp up period)
        lambda_g_max: Maximum lambda_g value
        schedule_type: "linear" or "cosine"

    Returns:
        Current lambda_g value
    """
    # Stage 0: Global loss disabled
    if current_step <= warm_up_steps:
        return 0.0

    # Stage 1: Ramp up
    ramp_end = warm_up_steps + emd_ramp_steps
    if current_step <= ramp_end and emd_ramp_steps > 0:
        progress = (current_step - warm_up_steps) / emd_ramp_steps
        progress = min(1.0, max(0.0, progress))

        if schedule_type == "linear":
            return lambda_g_max * progress
        elif schedule_type == "cosine":
            # Cosine ramp: 0 -> lambda_g_max
            return lambda_g_max * 0.5 * (1 - np.cos(np.pi * progress))
        else:
            return lambda_g_max * progress

    # Stage 2: Maintain
    return lambda_g_max



