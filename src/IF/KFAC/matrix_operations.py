import re
from typing import Tuple
import torch
from torch import nn
from asdl.matrices import SHAPE_KRON


__all__ = [
    "AdvanceKFACGradientMaker",
    "AdvancedSmallKfacGradientMaker",
]


_NAME_PAT_MLP = re.compile(r"model\.layers\.(\d+)\.mlp\.(c_fc|c_proj|up_proj|down_proj)\.weight$")
_NAME_PAT_Q = re.compile(r"model\.layers\.(\d+)\.self_attn\.(q_proj)\.weight$")
_NAME_PAT_K = re.compile(r"model\.layers\.(\d+)\.self_attn\.(k_proj)\.weight$")
_NAME_PAT_V = re.compile(r"model\.layers\.(\d+)\.self_attn\.(v_proj)\.weight$")
_NAME_PAT = _NAME_PAT_MLP
_INVALID_EMA_DECAY = -1


def _symm(M: torch.Tensor) -> torch.Tensor:
    return 0.5 * (M + M.T)


def _chol_inv(L: torch.Tensor) -> torch.Tensor:
    I = torch.eye(L.size(0), device=L.device, dtype=L.dtype)
    if L.dtype in (torch.float16, torch.bfloat16):
        inv = torch.cholesky_solve(I.float(), L.float())
        return inv.to(dtype=L.dtype)
    return torch.cholesky_solve(I, L)


def _chol_with_adaptive_damping(M, lam_base, growth=10.0, max_tries=6):
    M = _symm(M)
    I = torch.eye(M.size(0), device=M.device, dtype=M.dtype)
    lam = lam_base
    for _ in range(max_tries):
        try:
            return torch.linalg.cholesky(M + lam * I), lam
        except RuntimeError:
            lam = max(lam * growth, 1e-5)
    # PSD projection fallback
    evals, evecs = torch.linalg.eigh(M)
    floor = max(lam, 1e-5)
    evals = torch.clamp(evals, min=floor)
    M_psd = (evecs * evals) @ evecs.T
    return torch.linalg.cholesky(M_psd), floor

# def _chol_with_adaptive_damping(
#     M: torch.Tensor, lam_base: float, growth: float = 10.0, max_tries: int = 6
# ) -> Tuple[torch.Tensor, float]:
#     orig_dtype = M.dtype
#     M = _symm(M)
#     I = torch.eye(M.size(0), device=M.device, dtype=M.dtype)
#     lam = lam_base
#     for _ in range(max_tries):
#         try:
#             if M.dtype in (torch.float16, torch.bfloat16):
#                 L = torch.linalg.cholesky((M + lam * I).float()).to(dtype=orig_dtype)
#             else:
#                 L = torch.linalg.cholesky(M + lam * I)
#             return L, lam
#         except RuntimeError:
#             lam = max(lam * growth, 1e-5)
#     if M.dtype in (torch.float16, torch.bfloat16):
#         M = M.float()
#     evals, evecs = torch.linalg.eigh(M)
#     floor = max(lam, 1e-5)
#     evals = torch.clamp(evals, min=floor)
#     M_psd = (evecs * evals) @ evecs.T
#     M_psd.diagonal().add_(1e-6)
#     if M_psd.dtype in (torch.float16, torch.bfloat16):
#         L_psd = torch.linalg.cholesky(M_psd.float()).to(dtype=orig_dtype)
#     else:
#         L_psd = torch.linalg.cholesky(M_psd)
#     return L_psd, floor


def _mean_eig_scale(M_sum: torch.Tensor, count: int) -> Tuple[float, torch.Tensor]:
    M = _symm(M_sum / max(count, 1))
    tr = torch.trace(M).item()
    d = M.size(0)
    return tr / d, M


def _bias_requires_grad(module: nn.Module) -> bool:
    return hasattr(module, "bias") and module.bias is not None and module.bias.requires_grad


def _left_inv_Ad_apply_full(X, W_u, L_Au, lam):
    inv_lam = 1.0 / lam
    Au_inv = _chol_inv(L_Au)
    S = _symm(Au_inv + inv_lam * (W_u.T @ W_u))
    L_S, _ = _chol_with_adaptive_damping(S, 0.0)
    UTX = W_u.T @ X
    S_inv_UTX = torch.cholesky_solve(UTX, L_S)
    return inv_lam * (X - W_u @ S_inv_UTX * inv_lam)


def _right_inv_Gu_apply_full(X, W_d, L_Gd, lam):
    inv_lam = 1.0 / lam
    Gd_inv = _chol_inv(L_Gd)
    S = _symm(Gd_inv + inv_lam * (W_d @ W_d.T))
    L_S, _ = _chol_with_adaptive_damping(S, 0.0)
    XUdT = X @ W_d.T
    XS = torch.cholesky_solve(XUdT.T, L_S).T
    return inv_lam * (X - XS @ W_d * inv_lam)

 
def _precond_down_full(gW_d, W_u, L_Au_raw, L_Gd_damped, lam_Ad):
    Zt = _left_inv_Ad_apply_full(gW_d.T, W_u, L_Au_raw, lam_Ad)
    Z = Zt.T
    Gd_inv = _chol_inv(L_Gd_damped)
    return Gd_inv @ Z


def _precond_up_full(gW_u, W_d, L_Au_damped, L_Gd_raw, lam_Gu):
    Au_inv = _chol_inv(L_Au_damped)
    Y = gW_u @ Au_inv
    Yt = _right_inv_Gu_apply_full(Y.T, W_d, L_Gd_raw, lam_Gu)
    return Yt.T


"""
_module_level_shapes = [
    SHAPE_LAYER_WISE,
    SHAPE_KRON,
    SHAPE_SWIFT_KRON,
    SHAPE_KFE,
    SHAPE_UNIT_WISE,
    SHAPE_DIAG,
]
"""