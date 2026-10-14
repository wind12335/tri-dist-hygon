"""Consumer-kernel solo repro (2026-10-14 v13).

Isolation test for the AG-GEMM tear: launch kernel_gemm_ag_tr_consumer_pred
EXACTLY as i1_ag_tr_op launches it (grid=128 persistent, num_stages=1,
same table/flag layout, same strides) but with ZERO concurrency:
single GPU, no MPI, no peer agents -- ws pre-staged by plain torch copies,
ready flags pre-satisfied before the launch, everything host-ordered.

If a wrong tile still appears here, the tear is INSIDE the consumer kernel
(compute/codegen class), independent of the transport mechanism -- that
would explain why every transport variant since S12 kept a residual with
clean gate forensics, clean ws, and content matching no known data.

Arms (each a fresh sequential loop, same data regen per iter):
  A: grid=128 persistent (production form), DIAG=0
  B: grid=128 persistent, DIAG=1 (the v11 soak form)
  C: grid=512 (one tile per CTA, non-persistent), DIAG=0
"""
import os
import sys

import torch

sys.path.insert(0, "/root/private_data/tri-dist-hygon")
import triton

from triton_dist.kernels.dcu.ag_tr_pred import (  # noqa: E402
    kernel_gemm_ag_tr_consumer_pred,
    _CONSUMER_BLOCK_M,
    _CONSUMER_BLOCK_N,
    _CONSUMER_BLOCK_K,
    _CONSUMER_GROUP_M,
    _CONSUMER_WARPS,
    _CONSUMER_GRID_CAP,
)

M_pr, K, N = 2048, 1024, 2048
R = 2
M = M_pr * R
CR = 1024                       # M_PER_CHUNK (auto chunk_rows at this shape)
n_chunks = M_pr // CR
ROUND = 100
SCALE = 0.02
ROUNDS = int(os.environ.get("SOLO_ROUNDS", "2000"))

assert M % _CONSUMER_BLOCK_M == 0 and N % _CONSUMER_BLOCK_N == 0

b = (torch.randn(N, K, device="cuda") * SCALE).half()
# pre-satisfied ready plane: every remote chunk's ticket already up
flags = torch.full((R * n_chunks,), ROUND + 1 + n_chunks,
                   dtype=torch.int32, device="cuda")
# ctl table: mode 0, own-first rotation for rank 0 (order [0, 1]), budget 128
ctl = torch.zeros(4 + 2 * R, dtype=torch.int32, device="cuda")
ctl[2] = 0
ctl[4 + 0] = 0
ctl[4 + 1] = 1
ctl[4 + R + 0] = 128
ctl[4 + R + 1] = 128

diag_buf = torch.full((R * n_chunks * 2,), 1 << 30,
                      dtype=torch.int32, device="cuda")

fails = 0
worst = 0.0
for arm, grid_c, use_diag in (("A", min(M // _CONSUMER_BLOCK_M * N // _CONSUMER_BLOCK_N,
                                        _CONSUMER_GRID_CAP), False),
                              ("B", _CONSUMER_GRID_CAP, True),
                              ("C", M // _CONSUMER_BLOCK_M * N // _CONSUMER_BLOCK_N,
                               False)):
    fails = 0
    worst = 0.0
    for it in range(ROUNDS):
        torch.manual_seed(5000 + 100 * (it % 97) + 7777)
        A_full = (torch.randn(M, K, device="cuda") * SCALE).half()
        a = A_full[:M_pr].contiguous()
        ws = A_full.contiguous()          # pre-staged gathered workspace
        with torch.no_grad():
            ref = A_full.float() @ b.float().t()
        c = torch.empty(M, N, dtype=torch.float16, device="cuda")
        kernel_gemm_ag_tr_consumer_pred[(grid_c, 1, 1)](
            ws,
            a,
            b,
            c,
            M, N, K,
            K, 1,
            a.stride(0), a.stride(1),
            b.stride(1), b.stride(0),
            c.stride(0), c.stride(1),
            0,                       # rank
            R,
            n_chunks,
            flags.data_ptr(),
            ctl.data_ptr(),
            ROUND,
            diag_ptr=(diag_buf.data_ptr() if use_diag else 0),
            BLOCK_SIZE_M=_CONSUMER_BLOCK_M,
            BLOCK_SIZE_N=_CONSUMER_BLOCK_N,
            BLOCK_SIZE_K=_CONSUMER_BLOCK_K,
            GROUP_SIZE_M=_CONSUMER_GROUP_M,
            M_PER_CHUNK=CR,
            NUM_SMS=grid_c,
            EVEN_K=(K % _CONSUMER_BLOCK_K == 0),
            DIAG=use_diag,
            num_warps=_CONSUMER_WARPS,
            num_stages=1 if M > 512 else 2,
        )
        torch.cuda.synchronize()
        d = (c.float() - ref).abs()
        md = d.max().item()
        worst = max(worst, md)
        if md >= 1e-2:
            fails += 1
            wrong = d > 1e-2
            rows = torch.nonzero(wrong.sum(dim=1)).flatten()
            cols = torch.nonzero(wrong.sum(dim=0)).flatten()
            ntiles = sorted(set((cols // 128).tolist()))
            print(f"[arm {arm}] FAIL it={it} md={md:.5f} "
                  f"rows={rows[:6].tolist()}..{int(rows.numel())} "
                  f"ntiles={ntiles[:6]}", flush=True)
        if it % 500 == 0:
            print(f"[arm {arm}] it={it} worst={worst:.5f} fails={fails}",
                  flush=True)
    print(f"SOLO RESULT arm {arm} grid={grid_c} diag={use_diag}: "
          f"fails={fails}/{ROUNDS} worst={worst:.5f}", flush=True)

print("SOLODONE")
