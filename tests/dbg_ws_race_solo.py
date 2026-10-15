"""Single-GPU write-race repro rig (2026-10-14 v15).

The real AG tear: consumer reads ws lines while a same-stream
[engine copy X_new -> ws] + [flag raise] chain is still draining. The
flag is wire-ordered after the copy, so any post-gate read must see
X_new (or, if it catches the drain window, SOME mixture with the
pre-copy X_prev). The v11/v13 forensics say the real events match
NEITHER -- pure noise. This rig reproduces the race with fully
controlled content on ONE GPU, no MPI:

  per iter:
    ws = X_prev (plain copy)
    noise stream: engine copy X_new -> ws  (8 MB), then flag = ticket
                  [arm E: engine 4B flag write; arm K: flag KERNEL --
                   the I1_READY=kernel form, the documented
                   launch-completion-is-not-an-ordering-point race]
    main stream: consumer launched BEFORE, spins on the flag, GEMMs
  verdict per iter: c must equal X_new@b (mixture with the KNOWN X_prev
  is checked separately per 64-wide k-line, so noise vs mixture is
  decided on the spot).

Env: RIG_ITERS (default 4000), RIG_ARM=E|K (default K), RIG_PULLS=1.
"""
import ctypes
import os
import sys

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO)
os.environ.setdefault("TRITON_CACHE_DIR", "/tmp/tc_rig15")

import torch  # noqa: E402
import triton  # noqa: E402
import triton.language as tl  # noqa: E402

torch.cuda.set_device(0)

from triton_dist.kernels.dcu import ag_tr_pred as i1  # noqa: E402

# raw hip wrappers (no dushmem control plane needed on this single-GPU rig)
_hip = ctypes.CDLL("/opt/dtk-26.04/lib/libamdhip64.so")


def mycpy(dst, src, nbytes, stream_h):
    _hip.hipMemcpyAsync.argtypes = [ctypes.c_void_p, ctypes.c_void_p,
                                    ctypes.c_size_t, ctypes.c_int,
                                    ctypes.c_void_p]
    rc = _hip.hipMemcpyAsync(ctypes.c_void_p(dst), ctypes.c_void_p(src),
                             nbytes, 3, ctypes.c_void_p(stream_h))
    assert rc == 0, f"hipMemcpyAsync rc={rc}"


def setw32(ptr, val, stream_h):
    _hip.hipStreamWriteValue32.argtypes = [ctypes.c_void_p, ctypes.c_void_p,
                                           ctypes.c_uint, ctypes.c_uint]
    rc = _hip.hipStreamWriteValue32(ctypes.c_void_p(stream_h),
                                    ctypes.c_void_p(ptr),
                                    ctypes.c_uint(val), 0)
    assert rc == 0, f"hipStreamWriteValue32 rc={rc}"


@triton.jit(do_not_specialize=["dst_ptr", "val"])
def _flag_xchg1(dst_ptr, val):
    p = dst_ptr.to(tl.pointer_type(tl.int32))
    tl.atomic_xchg(p, val)

M_pr, K, N = 2048, 1024, 2048
R = 2
M = M_pr * R
CR = 1024
n_chunks = M_pr // CR
SCALE = 0.02
ITERS = int(os.environ.get("RIG_ITERS", "4000"))
ARM = os.environ.get("RIG_ARM", "K")
PULLS = int(os.environ.get("RIG_PULLS", "1"))

b = (torch.randn(N, K, device="cuda") * SCALE).half()
# table/flag world
flags = torch.zeros(R * n_chunks, dtype=torch.int32, device="cuda")
ctl = torch.zeros(4 + 2 * R, dtype=torch.int32, device="cuda")
ctl[2] = 0
ctl[4 + 1] = 1
ctl[4 + R] = 128
ctl[4 + R + 1] = 128

grid_c = min(M // i1._CONSUMER_BLOCK_M * N // i1._CONSUMER_BLOCK_N,
             i1._CONSUMER_GRID_CAP)
s_noise = torch.cuda.Stream()

print(f"rig15: arm={ARM} iters={ITERS} pulls={PULLS} "
      f"grid={grid_c}", flush=True)

fails = 0
worst = 0.0
for it in range(ITERS):
    ticket = it * 8 + 5
    flags.fill_(0)
    torch.manual_seed(9000 + it % 211)
    X_prev = (torch.randn(M, K, device="cuda") * SCALE).half()
    torch.manual_seed(70000 + it % 211)
    X_new = (torch.randn(M, K, device="cuda") * SCALE).half()
    a = X_new[:M_pr].contiguous()
    ws = X_prev.clone()                       # the racing region
    with torch.no_grad():
        ref_new = X_new.float() @ b.float().t()
    c = torch.empty(M, N, dtype=torch.float16, device="cuda")

    # noise chain on s_noise: copy X_new -> ws, then raise ALL chunk
    # flags to this round's ticket (wire/kernel ordered after the copy)
    with torch.cuda.stream(s_noise):
        for _ in range(PULLS):
            mycpy(ws.data_ptr(), X_new.data_ptr(),
                  M * K * 2, s_noise.cuda_stream)
        # per-cell ticket exactly as the real op raises them:
        # expected(cell) = round + 1 + cell % n_chunks, round = ticket-1
        if ARM == "E":
            for cell in range(R * n_chunks):
                setw32(flags.data_ptr() + cell * 4,
                       ticket + cell % n_chunks, s_noise.cuda_stream)
        else:
            for cell in range(R * n_chunks):
                _flag_xchg1[(1,)](flags.data_ptr() + cell * 4,
                                  ticket + cell % n_chunks)

    i1.kernel_gemm_ag_tr_consumer_pred[(grid_c, 1, 1)](
        ws, a, b, c,
        M, N, K,
        K,  # single A row stride
        b.stride(1), b.stride(0),
        c.stride(0), c.stride(1),
        0, R, n_chunks,
        flags.data_ptr(), ctl.data_ptr(), ticket - 1,
        diag_ptr=0,
        BLOCK_SIZE_M=i1._CONSUMER_BLOCK_M,
        BLOCK_SIZE_N=i1._CONSUMER_BLOCK_N,
        BLOCK_SIZE_K=i1._CONSUMER_BLOCK_K,
        GROUP_SIZE_M=i1._CONSUMER_GROUP_M,
        M_PER_CHUNK=CR,
        NUM_SMS=grid_c,
        EVEN_K=True,
        DIAG=False,
        num_warps=i1._CONSUMER_WARPS,
        num_stages=1,
    )
    torch.cuda.synchronize()
    d = (c.float() - ref_new).abs()
    md = d.max().item()
    worst = max(worst, md)
    if md >= 1e-2:
        fails += 1
        wrong = d > 1e-2
        rows = torch.nonzero(wrong.sum(dim=1)).flatten()
        cols = torch.nonzero(wrong.sum(dim=0)).flatten()
        ntiles = sorted(set((cols // 128).tolist()))[:6]
        # mixture check: does the error on damaged rows decompose into
        # per-64-line (X_prev - X_new) contributions? (agoff8 method)
        i0 = int(rows[0])
        e_row = (c.float() - ref_new)[i0]
        D = (X_prev[i0] - X_new[i0])
        LB = 64
        nl = K // LB
        jc = cols[:128]
        y = e_row[jc]
        Xsub = torch.stack([(D[l*LB:(l+1)*LB][None, :]
                             * b[jc][:, l*LB:(l+1)*LB]).sum(dim=1)
                            for l in range(nl)], dim=1)
        x = torch.linalg.lstsq(Xsub, y.unsqueeze(1)).solution.squeeze(1)
        r = (y - Xsub @ x).abs().median()
        print(f"FAIL it={it} md={md:.5f} rows={int(rows.numel())} "
              f"ntiles={ntiles} row{i0} linefit_resid={float(r):.5f} "
              f"base={float(y.abs().median()):.5f} "
              f"x={[f'{float(v):.2f}' for v in x[:8]]}", flush=True)
        torch.save({"it": it, "c": c.cpu(), "ref_new": ref_new.cpu(),
                    "X_prev": X_prev.cpu(), "X_new": X_new.cpu(),
                    "b": b.cpu(), "rows": rows.cpu()},
                   f"/tmp/rig15fail_it{it}.pt")
    if it % 500 == 0:
        print(f"it={it} worst={worst:.5f} fails={fails}", flush=True)

print(f"RIG15 RESULT arm={ARM}: fails={fails}/{ITERS} worst={worst:.5f}",
      flush=True)
print("RIG15DONE")
