"""2-card correctness test for the DCU MoE a2a port (dispatch + combine).

Flow per iteration (fresh data every round):
  dispatch(tokens, topk_idx) -> every rank's recv holds all rows routed to
  it; fake expert phase writes expert_out[row] = recv[row] + row; combine
  produces out[t] = sum_j w[t,j] * (tokens[t] + scatter[t, j]).

All ranks generate every rank's inputs deterministically on the CPU (same
seeds), so rank 0 gathers the recv dumps + scatter indices + outputs over
MPI and audits the routing end-to-end:
  1. recv_d[scatter_s[t,j]] == tokens_s[t]      (content, all (t,j))
  2. global (dst, row) pairs unique & < recv_total_d  (bijection)
  3. expert_out_d[r] == recv_d[r] + r            (fake expert phase)
  4. out_s[t] ~= sum_j w[t,j] * (tokens_s[t] + scatter_s[t,j])
P10.2 discipline: reference data comes from the same CPU-side streams that
produced the GPU inputs.

mpirun --allow-run-as-root -np 2 --oversubscribe --bind-to none \
  -mca coll ^hcoll -x HSA_FORCE_FINE_GRAIN_PCIE=1 \
  python3 tests/test_moe_a2a.py
"""
import os
import sys

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO)
os.environ.setdefault("TRITON_CACHE_DIR",
                      f"/tmp/tc_s7a2a_{os.environ.get('OMPI_COMM_WORLD_RANK', '0')}")

import ctypes  # noqa: E402

import numpy as np  # noqa: E402

import pydushmem.host as pHost  # noqa: E402

mpi, comm = pHost.mpi_init()
rank = pHost.mpi_comm_rank(mpi, comm)
size = pHost.mpi_comm_size(mpi, comm)
assert size == 2

hip = ctypes.CDLL("/opt/dtk-26.04/lib/libamdhip64.so")
hip.hipSetDevice.restype = ctypes.c_int
assert hip.hipSetDevice(rank) == 0

import torch  # noqa: E402
import triton  # noqa: E402
import triton.language as tl  # noqa: E402

from triton_dist.kernels.dcu import moe_a2a  # noqa: E402
from triton_dist.kernels.dcu.common_ops import dcu_state  # noqa: E402

torch.cuda.set_device(rank)

NUM_EXPERTS, H, T, TOPK = 32, 2048, 1024, 8
DTYPE = torch.float16
EPR = NUM_EXPERTS // size
MRR = T * TOPK * size  # max recv rows (ctx worst case: all (t,j) to one rank)


@triton.jit(do_not_specialize=["recv_base", "out_base"])
def k_expert_fanout(recv_base, out_base, H: tl.constexpr,
                    BLOCK: tl.constexpr):
    """Fake expert compute: expert_out[row] = fp16(recv[row]*2 + row%5).
    Row-dependent but bounded -- a bare +row transform pushes the combine
    sum past the fp16 max (rows reach 16K -> ~100K > 65504 -> inf)."""
    row = tl.program_id(0).to(tl.int64)
    rp = recv_base.to(tl.pointer_type(tl.float16))
    op = out_base.to(tl.pointer_type(tl.float16))
    for off in range(0, H, BLOCK):
        offs = off + tl.arange(0, BLOCK)
        v = tl.load(rp + row * H + offs).to(tl.float32) * 2.0 \
            + (row % 5).to(tl.float32)
        tl.store(op + row * H + offs, v.to(tl.float16))


@triton.jit(do_not_specialize=["base", "n"])
def k_dump(base, dst_ptr, n, BLOCK: tl.constexpr):
    pid = tl.program_id(0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    m = offs < n
    p = base.to(tl.pointer_type(tl.float16))
    tl.store(dst_ptr + offs, tl.load(p + offs, mask=m), mask=m)


def _mpi_helper():
    if not hasattr(mpi, "_tri_mpi_byte"):
        mpi._tri_mpi_byte = ctypes.addressof(
            ctypes.c_int.in_dll(mpi, "ompi_mpi_byte"))
        mpi._tri_mpi_int = ctypes.addressof(
            ctypes.c_int.in_dll(mpi, "ompi_mpi_int"))
        # OpenMPI handles are &symbol (struct variables in BSS), never the
        # small-int Fortran values -- passing e.g. 2 as the op SEGVs inside
        # PMPI (deref of ~2+offset)
        mpi._tri_mpi_max = ctypes.addressof(
            ctypes.c_char.in_dll(mpi, "ompi_mpi_op_max"))
    return mpi._tri_mpi_byte, mpi._tri_mpi_int


def mpi_allgather_bytes(arr):
    MPI_BYTE, _ = _mpi_helper()
    mpi.MPI_Allgather.argtypes = [ctypes.c_void_p, ctypes.c_int,
                                  ctypes.c_void_p, ctypes.c_void_p,
                                  ctypes.c_int, ctypes.c_void_p,
                                  ctypes.c_void_p]
    mpi.MPI_Allgather.restype = ctypes.c_int
    out = np.empty(arr.nbytes * size, dtype=np.uint8)
    rc = mpi.MPI_Allgather(ctypes.c_void_p(arr.ctypes.data), arr.nbytes,
                           ctypes.c_void_p(MPI_BYTE),
                           ctypes.c_void_p(out.ctypes.data), arr.nbytes,
                           ctypes.c_void_p(MPI_BYTE),
                           ctypes.c_void_p(comm))
    assert rc == 0
    return out


def mpi_err_allreduce(count):
    """MAX-reduce the per-rank error count so every rank agrees on PASS.

    NB: MPI_Allreduce is SIX args (sendbuf, recvbuf, count, datatype, op,
    comm) -- NOT the Allgather layout; passing Allgather-style arg lists
    lands an int in the op slot and PMPI derefs 0x1+0x54 -> SEGV 0x55."""
    _, MPI_INT = _mpi_helper()
    mpi.MPI_Allreduce.argtypes = [ctypes.c_void_p, ctypes.c_void_p,
                                  ctypes.c_int, ctypes.c_void_p,
                                  ctypes.c_void_p, ctypes.c_void_p]
    mpi.MPI_Allreduce.restype = ctypes.c_int
    send = np.array([count], dtype=np.int32)
    recv = np.zeros(1, dtype=np.int32)
    rc = mpi.MPI_Allreduce(ctypes.c_void_p(send.ctypes.data),
                           ctypes.c_void_p(recv.ctypes.data), 1,
                           ctypes.c_void_p(MPI_INT),
                           ctypes.c_void_p(mpi._tri_mpi_max),
                           ctypes.c_void_p(comm))
    assert rc == 0
    return int(recv[0])


ctx = moe_a2a.create_moe_a2a_intra_node_context(
    num_experts=NUM_EXPERTS, hidden=H, dtype=DTYPE, max_tokens=T, topk=TOPK,
    rank=rank, num_ranks=size, mpi=mpi, comm=comm)

B_RECV = MRR * H * 2  # one buffer dump in bytes
BLOB = 2 * B_RECV + T * TOPK * 4 + 8  # recv + out + scatter + recv_total

failures = []
for it in range(3):
    # every rank generates EVERY rank's inputs identically on the CPU
    tok_cpu, topk_cpu, w_cpu = [], [], []
    for s in range(size):
        g = torch.Generator().manual_seed(31000 + 100 * it + 7 * s)
        tok_cpu.append(torch.randn(T, H, generator=g).to(DTYPE))
        topk_cpu.append(torch.randint(0, NUM_EXPERTS, (T, TOPK),
                                      generator=g).to(torch.int32))
        w_cpu.append(torch.rand(T, TOPK, generator=g) + 0.5)  # fp32
    tokens = tok_cpu[rank].cuda()
    topk_idx = topk_cpu[rank].cuda()
    weights = w_cpu[rank].cuda()

    recv_total_t = moe_a2a.moe_dispatch_intra_node_op(tokens, topk_idx, ctx)
    recv_total = int(recv_total_t.item())  # host copy for the audit below

    k_expert_fanout[(recv_total,)](ctx.recv_ptr, ctx.out_ptr,
                                   H=H, BLOCK=1024, num_warps=4)

    output = torch.empty(T, H, dtype=DTYPE, device="cuda")
    moe_a2a.moe_combine_intra_node_op(ctx, topk_idx, output, weights)
    torch.cuda.synchronize()

    # ---- pack [recv | out | scatter | recv_total], gather on all, rank 0
    dump = torch.empty(MRR * H, dtype=DTYPE, device="cuda")
    n = MRR * H
    k_dump[(triton.cdiv(n, 4096),)](ctx.recv_ptr, dump, n, BLOCK=4096)
    blob = bytearray(dump.cpu().numpy().tobytes())
    k_dump[(triton.cdiv(n, 4096),)](ctx.out_ptr, dump, n, BLOCK=4096)
    blob += dump.cpu().numpy().tobytes()
    del dump
    torch.cuda.empty_cache()
    blob += ctx.scatter_idx[:T].cpu().numpy().astype(np.int32).tobytes()
    blob += np.array([recv_total], dtype=np.int64).tobytes()
    blob += output.cpu().numpy().tobytes()  # appended tail (see offset)
    blob = np.frombuffer(bytes(blob), dtype=np.uint8)
    STRIDE = blob.nbytes  # BLOB + T*H*2 per rank in the gathered buffer
    allb = mpi_allgather_bytes(blob)

    errs = []
    if rank == 0:
        f_recv, f_out, f_sc, f_rt, f_out_t = [], [], [], [], []
        for s in range(size):
            b = allb[s * STRIDE:(s + 1) * STRIDE]
            f_recv.append(np.frombuffer(b[:B_RECV], np.float16)
                          .reshape(MRR, H))
            o0 = B_RECV
            f_out.append(np.frombuffer(b[o0:o0 + B_RECV], np.float16)
                         .reshape(MRR, H))
            s0 = 2 * B_RECV
            f_sc.append(np.frombuffer(
                b[s0:s0 + T * TOPK * 4], np.int32).reshape(T, TOPK))
            f_rt.append(int(np.frombuffer(b[2 * B_RECV + T * TOPK * 4:
                                           2 * B_RECV + T * TOPK * 4 + 8],
                                          np.int64)[0]))
            t0 = BLOB
            f_out_t.append(np.frombuffer(
                allb[s * STRIDE + t0:s * STRIDE + t0 + T * H * 2],
                np.float16).reshape(T, H))

        # 1) routing + content: recv_d[sc_s[t,j]] == tokens_s[t]
        for s in range(size):
            tk, sc = tok_cpu[s].numpy(), f_sc[s]
            for t in range(T):
                for j in range(TOPK):
                    d = int(topk_cpu[s].numpy()[t, j]) // EPR
                    if not np.array_equal(f_recv[d][int(sc[t, j])], tk[t]):
                        errs.append(f"it{it} s{s} t{t} j{j} recv mismatch")
                        break
                else:
                    continue
                break
        # 2) bijection: global (dst, row) unique and < dst recv_total
        if not errs:
            pairs = []
            for s in range(size):
                tk_idx, sc = topk_cpu[s].numpy(), f_sc[s]
                for t in range(T):
                    for j in range(TOPK):
                        pairs.append((int(tk_idx[t, j]) // EPR,
                                      int(sc[t, j])))
            if len(set(pairs)) != len(pairs):
                errs.append(f"it{it} scatter rows not unique")
            elif any(r >= f_rt[d] for (d, r) in pairs):
                errs.append(f"it{it} scatter row >= dst recv_total")
        # 3) fake expert phase: same f16-in/fp32/f16-out rounding path as
        #    the kernel -> must be BIT-exact
        if not errs:
            for d in range(size):
                rt = f_rt[d]
                if rt == 0:
                    continue
                rows = np.arange(rt, dtype=np.int64)
                exp = (f_recv[d][:rt].astype(np.float32) * 2.0
                       + (rows % 5)[:, None].astype(np.float32)
                       ).astype(np.float16)
                if not np.array_equal(f_out[d][:rt], exp):
                    md = np.abs(f_out[d][:rt].astype(np.float32)
                                - exp.astype(np.float32)).max()
                    errs.append(f"it{it} dst{d} expert out md={md}")
        # 4) combine sum: fp32 reference over the SAME fp16 expert outputs
        #    the kernel read (v_j[t] = expert_out[dst(t,j)][scatter[t,j]]),
        #    RELATIVE tolerance (row indices reach 16K -> fp16 ulp 32)
        if not errs:
            for s in range(size):
                sc, w = f_sc[s], w_cpu[s].numpy()
                tk_idx = topk_cpu[s].numpy()
                exp = np.zeros((T, H), dtype=np.float32)
                for j in range(TOPK):
                    for t in range(T):
                        dd = int(tk_idx[t, j]) // EPR
                        vj = f_out[dd][int(sc[t, j])].astype(np.float32)
                        exp[t] += w[t, j] * vj
                got = f_out_t[s].astype(np.float32)
                md = np.abs(got - exp).max()
                rel = md / max(1.0, np.abs(exp).max())
                if rel > 1e-2:
                    errs.append(f"it{it} rank{s} combine md={md} rel={rel}")
                else:
                    print(f"[rank 0] it{it} rank{s}: combine md={md:.3e} "
                          f"(rel {rel:.1e}) recv_total={f_rt[s]}", flush=True)

    n_err = mpi_err_allreduce(len(errs))
    if n_err:
        failures.append(f"it{it}")
        print(f"[rank {rank}] it{it}: FAIL ({n_err} checks)"
              f"{'' if rank else ' ' + '; '.join(errs[:3])}", flush=True)

ok = not failures
print(f"[rank {rank}] test_moe_a2a: {'PASS' if ok else 'FAIL'}", flush=True)

dcu_state()["host"].finalize()
mpi.MPI_Barrier.argtypes = [ctypes.c_void_p]
mpi.MPI_Barrier.restype = ctypes.c_int
mpi.MPI_Barrier(ctypes.c_void_p(comm))
mpi.MPI_Finalize()
os._exit(0 if ok else 1)
