"""S13 debug (2-card): RS v5 hang at M8192 -- the op ENQUEUE ITSELF wedges
(py-spy: hipMemcpyAsync of the 4B consumer ack never returns, host spinning
at 200% CPU, GPU idle). A full stream ring-buffer means some GTE wait never
fired. This variant runs the enqueue in a daemon thread so the main thread
can poll stream states and dump the ready/arrival/ack planes while stuck.

mpirun --allow-run-as-root -np 2 --oversubscribe --bind-to none \
  -mca coll ^hcoll -x HSA_FORCE_FINE_GRAIN_PCIE=1 \
  python3 tests/dbg_rs_big.py [M] [N] [K] [slots] [lanes]
"""
import ctypes
import os
import sys
import threading
import time

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO)
os.environ.setdefault("TRITON_CACHE_DIR",
                      f"/tmp/tc_s13dbg_{os.environ.get('OMPI_COMM_WORLD_RANK', '0')}")

import pydushmem.host as pHost  # noqa: E402

mpi, comm = pHost.mpi_init()
rank = pHost.mpi_comm_rank(mpi, comm)
size = pHost.mpi_comm_size(mpi, comm)

hip = ctypes.CDLL("/opt/dtk-26.04/lib/libamdhip64.so")
hip.hipSetDevice.restype = ctypes.c_int
hip.hipStreamQuery.argtypes = [ctypes.c_void_p]
hip.hipStreamQuery.restype = ctypes.c_int
hip.hipMemcpy.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_size_t,
                          ctypes.c_int]
hip.hipMemcpy.restype = ctypes.c_int
assert hip.hipSetDevice(rank) == 0

import torch  # noqa: E402

torch.cuda.set_device(rank)

from triton_dist.kernels.dcu import gemm_rs_v5 as rs5  # noqa: E402

M = int(sys.argv[1]) if len(sys.argv) > 1 else 8192
N = int(sys.argv[2]) if len(sys.argv) > 2 else 28672
K = int(sys.argv[3]) if len(sys.argv) > 3 else 8192
SLOTS = int(sys.argv[4]) if len(sys.argv) > 4 else 4
LANES = int(sys.argv[5]) if len(sys.argv) > 5 else 2
DTYPE = torch.float16
SCALE = 0.02

torch.manual_seed(999 + rank)
b = (torch.randn(N, K, device="cuda") * SCALE).half()
a = (torch.randn(M, K, device="cuda") * SCALE).half()

ctx = rs5.create_gemm_rs_v5_context(
    max_M=M, N=N, output_dtype=DTYPE, rank=rank, num_ranks=size,
    mpi=mpi, comm=comm, stage_slots=SLOTS, comm_lanes=LANES)
print(f"[rank {rank}] ctx: CR={ctx.chunk_rows} chunks={ctx.num_chunks} "
      f"S={ctx.stage_slots} L={ctx.comm_lanes}", flush=True)

out = torch.empty(M // size, N, dtype=DTYPE, device="cuda")

state = {"enqueue_done": False, "exc": None}


def enqueuer():
    import traceback
    # a fresh thread's current device defaults to 0: on rank 1 that makes
    # torch.cuda.current_stream() return a device-0 stream and the op's
    # fork hipEventRecord fails with rc=400 (hipErrorInvalidHandle)
    hip.hipSetDevice(rank)
    torch.cuda.set_device(rank)
    try:
        rs5.gemm_rs_v5_op(a, b, out, ctx)  # enqueue only, never syncs
    except BaseException:  # noqa: BLE001
        state["exc"] = traceback.format_exc()
        print(f"[rank {rank}] ENQUEUE DIED:\n{state['exc']}", flush=True)
    state["enqueue_done"] = True


t_enq = threading.Thread(target=enqueuer, daemon=True)
t0 = time.time()
t_enq.start()

streams = {"compute": ctx.compute_stream,
           **{f"lane{i}": st for i, st in enumerate(ctx.lane_streams)},
           **{f"slot{i}": st for i, st in enumerate(ctx.slot_streams)}}
S = ctx.stage_slots
nflags = size * S


def dump(tag):
    # ready is a LOCAL torch tensor: .tolist() syncs on the default stream
    # (empty), so it reads the plane even while op streams are wedged.
    try:
        ready = ctx.ready.tolist()
    except Exception as e:  # noqa: BLE001
        ready = f"<{e!r}>"
    try:
        arr = torch.empty(nflags, dtype=torch.int32, device="cpu")
        ack = torch.empty(nflags, dtype=torch.int32, device="cpu")
        rc1 = hip.hipMemcpy(arr.data_ptr(), ctypes.c_void_p(ctx.arrival_ptr),
                            4 * nflags, 2)
        rc2 = hip.hipMemcpy(ack.data_ptr(), ctypes.c_void_p(ctx.ack_ptr),
                            4 * nflags, 2)
        print(f"[rank {rank}] {tag} ready={ready}", flush=True)
        if rc1 == 0 and rc2 == 0:
            rb = ctx.round_base
            print(f"[rank {rank}] {tag} arrival(src,slot)="
                  f"{arr.view(size, -1).tolist()}", flush=True)
            print(f"[rank {rank}] {tag} ack(dest,slot)="
                  f"{ack.view(size, -1).tolist()}", flush=True)
            print(f"[rank {rank}] {tag} round_base={rb} "
                  f"ticket(ck)=17+ck+1; arrival[src=1,s] should reach "
                  f"ticket(15)={rb + 16}; ack[d=1,s] likewise", flush=True)
        else:
            print(f"[rank {rank}] {tag} hipMemcpy rc1={rc1} rc2={rc2}",
                  flush=True)
    except Exception as e:  # noqa: BLE001
        print(f"[rank {rank}] {tag} dump failed: {e!r}", flush=True)


deadline = time.time() + 120
done = False
next_dump = time.time() + 20
while time.time() < deadline:
    states = {}
    for nm, st in streams.items():
        rc = hip.hipStreamQuery(ctypes.c_void_p(st.cuda_stream))
        states[nm] = 0 if rc == 0 else rc
    enq = "done" if state["enqueue_done"] else "STUCK"
    print(f"[rank {rank}] t+{time.time() - t0:5.1f}s enq={enq} "
          f"exc={state['exc']} {states}", flush=True)
    if all(v == 0 for v in states.values()) and state["enqueue_done"]:
        done = True
        break
    if time.time() >= next_dump:
        dump(f"t+{time.time() - t0:.0f}s")
        next_dump += 40
    time.sleep(5)

if done:
    print(f"[rank {rank}] ALL DRAINED at {time.time() - t0:.1f}s", flush=True)
    torch.cuda.synchronize()
    with torch.no_grad():
        partial = a.float() @ b.float().t()
        # reconstruct the peer's operands from the seed scheme (999+rank)
        torch.manual_seed(999 + (1 - rank))
        b_peer = (torch.randn(N, K, device="cuda") * SCALE).half()
        a_peer = (torch.randn(M, K, device="cuda") * SCALE).half()
        peer_partial = a_peer.float() @ b_peer.float().t()
        del a_peer, b_peer
        full = partial + peer_partial
    m_pr = M // size
    shard = full[rank * m_pr:(rank + 1) * m_pr]
    local_shard = partial[rank * m_pr:(rank + 1) * m_pr]
    d_full = (out.float() - shard).abs()
    d_local = (out.float() - local_shard).abs()
    CR = ctx.chunk_rows
    nch = ctx.num_chunks
    per_chunk_full = d_full.view(nch, CR // 128, 128, N).amax(dim=(1, 2, 3))
    per_chunk_local = d_local.view(nch, CR // 128, 128, N).amax(dim=(1, 2, 3))
    print(f"[rank {rank}] md vs FULL ref  = {d_full.max().item():.4f} "
          f"per-chunk={[f'{v:.3f}' for v in per_chunk_full.tolist()]}",
          flush=True)
    print(f"[rank {rank}] md vs LOCAL-only = {d_local.max().item():.4f} "
          f"(~0 means scatter contributed nothing)",
          flush=True)
    print(f"[rank {rank}] per-chunk LOCAL-only md="
          f"{[f'{v:.3f}' for v in per_chunk_local.tolist()]}", flush=True)
    rs5.finalize_gemm_rs_v5_context(ctx)
    from triton_dist.kernels.dcu.common_ops import dcu_state
    dcu_state()["host"].finalize()
    mpi.MPI_Barrier.argtypes = [ctypes.c_void_p]
    mpi.MPI_Barrier.restype = ctypes.c_int
    mpi.MPI_Barrier(ctypes.c_void_p(comm))
    mpi.MPI_Finalize()
    os._exit(0)

dump("FINAL")
print(f"[rank {rank}] WEDGED -- exiting hard", flush=True)
os._exit(3)
