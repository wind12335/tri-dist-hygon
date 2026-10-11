"""Cross-process IPC helpers for DCU (hipIpcGetMemHandle/OpenMemHandle).

Data plane of the DCU port: buffers are raw hipMalloc allocations (NOT
torch tensors -- torch's caching allocator may hand out interior pointers
into a larger hipMalloc block, and hipIpcGetMemHandle wants the allocation
base; the upstream C++ helper hipc_create_tensor_list hipMallocs its own
memory for the same reason). Kernels tl.store / tl.load through the raw
peer VAs (verified on 2x BW1000: direct tl.store into the peer mapping
works; sys-scope tl.atomic_cas on remote IPC memory HANGS, so signals ride
dushmem atomics instead -- see triton_dist/kernels/dcu/common_ops.py).

Handle exchange rides the same MPI comm used for the dushmem bootstrap
(torch bundles its own renamed-soname OpenMPI; keep MPI init before
import torch or the two runtimes interpose -- see pydushmem.host).
"""
import ctypes

HIP_LIB = "/opt/dtk-26.04/lib/libamdhip64.so"  # symlink -> libgalaxyhip.so.5


class IpcMemHandle(ctypes.Structure):
    _fields_ = [("reserved", ctypes.c_char * 64)]  # HIP_IPC_HANDLE_SIZE


def hip_lib():
    hip = ctypes.CDLL(HIP_LIB, mode=ctypes.RTLD_GLOBAL)
    hip.hipMalloc.restype = ctypes.c_int
    hip.hipMalloc.argtypes = [ctypes.POINTER(ctypes.c_void_p), ctypes.c_size_t]
    hip.hipMemset.restype = ctypes.c_int
    hip.hipMemset.argtypes = [ctypes.c_void_p, ctypes.c_int, ctypes.c_size_t]
    hip.hipFree.restype = ctypes.c_int
    hip.hipFree.argtypes = [ctypes.c_void_p]
    hip.hipIpcGetMemHandle.restype = ctypes.c_int
    hip.hipIpcGetMemHandle.argtypes = [ctypes.POINTER(IpcMemHandle),
                                       ctypes.c_void_p]
    hip.hipIpcOpenMemHandle.restype = ctypes.c_int
    hip.hipIpcOpenMemHandle.argtypes = [ctypes.POINTER(ctypes.c_void_p),
                                        IpcMemHandle, ctypes.c_uint]
    hip.hipIpcCloseMemHandle.restype = ctypes.c_int
    hip.hipIpcCloseMemHandle.argtypes = [ctypes.c_void_p]
    return hip


def make_allgather_bytes(mpi, comm, size):
    """Returns allgather(payload: bytes) -> bytes over MPI_BYTE."""
    mpi.MPI_Allgather.argtypes = [ctypes.c_void_p, ctypes.c_int, ctypes.c_void_p,
                                  ctypes.c_void_p, ctypes.c_int, ctypes.c_void_p,
                                  ctypes.c_void_p]
    mpi.MPI_Allgather.restype = ctypes.c_int
    mpi_byte = ctypes.c_void_p(
        ctypes.addressof(ctypes.c_int.in_dll(mpi, "ompi_mpi_byte")))

    def allgather_bytes(payload: bytes) -> bytes:
        n = len(payload)
        src = ctypes.create_string_buffer(payload, n)
        dst = ctypes.create_string_buffer(n * size)
        rc = mpi.MPI_Allgather(ctypes.cast(src, ctypes.c_void_p), n, mpi_byte,
                               ctypes.cast(dst, ctypes.c_void_p), n, mpi_byte,
                               ctypes.c_void_p(comm))
        assert rc == 0, f"MPI_Allgather rc={rc}"
        return dst.raw

    return allgather_bytes


def hip_malloc(hip, nbytes, zero=False):
    """Raw allocation, deliberately NOT zeroed: peer IPC stores reach DRAM
    but do not invalidate this GPU's L2 lines, so memset-populated zero
    lines make the first local read of peer-written data return zeros
    (observed on 2x BW1000). Producers must fully overwrite before any
    consumer read (barrier-ordered), like the upstream hipMalloc+from_blob."""
    p = ctypes.c_void_p()
    assert hip.hipMalloc(ctypes.byref(p), nbytes) == 0
    if zero:
        assert hip.hipMemset(p, 0, nbytes) == 0
    return p.value


class IpcPtrList:
    """One raw hipMalloc buffer per rank. ptrs[i] is the VA rank i's buffer
    maps to on THIS rank (own slot = own allocation). buffers is an int64
    device tensor holding ptrs -- feed it straight to kernels like the
    upstream scatter_bufs_ptr. free() must run before exit."""

    def __init__(self, nbytes, mpi, comm, size, rank):
        import torch  # caller has already ordered MPI before torch import
        self.hip = hip_lib()
        self.nbytes = nbytes
        self.own = hip_malloc(self.hip, nbytes)
        h = IpcMemHandle()
        rc = self.hip.hipIpcGetMemHandle(ctypes.byref(h),
                                         ctypes.c_void_p(self.own))
        assert rc == 0, f"hipIpcGetMemHandle rc={rc}"
        allgather_bytes = make_allgather_bytes(mpi, comm, size)
        blob = allgather_bytes(
            ctypes.string_at(ctypes.addressof(h), 64))  # NUL-safe raw copy
        self.ptrs = [0] * size
        self._opened = []
        for i in range(size):
            if i != rank:
                ph = IpcMemHandle.from_buffer_copy(blob[i * 64:(i + 1) * 64])
                p = ctypes.c_void_p()
                rc = self.hip.hipIpcOpenMemHandle(ctypes.byref(p), ph, 1)
                assert rc == 0, f"hipIpcOpenMemHandle(rank {i}) rc={rc}"
                self.ptrs[i] = p.value
                self._opened.append(p.value)
            else:
                self.ptrs[i] = self.own
        self.buffers = torch.tensor(self.ptrs, dtype=torch.int64, device="cuda")

    def free(self):
        for p in self._opened:
            self.hip.hipIpcCloseMemHandle(ctypes.c_void_p(p))
        self._opened = []
        self.hip.hipFree(ctypes.c_void_p(self.own))
        self.own = None
