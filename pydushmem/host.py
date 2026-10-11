"""ctypes bindings for libdushmem_host.so (DUSHMEM 3.x, NVSHMEM fork).

Struct layouts and versions verified against DTK 26.04 headers
(include/dushmem/{dushmem.h,host/dushmemx_api.h}); all struct versions are
(1<<16) + sizeof(struct).

Init ordering matters: dushmemx_hostlib_init_attr RESETS the host mirror of
dushmemi_device_state to INVALID and only dushmem_malloc (via
dushmemi_check_state_and_init -> dushmemi_init_device_state) fills it with
real values (mype/npes/heap/proxy). dushmemx_cumodule_init memcpys that
mirror into every registered kernel module, so malloc() must run between
hostlib_init_attr() and the first cumodule_init() -- see init().
"""
import ctypes
import os
import time

DUSHMEM_LIBDIR = os.environ.get(
    "DUSHMEM_LIBDIR", "/opt/dtk-26.04/lib/dushmem")
DUSHMEMX_INIT_WITH_UNIQUEID = 1 << 3
DUSHMEMX_INIT_WITH_MPI_COMM = 1 << 1
# OpenMPI MPI_Comm/MPI_Datatype are struct pointers; the C macros are just
# &ompi_mpi_comm_world / &ompi_mpi_int (both exported data symbols).
MPI_LIB = os.environ.get("PYDUSHMEM_MPI_LIB", "/opt/mpi/lib/libmpi.so")


def _sym_addr(lib, name):
    return ctypes.addressof(ctypes.c_int.in_dll(lib, name))


def mpi_init():
    """Load OpenMPI via ctypes and MPI_Init the process (call before
    init_mpi; must be running under mpirun). Returns (lib, comm) with comm
    the C MPI_Comm handle for MPI_COMM_WORLD."""
    mpi = ctypes.CDLL(MPI_LIB, mode=ctypes.RTLD_GLOBAL)
    mpi.MPI_Init.argtypes = [ctypes.c_void_p, ctypes.c_void_p]
    mpi.MPI_Init.restype = ctypes.c_int
    rc = mpi.MPI_Init(None, None)
    assert rc == 0, f"MPI_Init rc={rc} (must run under mpirun)"
    comm = _sym_addr(mpi, "ompi_mpi_comm_world")
    assert comm, "ompi_mpi_comm_world symbol not found"
    return mpi, comm


def mpi_type_int(mpi):
    """C handle for MPI_INT."""
    return _sym_addr(mpi, "ompi_mpi_int")


def mpi_comm_rank(mpi, comm):
    r = ctypes.c_int(-1)
    rc = mpi.MPI_Comm_rank(ctypes.c_void_p(comm), ctypes.byref(r))
    assert rc == 0 and r.value >= 0, f"MPI_Comm_rank rc={rc} rank={r.value}"
    return r.value


def mpi_comm_size(mpi, comm):
    n = ctypes.c_int(-1)
    rc = mpi.MPI_Comm_size(ctypes.c_void_p(comm), ctypes.byref(n))
    assert rc == 0 and n.value > 0, f"MPI_Comm_size rc={rc} size={n.value}"
    return n.value


class UidV1(ctypes.Structure):
    _fields_ = [("version", ctypes.c_int), ("internal", ctypes.c_char * 124)]


class UidArgsV1(ctypes.Structure):
    _fields_ = [("version", ctypes.c_int), ("_pad", ctypes.c_int),
                ("id", ctypes.POINTER(UidV1)),
                ("myrank", ctypes.c_int), ("nranks", ctypes.c_int)]


class InitArgsV1(ctypes.Structure):
    _fields_ = [("version", ctypes.c_int), ("_pad", ctypes.c_int),
                ("uid_args", UidArgsV1), ("content", ctypes.c_char * 96)]


class InitAttrV1(ctypes.Structure):
    _fields_ = [("version", ctypes.c_int), ("_pad", ctypes.c_int),
                ("mpi_comm", ctypes.c_void_p), ("args", InitArgsV1)]


assert ctypes.sizeof(UidV1) == 128
assert ctypes.sizeof(UidArgsV1) == 24
assert ctypes.sizeof(InitArgsV1) == 128
assert ctypes.sizeof(InitAttrV1) == 144

UID_SIZE = ctypes.sizeof(UidV1)
UID_PADDING = 124


class DushmemHost:
    """Low-level host handle. One per process (dushmem global state)."""

    def __init__(self, libdir=DUSHMEM_LIBDIR, preload_bootstrap=True):
        # preload the uid bootstrap plugin so libdushmem's dlopen-by-name
        # finds it; both RTLD_GLOBAL so triton-loaded modules resolve nothing
        # from these but the process keeps one copy.
        self.libdir = libdir
        if preload_bootstrap:
            ctypes.CDLL(f"{libdir}/dushmem_bootstrap_uid.so", mode=ctypes.RTLD_GLOBAL)
        self.lib = ctypes.CDLL(f"{libdir}/libdushmem_host.so", mode=ctypes.RTLD_GLOBAL)
        lib = self.lib
        lib.dushmemx_get_uniqueid.restype = ctypes.c_int
        lib.dushmemx_get_uniqueid.argtypes = [ctypes.POINTER(UidV1)]
        lib.dushmemx_set_attr_uniqueid_args.restype = ctypes.c_int
        lib.dushmemx_set_attr_uniqueid_args.argtypes = [
            ctypes.c_int, ctypes.c_int, ctypes.POINTER(UidV1),
            ctypes.POINTER(InitAttrV1)]
        lib.dushmemx_hostlib_init_attr.restype = ctypes.c_int
        lib.dushmemx_hostlib_init_attr.argtypes = [
            ctypes.c_uint, ctypes.POINTER(InitAttrV1)]
        lib.dushmemx_hostlib_finalize.restype = None
        lib.dushmemx_hostlib_finalize.argtypes = []
        lib.dushmemx_cumodule_init.restype = ctypes.c_int
        lib.dushmemx_cumodule_init.argtypes = [ctypes.c_void_p]
        lib.dushmemx_cumodule_finalize.restype = ctypes.c_int
        lib.dushmemx_cumodule_finalize.argtypes = [ctypes.c_void_p]

        lib.dushmem_malloc.restype = ctypes.c_void_p
        lib.dushmem_malloc.argtypes = [ctypes.c_size_t]
        lib.dushmem_calloc.restype = ctypes.c_void_p
        lib.dushmem_calloc.argtypes = [ctypes.c_size_t, ctypes.c_size_t]
        lib.dushmem_align.restype = ctypes.c_void_p
        lib.dushmem_align.argtypes = [ctypes.c_size_t, ctypes.c_size_t]
        lib.dushmem_free.restype = None
        lib.dushmem_free.argtypes = [ctypes.c_void_p]
        # NB: dushmem_realloc is declared in dushmem_api.h but not exported
        # by this build of libdushmem_host.so.

        lib.dushmem_my_pe.restype = ctypes.c_int
        lib.dushmem_my_pe.argtypes = []
        lib.dushmem_n_pes.restype = ctypes.c_int
        lib.dushmem_n_pes.argtypes = []
        lib.dushmem_ptr.restype = ctypes.c_void_p
        lib.dushmem_ptr.argtypes = [ctypes.c_void_p, ctypes.c_int]

        lib.dushmem_putmem.restype = None
        lib.dushmem_putmem.argtypes = [
            ctypes.c_void_p, ctypes.c_void_p, ctypes.c_size_t, ctypes.c_int]
        lib.dushmem_putmem_nbi.restype = None
        lib.dushmem_putmem_nbi.argtypes = [
            ctypes.c_void_p, ctypes.c_void_p, ctypes.c_size_t, ctypes.c_int]
        lib.dushmem_getmem.restype = None
        lib.dushmem_getmem.argtypes = [
            ctypes.c_void_p, ctypes.c_void_p, ctypes.c_size_t, ctypes.c_int]
        lib.dushmem_getmem_nbi.restype = None
        lib.dushmem_getmem_nbi.argtypes = [
            ctypes.c_void_p, ctypes.c_void_p, ctypes.c_size_t, ctypes.c_int]
        lib.dushmem_quiet.restype = None
        lib.dushmem_quiet.argtypes = []
        lib.dushmem_fence.restype = None
        lib.dushmem_fence.argtypes = []
        lib.dushmem_barrier_all.restype = None
        lib.dushmem_barrier_all.argtypes = []
        self._inited = False

    # ---- lifecycle -------------------------------------------------------
    def get_uniqueid(self):
        """124 raw uid bytes for out-of-band exchange (rank 0 generates).
        NB: uid.internal converts to bytes STOPPING AT THE FIRST NUL -- the
        sockaddr at internal[0:8] contains NULs, so copy the raw storage."""
        uid = UidV1()
        rc = self.lib.dushmemx_get_uniqueid(ctypes.byref(uid))
        assert rc == 0, f"dushmemx_get_uniqueid rc={rc}"
        return ctypes.string_at(ctypes.addressof(uid) + 4, UID_PADDING)

    def init(self, rank=0, nranks=1, uid=None):
        """uid-bootstrap init. For nranks>1 all ranks must pass the SAME uid
        bytes (from get_uniqueid on rank 0, exchanged out of band)."""
        if self._inited:
            return
        uid_struct = UidV1()
        if uid is None:
            if nranks != 1:
                raise ValueError("multi-PE init requires shared uid bytes")
            rc = self.lib.dushmemx_get_uniqueid(ctypes.byref(uid_struct))
            assert rc == 0, f"dushmemx_get_uniqueid rc={rc}"
        else:
            uid_struct.internal = uid

        attr = InitAttrV1()
        attr.version = (1 << 16) + ctypes.sizeof(InitAttrV1)
        attr.args.version = (1 << 16) + ctypes.sizeof(InitArgsV1)
        attr.args.uid_args.version = (1 << 16) + ctypes.sizeof(UidArgsV1)
        rc = self.lib.dushmemx_set_attr_uniqueid_args(
            rank, nranks, ctypes.byref(uid_struct), ctypes.byref(attr))
        assert rc == 0, f"dushmemx_set_attr_uniqueid_args rc={rc}"
        rc = self.lib.dushmemx_hostlib_init_attr(
            DUSHMEMX_INIT_WITH_UNIQUEID, ctypes.byref(attr))
        assert rc == 0, f"dushmemx_hostlib_init_attr rc={rc}"

        # LAZY-INIT TRIGGER: fill the device-state mirror (gdb-established:
        # only dushmem_malloc reaches dushmemi_init_device_state; my_pe does
        # not). Must precede every dushmemx_cumodule_init.
        scratch = self.lib.dushmem_malloc(1024)
        assert scratch, "dushmem_malloc (lazy state init) failed"
        self.lib.dushmem_free(ctypes.c_void_p(scratch))

        assert self.my_pe() == rank and self.n_pes() == nranks, \
            f"init mismatch: my_pe={self.my_pe()} n_pes={self.n_pes()}"
        self._inited = True

    def init_mpi(self, comm):
        """MPI-comm bootstrap init (DUSHMEMX_INIT_WITH_MPI_COMM). Requires the
        process to be MPI-initialized (e.g. under mpirun); `comm` is the raw
        MPI_Comm handle. The uid bootstrap's set_attr path NUL-truncates the
        sockaddr inside uid.internal (DTK 26.04 bug: AF_INET family bytes
        02 00 stop the internal strcpy at byte 1, so the listener binds
        0.0.0.0:0 -> ephemeral port and peers dial a dead port) -- MPI mode
        sidesteps uid exchange entirely."""
        if self._inited:
            return
        attr = InitAttrV1()
        attr.version = (1 << 16) + ctypes.sizeof(InitAttrV1)
        attr.args.version = (1 << 16) + ctypes.sizeof(InitArgsV1)
        attr.args.uid_args.version = (1 << 16) + ctypes.sizeof(UidArgsV1)
        # NB: mpi_comm is a POINTER TO an MPI_Comm variable (NVSHMEM API:
        # attr.mpi_comm = &comm), not the MPI_Comm itself.
        comm_holder = ctypes.c_void_p(comm)
        attr.mpi_comm = ctypes.cast(ctypes.byref(comm_holder), ctypes.c_void_p)
        rc = self.lib.dushmemx_hostlib_init_attr(
            DUSHMEMX_INIT_WITH_MPI_COMM, ctypes.byref(attr))
        assert rc == 0, f"dushmemx_hostlib_init_attr(mpi) rc={rc}"
        # LAZY-INIT TRIGGER: same as uid path -- fill the device-state mirror
        # before any dushmemx_cumodule_init.
        scratch = self.lib.dushmem_malloc(1024)
        assert scratch, "dushmem_malloc (lazy state init) failed"
        self.lib.dushmem_free(ctypes.c_void_p(scratch))
        self._inited = True

    def finalize(self):
        if self._inited:
            self.lib.dushmemx_hostlib_finalize()
            self._inited = False

    # ---- symmetric heap ---------------------------------------------------
    def malloc(self, size):
        return self._require_addr(self.lib.dushmem_malloc(size), "malloc")

    def calloc(self, count, size):
        return self._require_addr(self.lib.dushmem_calloc(count, size), "calloc")

    def align(self, alignment, size):
        return self._require_addr(self.lib.dushmem_align(alignment, size), "align")

    def free(self, ptr):
        self.lib.dushmem_free(ctypes.c_void_p(ptr))

    @staticmethod
    def _require_addr(addr, what):
        assert addr, f"dushmem_{what} returned NULL"
        return addr

    # ---- PE info / address translation ------------------------------------
    def my_pe(self):
        return self.lib.dushmem_my_pe()

    def n_pes(self):
        return self.lib.dushmem_n_pes()

    def ptr(self, addr, pe):
        return self.lib.dushmem_ptr(ctypes.c_void_p(addr), pe)

    # ---- host pt2pt (addresses are device/symmetric VAs) -------------------
    def putmem(self, dst, src, nbytes, pe):
        self.lib.dushmem_putmem(ctypes.c_void_p(dst), ctypes.c_void_p(src),
                                nbytes, pe)

    def putmem_nbi(self, dst, src, nbytes, pe):
        self.lib.dushmem_putmem_nbi(ctypes.c_void_p(dst), ctypes.c_void_p(src),
                                    nbytes, pe)

    def getmem(self, dst, src, nbytes, pe):
        self.lib.dushmem_getmem(ctypes.c_void_p(dst), ctypes.c_void_p(src),
                                nbytes, pe)

    def getmem_nbi(self, dst, src, nbytes, pe):
        self.lib.dushmem_getmem_nbi(ctypes.c_void_p(dst), ctypes.c_void_p(src),
                                    nbytes, pe)

    def quiet(self):
        self.lib.dushmem_quiet()

    def fence(self):
        self.lib.dushmem_fence()

    def barrier_all(self):
        self.lib.dushmem_barrier_all()

    # ---- triton interop -----------------------------------------------------
    def cumodule_init(self, module):
        """Register a loaded hipModule (triton CompiledKernel.module or raw
        int) with dushmem; pushes device state into its globals."""
        rc = self.lib.dushmemx_cumodule_init(ctypes.c_void_p(module))
        assert rc == 0, f"dushmemx_cumodule_init rc={rc}"

    def cumodule_finalize(self, module):
        rc = self.lib.dushmemx_cumodule_finalize(ctypes.c_void_p(module))
        assert rc == 0, f"dushmemx_cumodule_finalize rc={rc}"


def init_cluster_from_file(path, rank, nranks, timeout=120.0):
    """File-based uid exchange for multi-process (S4): rank 0 writes the uid,
    others poll for it. Returns an inited DushmemHost. Caller deletes `path`
    after all ranks are up."""
    host = DushmemHost()
    if rank == 0:
        uid = host.get_uniqueid()
        tmp = f"{path}.{rank}"
        with open(tmp, "wb") as f:
            f.write(uid)
        os.replace(tmp, path)
        host.init(rank, nranks, uid)
        return host
    deadline = time.time() + timeout
    while not os.path.exists(path):
        if time.time() > deadline:
            raise TimeoutError(f"uid file {path} never appeared")
        time.sleep(0.1)
    with open(path, "rb") as f:
        uid = f.read()
    host.init(rank, nranks, uid)
    return host
