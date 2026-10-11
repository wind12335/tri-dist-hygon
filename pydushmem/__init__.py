"""pydushmem: DUSHMEM host bindings + triton device externs for Hygon DCU.

Single-card acceptance: python -m pydushmem.selftest (or selftest.py)
S1 chain: pydushmem.init() -> malloc -> triton kernel with device externs ->
dushmemx_cumodule_init -> launch -> put/get loopback.
"""
import os

from .host import DushmemHost, UID_SIZE, init_cluster_from_file  # noqa: F401

_host = None


def get_host():
    """Process-wide host handle (dushmem state is process-global)."""
    global _host
    if _host is None:
        _host = DushmemHost()
    return _host


def init(rank=0, nranks=1, uid=None):
    """Init dushmem (uid bootstrap). For nranks>1 pass shared uid bytes on
    every rank (see get_uniqueid / init_cluster_from_file)."""
    get_host().init(rank, nranks, uid)


def get_uniqueid():
    """124 raw uid bytes from rank 0 for out-of-band exchange."""
    return get_host().get_uniqueid()


def finalize():
    if _host is not None:
        _host.finalize()


def _delegated(name):
    def fn(*args, **kwargs):
        return getattr(get_host(), name)(*args, **kwargs)
    fn.__name__ = name
    return fn


malloc = _delegated("malloc")
calloc = _delegated("calloc")
align = _delegated("align")
free = _delegated("free")
my_pe = _delegated("my_pe")
n_pes = _delegated("n_pes")
ptr = _delegated("ptr")
putmem = _delegated("putmem")
putmem_nbi = _delegated("putmem_nbi")
getmem = _delegated("getmem")
getmem_nbi = _delegated("getmem_nbi")
quiet = _delegated("quiet")
fence = _delegated("fence")
barrier_all = _delegated("barrier_all")
cumodule_init = _delegated("cumodule_init")
cumodule_finalize = _delegated("cumodule_finalize")


def extern_libs():
    """extern_libs argument for triton kernel compile (lazy so torch/triton
    import stays out of host-only flows)."""
    from .device import BC_PATH
    return [("dushmem", BC_PATH)]


# back-compat alias used by early S1/S2 tests
EXTERN_LIBS = extern_libs
