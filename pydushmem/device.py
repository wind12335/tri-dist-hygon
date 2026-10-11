"""Triton extern declarations for dushmem device pt2pt + wrappers.

Importing this module applies the llc fixup (required before compiling any
kernel that links the bundle). Link kernels with pydushmem.EXTERN_LIBS.

Conventions:
- pointers are passed as int64 scalars (triton pointer_type is unhashable as
  an extern_elementwise dispatch key); the C wrappers cast back to void*
- wrappers return int32 0 (extern_elementwise requires a dtype return type;
  the underlying device APIs are void)
- symbols keep the dushmem_* prefix (hcu.need_extern_lib prefix rule)
"""
import os

from . import llc_fixup

llc_fixup.apply()

from triton.language import core  # noqa: E402

i32 = core.dtype("int32")
i64 = core.dtype("int64")

BC_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                       "lib", "pydushmem.bc")


@core.extern
def dushmem_my_pe(_builder=None):
    return core.extern_elementwise(
        "", "", [],
        {(): ("dushmem_my_pe_wrapper", i32)},
        is_pure=False, _builder=_builder)


@core.extern
def dushmem_n_pes(_builder=None):
    return core.extern_elementwise(
        "", "", [],
        {(): ("dushmem_n_pes_wrapper", i32)},
        is_pure=True, _builder=_builder)


@core.extern
def dushmem_putmem_nbi(dest, source, nelems, pe, _builder=None):
    return core.extern_elementwise(
        "", "",
        [core.cast(dest, i64, _builder=_builder),
         core.cast(source, i64, _builder=_builder),
         core.cast(nelems, i64, _builder=_builder),
         core.cast(pe, i32, _builder=_builder)],
        {(i64, i64, i64, i32): ("dushmem_putmem_nbi_wrapper", i32)},
        is_pure=False, _builder=_builder)


@core.extern
def dushmem_getmem_nbi(dest, source, nelems, pe, _builder=None):
    return core.extern_elementwise(
        "", "",
        [core.cast(dest, i64, _builder=_builder),
         core.cast(source, i64, _builder=_builder),
         core.cast(nelems, i64, _builder=_builder),
         core.cast(pe, i32, _builder=_builder)],
        {(i64, i64, i64, i32): ("dushmem_getmem_nbi_wrapper", i32)},
        is_pure=False, _builder=_builder)


@core.extern
def dushmem_putmem_signal(dest, source, nelems, sig_addr, signal, sig_op, pe,
                          _builder=None):
    """Blocking put + dest-side uint64 signal (sig_op 9=SET, 10=ADD)."""
    return core.extern_elementwise(
        "", "",
        [core.cast(dest, i64, _builder=_builder),
         core.cast(source, i64, _builder=_builder),
         core.cast(nelems, i64, _builder=_builder),
         core.cast(sig_addr, i64, _builder=_builder),
         core.cast(signal, i64, _builder=_builder),
         core.cast(sig_op, i32, _builder=_builder),
         core.cast(pe, i32, _builder=_builder)],
        {(i64, i64, i64, i64, i64, i32, i32): ("dushmem_putmem_signal_wrapper", i32)},
        is_pure=False, _builder=_builder)


@core.extern
def dushmem_quiet(_builder=None):
    return core.extern_elementwise(
        "", "", [],
        {(): ("dushmem_quiet_wrapper", i32)},
        is_pure=False, _builder=_builder)


@core.extern
def dushmem_fence(_builder=None):
    return core.extern_elementwise(
        "", "", [],
        {(): ("dushmem_fence_wrapper", i32)},
        is_pure=False, _builder=_builder)


@core.extern
def dushmem_ptr(ptr, pe, _builder=None):
    """PE-local virtual address of a symmetric object, as int64."""
    return core.extern_elementwise(
        "", "",
        [core.cast(ptr, i64, _builder=_builder),
         core.cast(pe, i32, _builder=_builder)],
        {(i64, i32): ("dushmem_ptr_wrapper", i64)},
        is_pure=True, _builder=_builder)


@core.extern
def dushmem_int_atomic_add(dest, value, pe, _builder=None):
    return core.extern_elementwise(
        "", "",
        [core.cast(dest, i64, _builder=_builder),
         core.cast(value, i32, _builder=_builder),
         core.cast(pe, i32, _builder=_builder)],
        {(i64, i32, i32): ("dushmem_int_atomic_add_wrapper", i32)},
        is_pure=False, _builder=_builder)


@core.extern
def dushmem_int_atomic_set(dest, value, pe, _builder=None):
    return core.extern_elementwise(
        "", "",
        [core.cast(dest, i64, _builder=_builder),
         core.cast(value, i32, _builder=_builder),
         core.cast(pe, i32, _builder=_builder)],
        {(i64, i32, i32): ("dushmem_int_atomic_set_wrapper", i32)},
        is_pure=False, _builder=_builder)


@core.extern
def dushmem_uint64_atomic_add(dest, value, pe, _builder=None):
    return core.extern_elementwise(
        "", "",
        [core.cast(dest, i64, _builder=_builder),
         core.cast(value, i64, _builder=_builder),
         core.cast(pe, i32, _builder=_builder)],
        {(i64, i64, i32): ("dushmem_uint64_atomic_add_wrapper", i32)},
        is_pure=False, _builder=_builder)


@core.extern
def dushmem_uint64_atomic_set(dest, value, pe, _builder=None):
    return core.extern_elementwise(
        "", "",
        [core.cast(dest, i64, _builder=_builder),
         core.cast(value, i64, _builder=_builder),
         core.cast(pe, i32, _builder=_builder)],
        {(i64, i64, i32): ("dushmem_uint64_atomic_set_wrapper", i32)},
        is_pure=False, _builder=_builder)


@core.extern
def dushmem_uint64_atomic_fetch_add(dest, value, pe, _builder=None):
    return core.extern_elementwise(
        "", "",
        [core.cast(dest, i64, _builder=_builder),
         core.cast(value, i64, _builder=_builder),
         core.cast(pe, i32, _builder=_builder)],
        {(i64, i64, i32): ("dushmem_uint64_atomic_fetch_add_wrapper", i64)},
        is_pure=False, _builder=_builder)


@core.extern
def dushmem_barrier_all(_builder=None):
    return core.extern_elementwise(
        "", "", [],
        {(): ("dushmem_barrier_all_wrapper", i32)},
        is_pure=False, _builder=_builder)
