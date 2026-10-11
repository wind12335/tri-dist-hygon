// S1: dushmem device pt2pt (put/get) wrappers for triton kernels.
// Bundle: this wrapper + init.bc (state globals) + xfer.bc (transfer_device
// implementations, self-contained: only LLVM-intrinsic declares).
// Compiled like s2b_wrapper.cu (see build_dushmem_bc.sh for flags).
#include <hip/hip_runtime.h>
#include <dushmem/dushmem.h>

extern "C" __device__ int __attribute__((visibility("default"))) dushmem_my_pe_wrapper() {
    return dushmem_my_pe();
}

extern "C" __device__ int __attribute__((visibility("default"))) dushmem_n_pes_wrapper() {
    return dushmem_n_pes();
}

// NOTE: extern decls pass pointers as int64 (triton pointer_type is not
// hashable as an extern_elementwise dispatch key); cast back here.
// Wrappers return int (not void): triton's extern_elementwise requires a
// dtype return value (ret_type.to_ir). Matches nvshmem's int-returning API.
extern "C" __device__ int __attribute__((visibility("default")))
dushmem_putmem_nbi_wrapper(unsigned long dest, unsigned long source, size_t nelems, int pe) {
    dushmem_putmem_nbi((void *)dest, (const void *)source, nelems, pe);
    return 0;
}

extern "C" __device__ int __attribute__((visibility("default")))
dushmem_getmem_nbi_wrapper(unsigned long dest, unsigned long source, size_t nelems, int pe) {
    dushmem_getmem_nbi((void *)dest, (const void *)source, nelems, pe);
    return 0;
}

// blocking put + dest-side signal in one op (data/signal ordering guaranteed
// by the library). sig_op: DUSHMEM_SIGNAL_SET==9 / DUSHMEM_SIGNAL_ADD==10.
extern "C" __device__ int __attribute__((visibility("default")))
dushmem_putmem_signal_wrapper(unsigned long dest, unsigned long source, size_t bytes,
                              unsigned long sig_addr, unsigned long signal, int sig_op, int pe) {
    dushmem_putmem_signal((void *)dest, (const void *)source, bytes, (uint64_t *)sig_addr,
                          (uint64_t)signal, sig_op, pe);
    return 0;
}

extern "C" __device__ int __attribute__((visibility("default"))) dushmem_quiet_wrapper() {
    dushmem_quiet();
    return 0;
}

extern "C" __device__ int __attribute__((visibility("default"))) dushmem_fence_wrapper() {
    dushmem_fence();
    return 0;
}

// ---- atomics (symmetric heap addresses only) ------------------------------
extern "C" __device__ int __attribute__((visibility("default")))
dushmem_int_atomic_add_wrapper(unsigned long dest, int value, int pe) {
    dushmem_int_atomic_add((int *)dest, value, pe);
    return 0;
}

extern "C" __device__ int __attribute__((visibility("default")))
dushmem_int_atomic_set_wrapper(unsigned long dest, int value, int pe) {
    dushmem_int_atomic_set((int *)dest, value, pe);
    return 0;
}

extern "C" __device__ int __attribute__((visibility("default")))
dushmem_uint64_atomic_add_wrapper(unsigned long dest, unsigned long value, int pe) {
    dushmem_uint64_atomic_add((uint64_t *)dest, (uint64_t)value, pe);
    return 0;
}

extern "C" __device__ int __attribute__((visibility("default")))
dushmem_uint64_atomic_set_wrapper(unsigned long dest, unsigned long value, int pe) {
    dushmem_uint64_atomic_set((uint64_t *)dest, (uint64_t)value, pe);
    return 0;
}

extern "C" __device__ unsigned long __attribute__((visibility("default")))
dushmem_uint64_atomic_fetch_add_wrapper(unsigned long dest, unsigned long value, int pe) {
    return (unsigned long)dushmem_uint64_atomic_fetch_add((uint64_t *)dest, (uint64_t)value, pe);
}

extern "C" __device__ int __attribute__((visibility("default")))
dushmem_barrier_all_wrapper() {
    dushmem_barrier_all();
    return 0;
}

extern "C" __device__ unsigned long __attribute__((visibility("default")))
dushmem_ptr_wrapper(unsigned long ptr, int pe) {
    return (unsigned long)dushmem_ptr((const void *)ptr, pe);
}

// Keep dushmemi_device_lib_version_d alive through triton's O3 GlobalDCE
// (dushmemx_cumodule_init looks it up in the loaded module).
extern "C" {
extern __constant__ dushmemi_version_t dushmemi_device_lib_version_d;
}
__device__ __attribute__((used)) void *const __dcu_keep_lib_version =
    (void *)&dushmemi_device_lib_version_d;
