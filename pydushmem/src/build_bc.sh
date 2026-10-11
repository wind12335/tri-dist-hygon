#!/bin/bash
# Build the pydushmem triton bundle: pt2pt wrappers + dushmem init.bc +
# xfer.bc (transfer_device implementations, self-contained LLVM intrinsics).
# Stock DTK bitcode must be feature-stripped first (make_fixed_bc.sh,
# same dir): +sramecc in per-function target-features conflicts with
# triton's bare -mcpu=gfx936 at the .amdgcn_target check.
set -e
DIR=$(dirname "$(readlink -f "$0")")
CLANG=/opt/dtk/llvm/bin/clang
LLVM_LINK=/opt/dtk/llvm/bin/llvm-link
INC=(-I/opt/dtk-26.04/include -I/opt/dtk-26.04/include/dushmem)
INIT_BC=${INIT_BC:-$DIR/init_fixed.bc}
XFER_BC=${XFER_BC:-$DIR/xfer_fixed.bc}
OUT=${OUT:-$DIR/../lib/pydushmem.bc}

$CLANG -c -x hip --cuda-device-only -std=c++20 -emit-llvm -nogpulib \
  --offload-arch=gfx936 -mcode-object-version=4 \
  -D__HIPCC__ -DHIP_ENABLE_WARP_SYNC_BUILTINS \
  "${INC[@]}" \
  "$DIR/dushmem_wrapper.cu" -o "$DIR/dushmem_wrapper.bc"

$LLVM_LINK "$DIR/dushmem_wrapper.bc" "$INIT_BC" "$XFER_BC" -o "$OUT"
echo "built $OUT"
