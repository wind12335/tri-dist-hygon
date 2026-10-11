#!/bin/bash
# Regenerate *_fixed.bc from stock dushmem bitcode.
#
# Why: stock init.bc/xfer.bc carry per-function `target-cpu="gfx936"` +
# `target-features="...,+sramecc,..."` attributes. Triton's hcu pipeline runs
# llc with a bare -mcpu=gfx936, so those attributes make the asm printer emit
# `.amdgcn_target amdgcn-amd-amdhsa--gfx936:sramecc+`, which the assembler
# rejects against its plain gfx936 target id. Triton's own kernel functions
# carry no target attributes at all, so stripping them makes the bundle
# consistent with the rest of the module.
#
# Usage: make_fixed_bc.sh [srcdir]   (defaults to this dir; stock files must
# contain init.bc and xfer.bc)
set -e
DIR=$(dirname "$(readlink -f "$0")")
SRC=${1:-$DIR}
OPT=/opt/dtk/llvm/bin/opt

for f in init xfer; do
  [ -f "$SRC/$f.bc" ] || { echo "missing $SRC/$f.bc"; exit 1; }
  $OPT "$SRC/$f.bc" -S -o "$SRC/$f.ll"
  sed -i 's/ "target-cpu"="[^"]*"//g; s/ "target-features"="[^"]*"//g' "$SRC/$f.ll"
  $OPT "$SRC/$f.ll" -o "$SRC/${f}_fixed.bc"
  rm -f "$SRC/$f.ll"
  echo "built $SRC/${f}_fixed.bc"
done
