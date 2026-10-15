#!/bin/bash
# S20: 8-card natural-skew bench (see s20_bench_8card.py header for the
# pre-registered gates G1/G2/G3). One mpirun for all arms (single ictx),
# incremental per-arm per-rank JSON flush.
cd /root/private_data/tri-dist-hygon
ulimit -c 0
export HSA_FORCE_FINE_GRAIN_PCIE=1
# pin the transport knobs (defaults, but explicit): engine pulls -- the
# chunks=2 + >=4.5MB kernel-pull wedge shape must not sneak in
export I1_PULL=engine
LOG=bench/results/s20_full.log
mkdir -p bench/results
: > $LOG
timeout 2400 mpirun --allow-run-as-root -np 8 --bind-to none \
  -mca coll ^hcoll \
  -x HSA_FORCE_FINE_GRAIN_PCIE=1 -x I1_PULL=engine \
  -x TRITON_CACHE_DIR=/tmp/tc_s20 \
  python3 bench/s20_bench_8card.py 2>&1 | tee -a $LOG
echo "== exit $? (teardown segfault after DONE is harmless dushmem noise)"
