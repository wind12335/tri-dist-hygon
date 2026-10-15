#!/bin/bash
# S18 full runs at M8192 N29568 K8192 — one mpirun per family, timeout each
cd /root/private_data/tri-dist-hygon
ulimit -c 0
export HSA_FORCE_FINE_GRAIN_PCIE=1
LOG=bench/results/s18_full.log
: > $LOG
for fam in ar rs ag a2a; do
  echo "########## $fam $(date +%T) ##########" | tee -a $LOG
  timeout 1500 mpirun --allow-run-as-root -np 2 --oversubscribe \
    --bind-to none -mca coll ^hcoll -x HSA_FORCE_FINE_GRAIN_PCIE=1 \
    env S18_FAM=$fam python3 bench/s18_bench_unified.py >> $LOG 2>&1
  echo "## $fam rc=$? $(date +%T)" | tee -a $LOG
done
S18_FAM=merge python3 bench/s18_bench_unified.py > /dev/null 2>&1 || true
echo MERGED; grep -E "==>|====|SKIP" $LOG | grep -v warmup
