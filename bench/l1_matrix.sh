#!/bin/bash
# L1 matrix driver: 3 arms x 2 tiers x deltas {0,200,400,800,1600} = 30
# cells, one mpirun each (l1_pred_bench.py = one cell per run). Sequential,
# restartable: existing result JSONs are skipped (delete to re-run a
# cell). Aggregate afterwards: python3 bench/l1_pred_aggregate.py 40.
cd /root/private_data/tri-dist-hygon
ulimit -c 0
export HSA_FORCE_FINE_GRAIN_PCIE=1
# rig config locked by the S12-S15 bisect (see docs/I1_PREDICTOR_L1_20261013.md):
# all-kernel gated path + consumer acquire-confirm + capped consumer grid
export I1_STAGE=kernel I1_PULL=kernel I1_ARRIVE=kernel I1_READY=kernel
ROUNDS=${L1_ROUNDS:-40}
for tier in sticky decorr; do
  for delta in 0 200 400 800 1600; do
    for arm in nopred pred oracle; do
      tag="l1_${arm}_${tier}_${delta}us_r${ROUNDS}_rank0.json"
      if [ -f "bench/results/$tag" ] && [ -f "bench/results/l1_${arm}_${tier}_${delta}us_r${ROUNDS}_rank1.json" ]; then
        echo "=== SKIP $arm/$tier/${delta}us (exists) $(date +%T)"
        continue
      fi
      echo "=== L1 arm=$arm tier=$tier delta=${delta}us rounds=$ROUNDS $(date +%T)"
      L1_ARM=$arm L1_TIER=$tier L1_DELTA_US=$delta L1_ROUNDS=$ROUNDS \
      timeout 900 mpirun --allow-run-as-root -np 2 --oversubscribe --bind-to none \
        -mca coll ^hcoll -x HSA_FORCE_FINE_GRAIN_PCIE=1 \
        -x I1_STAGE=kernel -x I1_PULL=kernel -x I1_ARRIVE=kernel -x I1_READY=kernel \
        python3 bench/l1_pred_bench.py > /tmp/l1_${arm}_${tier}_${delta}us.log 2>&1
      echo "--- exit=$? $(date +%T): $(grep -h 'L1 DONE' /tmp/l1_${arm}_${tier}_${delta}us.log | tr '\n' ' ')"
    done
  done
done
echo "MATRIX DONE $(date +%T)"
python3 bench/l1_pred_aggregate.py $ROUNDS 2>&1 | tail -40
