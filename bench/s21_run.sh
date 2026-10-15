#!/bin/bash
# S21: upstream-style TP-MLP bench (ag_rs default; add gemm_ar via env).
# Default cell set at R=8: LLaMA-3.1-70B x M{512..8192} (Qwen2-72B and
# LLaMA-7B auto-SKIP at R=8 -- ragged 2I/R; they run at R=2).
cd "$(dirname "$0")/.."
mkdir -p bench/results
S21_ARMS_LOG="bench/results/s21_full.log"
export HSA_FORCE_FINE_GRAIN_PCIE=1
timeout 2400 mpirun --allow-run-as-root -np 8 --bind-to none \
  -mca coll ^hcoll \
  -x HSA_FORCE_FINE_GRAIN_PCIE=1 \
  -x TRITON_CACHE_DIR=/tmp/tc_s21 \
  python3 bench/s21_bench_tp_mlp_dcu.py 2>&1 | tee -a "$S21_ARMS_LOG"
# wider sweep (card time permitting):
#   S21_MODES=ag_rs,gemm_ar S21_MODELS=LLaMA-3.1-70B,LLaMA-3.1-405B \
#   S21_MS=512,1024,2048,4096,8192 bash bench/s21_run.sh
