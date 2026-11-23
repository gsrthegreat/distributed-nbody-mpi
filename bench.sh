#!/usr/bin/env bash
# Benchmark all decompositions: strong scaling, weak scaling (two definitions).
# Override defaults, e.g.:  STEPS=20 REPS=5 STRONG_SIZES="8192 16384" ./bench.sh
#
# Scenarios written to $OUT (best-of-REPS by total_s):
#   strong     fixed global N, p grows                      -> S_p = T_1/T_p
#   weak_work  N = N1*sqrt(p) (rounded to a multiple of p): pairs per rank N^2/p
#              stays constant, i.e. the *correct* weak scaling for O(N^2) work
#                                                            -> E_p = T_1/T_p
#   weak_n     n = N/p fixed (work per rank grows ~ p; kept for comparison)
set -euo pipefail

MPIEXEC=${MPIEXEC:-mpirun}
read -r -a MPIEXEC_CMD <<< "$MPIEXEC"
EXE=${EXE:-./nbody}
STEPS=${STEPS:-5}
REPS=${REPS:-3}
DT=${DT:-0.0005}
EPS=${EPS:-0.01}
MODES=${MODES:-"allgather ring_blocking ring force2d"}
STRONG_SIZES=${STRONG_SIZES:-"4096 8192 16384"}
WEAK_WORK_N1=${WEAK_WORK_N1:-4096}      # N used at p = 1 for constant work per rank
WEAK_LOCAL_N=${WEAK_LOCAL_N:-512}       # particles per rank for the fixed-n variant
RANKS=${RANKS:-"1 2 4 8 16"}
OUT=${OUT:-results.csv}
RAW=${RAW:-results_all_runs.csv}

if [[ ! -x "$EXE" ]]; then
  echo "Executable not found or not executable: $EXE" >&2
  echo "Build it first: mpicc -O3 -std=c11 -Wall -Wextra -pedantic nbody.c -lm -o nbody" >&2
  exit 1
fi

CORES=$(nproc 2>/dev/null || sysctl -n hw.ncpu 2>/dev/null || echo 1)
echo "Detected $CORES hardware threads. Runs with more ranks than that are flagged" >&2
echo "oversubscribed=1 in the CSV; do NOT use them for speed-up / efficiency claims." >&2

HEADER='mode,scenario,N,ranks,steps,total_s,compute_s,comm_exposed_s,comm_compute_ratio,recv_bytes_per_rank,interactions_per_rank,recv_bytes_per_flop,buffer_bytes_per_rank,kinetic_energy,pos_l1,momentum_rel,oversubscribed'
printf '%s\n' "$HEADER" > "$OUT"
printf '%s,rep\n' "$HEADER" > "$RAW"

is_square() { awk -v p="$1" 'BEGIN{q=int(sqrt(p)+0.5); exit !(q*q==p)}'; }

run_case() {
  local mode=$1 scenario=$2 n_particles=$3 ranks=$4
  local oversub=0 best="" best_t="" rep output result t
  (( ranks > CORES )) && oversub=1
  for (( rep = 1; rep <= REPS; rep++ )); do
    echo "Running: mode=$mode scenario=$scenario N=$n_particles ranks=$ranks steps=$STEPS rep=$rep/$REPS" >&2
    output=$("${MPIEXEC_CMD[@]}" -np "$ranks" "$EXE" "$n_particles" "$STEPS" "$DT" "$EPS" "$mode")
    # RESULT,<mode>,<N>,... : drop the tag AND the mode (we prepend mode,scenario ourselves)
    result=$(printf '%s\n' "$output" | grep '^RESULT,' | tail -n 1 | cut -d, -f3-)
    if [[ -z "$result" ]]; then
      echo "Could not find RESULT record. Full output:" >&2
      printf '%s\n' "$output" >&2
      exit 1
    fi
    printf '%s,%s,%s,%s,%s\n' "$mode" "$scenario" "$result" "$oversub" "$rep" >> "$RAW"
    t=$(printf '%s' "$result" | cut -d, -f4)       # total_s (fields: N,ranks,steps,total_s,...)
    if [[ -z "$best_t" ]] || awk -v a="$t" -v b="$best_t" 'BEGIN{exit !(a<b)}'; then
      best_t=$t; best=$result
    fi
  done
  printf '%s,%s,%s,%s\n' "$mode" "$scenario" "$best" "$oversub" >> "$OUT"
}

read -r -a sizes <<< "$STRONG_SIZES"
read -r -a rank_list <<< "$RANKS"

for mode in $MODES; do

  # Strong scaling: fixed global N while p increases.
  for n_particles in "${sizes[@]}"; do
    for ranks in "${rank_list[@]}"; do
      if [[ $mode == force2d ]] && ! is_square "$ranks"; then
        echo "Skipping force2d at p=$ranks (not a perfect square)" >&2; continue
      fi
      if (( n_particles % ranks != 0 )); then
        echo "Skipping N=$n_particles because it is not divisible by p=$ranks" >&2; continue
      fi
      run_case "$mode" strong "$n_particles" "$ranks"
    done
  done

  for ranks in "${rank_list[@]}"; do
    if [[ $mode == force2d ]] && ! is_square "$ranks"; then continue; fi
    # Weak scaling, constant work per rank: N = round(N1*sqrt(p)/p)*p.
    n_particles=$(awk -v n1="$WEAK_WORK_N1" -v p="$ranks" 'BEGIN{printf "%d", int(n1*sqrt(p)/p+0.5)*p}')
    run_case "$mode" weak_work "$n_particles" "$ranks"
    # Weak scaling, fixed particles per rank (work per rank grows linearly with p).
    run_case "$mode" weak_n "$((ranks * WEAK_LOCAL_N))" "$ranks"
  done
done

echo "Wrote best-of-$REPS results to $OUT (all runs in $RAW)"
echo "Next: python3 analyze.py $OUT   (tables for the report)"
