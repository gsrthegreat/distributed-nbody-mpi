#!/usr/bin/env bash
# Correctness check: every mode and rank count must give the same physics.
# Checksums (kinetic energy, sum |x|+|y|+|z|) must agree with the p=1 result to
# a relative 1e-9 (summation order differs between decompositions), and the
# total momentum must stay ~0 (equal-and-opposite pair forces).
# Usage: ./verify.sh            (override: MPIEXEC="mpirun --oversubscribe" N=256)
set -euo pipefail
MPIEXEC=${MPIEXEC:-mpirun}
read -r -a MPIEXEC_CMD <<< "$MPIEXEC"
EXE=${EXE:-./nbody}
N=${N:-256}          # divisible by 1,2,4,8,16
STEPS=${STEPS:-5}
RANKS=${RANKS:-"1 2 4 8 16"}
TOL=${TOL:-1e-9}

run() { "${MPIEXEC_CMD[@]}" -np "$2" "$EXE" "$N" "$STEPS" 0.0005 0.01 "$1" | grep '^RESULT,' | tail -n 1; }
ref=$(run allgather 1)
ref_ke=$(cut -d, -f14 <<< "$ref"); ref_l1=$(cut -d, -f15 <<< "$ref")
echo "reference (allgather, p=1): KE=$ref_ke  pos_l1=$ref_l1"
fail=0
for mode in allgather ring_blocking ring force2d; do
  for p in $RANKS; do
    if [[ $mode == force2d ]] && ! awk -v p="$p" 'BEGIN{q=int(sqrt(p)+0.5); exit !(q*q==p)}'; then continue; fi
    line=$(run "$mode" "$p")
    ke=$(cut -d, -f14 <<< "$line"); l1=$(cut -d, -f15 <<< "$line"); mom=$(cut -d, -f16 <<< "$line")
    ok=$(awk -v a="$ke" -v b="$ref_ke" -v c="$l1" -v d="$ref_l1" -v m="$mom" -v t="$TOL" \
      'function rel(x,y){return (x>y?x-y:y-x)/(y>0?y:-y)} BEGIN{print (rel(a,b)<t && rel(c,d)<t && m<1e-9) ? "PASS":"FAIL"}')
    printf '%-14s p=%-2d KE=%s pos_l1=%s mom_rel=%s  %s\n' "$mode" "$p" "$ke" "$l1" "$mom" "$ok"
    [[ $ok == PASS ]] || fail=1
  done
done
(( fail == 0 )) && echo "ALL CHECKS PASSED" || { echo "SOME CHECKS FAILED"; exit 1; }
