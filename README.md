# Problem 18 - Distributed N-Body with MPI

## Contents

- `nbody.c` - direct O(N^2) 3-D N-body with four decompositions: `allgather`, `ring_blocking`, `ring` (overlapped) and `force2d` (optional 2-D force decomposition).
- `bench.sh` - strong scaling, weak scaling (constant work per rank **and** fixed particles per rank) for 1, 2, 4, 8, 16 ranks; writes `results.csv` (best of `REPS` runs) and `results_all_runs.csv`.
- `verify.sh` - correctness check: all modes / rank counts must reproduce the same physics.
- `analyze.py` - turns `results.csv` into the Markdown tables for the report (speed-up, efficiency, comm/compute ratio, overlap gain, memory).
- `ref_nbody.py` - independent serial NumPy reference (needs `numpy`) for the checksums.
- `report_section.docx` - two-page report section; result tables are left blank for your measurements.
- `CHANGES.md` - what was found in the original submission and what changed.

## Build

```bash
mpicc -O3 -std=c11 -Wall -Wextra -pedantic nbody.c -lm -o nbody
# optional, faster kernel (do not use -ffast-math, it changes results):
# mpicc -O3 -march=native -fno-math-errno -std=c11 nbody.c -lm -o nbody
```

## Run one experiment

```bash
mpirun -np 4  ./nbody 4096 10 0.0005 0.01 allgather
mpirun -np 4  ./nbody 4096 10 0.0005 0.01 ring_blocking
mpirun -np 4  ./nbody 4096 10 0.0005 0.01 ring
mpirun -np 16 ./nbody 4096 10 0.0005 0.01 force2d      # p must be a perfect square
```

Arguments: `N` (global particles), `steps`, `dt`, `epsilon` (softening, > 0), mode.
`N` must be divisible by the rank count. Every rank generates its own initial particles from the global index (no initial scatter); a record is `(x, y, z, mass)` as four doubles.

## Verify correctness first

```bash
./verify.sh                                   # N=256, all modes, p = 1 2 4 8 16
MPIEXEC="mpirun --oversubscribe" ./verify.sh  # if you have fewer than 16 cores (Open MPI)
python3 ref_nbody.py 256 5 0.0005 0.01        # independent reference: KE, pos_l1, momentum
```

`verify.sh` requires kinetic energy and `sum(|x|+|y|+|z|)` to match the p=1 run to 1e-9 relative and the relative total momentum to stay below 1e-9 (pair forces are equal and opposite). Summation order differs between decompositions, so bitwise equality is not expected.

## Run the complete experiment matrix

```bash
./bench.sh
python3 analyze.py results_all_runs.csv > tables.md
```

Defaults: ranks `1 2 4 8 16`; strong-scaling sizes `4096 8192 16384`; weak work N1 = `4096`; `STEPS=5`; `REPS=3`. Override, e.g.:

```bash
STEPS=20 REPS=5 STRONG_SIZES="8192 16384 32768" WEAK_WORK_N1=8192 ./bench.sh
MODES="allgather ring" RANKS="1 2 4" ./bench.sh
```

Scenarios in the CSV:

- `strong` - fixed global N, p grows. `S_p = T_1/T_p`, `E_p = S_p/p`.
- `weak_work` - `N = N1*sqrt(p)` rounded to a multiple of p, so pairs per rank `N^2/p` stay constant. This is the meaningful weak scaling for an O(N^2) method; `E_p = T_1/T_p`.
- `weak_n` - fixed `n = N/p` (work per rank grows ~p, so time must rise); kept only for comparison.

`force2d` runs only for square p (1, 4, 16); other rank counts are skipped. Runs with more ranks than hardware threads get `oversubscribed=1`; do not use them for speed-up/efficiency claims. For fair timing use the same machine, compiler flags, process placement and load for all modes (and pin ranks, e.g. `mpirun --bind-to core`).

## Output columns (`RESULT` line / `results.csv`)

`mode,scenario,N,ranks,steps,total_s,compute_s,comm_exposed_s,comm_compute_ratio,recv_bytes_per_rank,interactions_per_rank,recv_bytes_per_flop,buffer_bytes_per_rank,kinetic_energy,pos_l1,momentum_rel,oversubscribed`

- `total_s` - max over ranks of wall time of the simulation loop.
- `compute_s` - max accumulated time in the force kernels.
- `comm_exposed_s` - max accumulated time in MPI calls on the critical path. For `ring` this is posting + `MPI_Testall` + the `MPI_Waitall` remainder, i.e. it excludes communication hidden behind the kernel. For `ring_blocking`, `allgather` and `force2d` it is the whole communication time.
- `comm_compute_ratio` - `comm_exposed_s / compute_s`. Compare `ring_blocking` (intrinsic) with `ring` (what is left after overlap).
- `recv_bytes_per_rank` - logical incoming payload over all steps (worst-case rank). For `force2d` the reduce-scatter part is a lower bound.
- `interactions_per_rank` - `steps * n * N` pair evaluations (self pair included; it evaluates to exactly 0).
- `recv_bytes_per_flop` - incoming payload / (interactions * 20 FLOP). A communication-intensity estimate, not a hardware counter.
- `buffer_bytes_per_rank` - communication buffers: O(N) allgather, O(N/p) ring, O(N/sqrt(p)) force2d.
- `kinetic_energy, pos_l1, momentum_rel` - validation checksums (outside the timed region).

## Notes and limitations

- Direct all-pairs solver: it demonstrates MPI data movement, not Barnes-Hut/FMM. Semi-implicit Euler with softened Newtonian gravity; keep `dt` small enough for a stable illustrative run.
- Overlap: `ring` posts `MPI_Irecv/MPI_Isend`, then runs the kernel in chunks of about 16k pairs, calling `MPI_Testall` between chunks so that libraries without an asynchronous progress thread advance the transfer. The send buffer is only read until `MPI_Waitall` returns. Whether overlap pays off still depends on the library, the interconnect and the block size; measure `ring_blocking` vs `ring`.
- With intra-node (shared-memory) runs, messages are small and fast, so little communication is left to hide; expect the largest overlap gains on multi-node runs.
- `force2d` needs MPI-2.2 or later (`MPI_Reduce_scatter_block`).
