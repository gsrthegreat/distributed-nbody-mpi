# Review notes: what was found and what changed

## Verdict on the original
The core algorithms were correct: allgather and ring reproduce the serial physics (ring owner mapping, buffer swaps and the p-1 shifts are right; no deadlock; send buffer not modified while in flight). The problems were in what surrounded the kernel.

## Defects fixed
1. **`bench.sh` wrote a misaligned CSV (real bug).** `RESULT,<mode>,...` was cut with `-f2-` and `mode` was then prepended again, giving 13 fields under a 12-column header (`N` held the mode name, `ranks` held N, `total_s` held the step count). Fixed (`-f3-`); the new header and all rows have 17 fields.
2. **"Weak scaling" was not weak scaling.** N = 256*p makes work per rank grow linearly with p for an O(N^2) method. The report admitted this, but the benchmark never ran the meaningful test. Added `weak_work` (N = N1*sqrt(p), constant pairs per rank); the old test stays as `weak_n`.
3. **Overlap could not be evaluated, and probably did not happen.** There was no non-overlapped ring to compare against, and no MPI progress calls during the kernel (large transfers often do not advance without them). Added `ring_blocking` (MPI_Sendrecv) and chunked `MPI_Testall` progress in `ring`.
4. **No correctness validation.** Added physics checksums (kinetic energy, position L1 norm, relative momentum), `verify.sh`, and an independent NumPy reference `ref_nbody.py`.
5. **Optional 2-D force decomposition was missing**, so the "comparison of the decompositions" compared only two 1-D variants. Added `force2d` (row Allgather, column Bcast, tile, row Reduce_scatter_block) for square p.
6. **Benchmark hygiene:** one run per case and tiny sizes (N <= 4096, ~5 ms of work at high p). Now best-of-`REPS`, larger defaults, oversubscription flagged in the CSV, all raw runs kept.
7. **Kernel:** removed the per-pair `if (self) continue` (with softening the self term is exactly 0). This keeps the inner loop branch-free and simplifies the 2-D mode; pairs per rank are now `n*N`, and the report formulas were updated (`1.6(p-1)/N` bytes per FLOP).

## Report
Updated to match the code (four modes, chunked-progress explanation, 2-D analysis with the p >= 9 volume crossover, constant-work weak scaling, validation). Still exactly two pages. Result tables are intentionally blank: no timings were measured, and none should be invented.

## How this was tested (and what was not)
No MPI library or multi-core machine was available in the review environment. Logic was tested with a small thread-based MPI stand-in (all collectives and point-to-point calls used by `nbody.c`): every mode at p = 1, 2, 4, 8, 16 (force2d at 1, 4, 16) matched the independent NumPy reference to 12 digits at N = 256 and N = 2048, a deliberately broken ring was caught by `verify.sh`, and `bench.sh` -> `analyze.py` ran end to end. The code compiles warning-free with `-Wall -Wextra -pedantic`. **Not tested:** real MPI libraries, performance, actual overlap behaviour. Run `./verify.sh` on your real MPI first, then `./bench.sh`.
