Checksums agree across all modes and rank counts.

## Strong scaling (T_p seconds, speed-up S_p = T_1/T_p, efficiency E_p = S_p/p)

### N = 4096

| p | allgather T / S / E |
|---|---|
| 1 | 0.239 / 1.00 / 1.00 |
| 2 | 0.118 / 2.03 / 1.01 |
| 4 | 0.071 / 3.37 / 0.84 |
| 8 | 0.049 / 4.85 / 0.61 |

## Weak scaling, constant work per rank (N = N1*sqrt(p)); E_p = T_1/T_p

| p | N | allgather T (s) / E |
|---|---|---|

## Weak scaling, fixed particles per rank (work per rank grows ~p); T_p only

| p | N | allgather T (s) |
|---|---|---|

## Communication-to-computation ratio, strong scaling, N = 4096

exposed-comm time / compute time (max over ranks). For 'ring' this counts only the part NOT hidden;
'ring_blocking' shows the intrinsic ratio of the same data movement.

| p | allgather |
|---|---|
| 1 | 0.0003 |
| 2 | 0.0022 |
| 4 | 0.0236 |
| 8 | 0.2675 |

## Per-rank communication volume and buffer memory, N = 4096

| p | allgather recv MB/step ; buf MB |
|---|---|
| 1 | 0.00 ; 0.13 |
| 2 | 0.07 ; 0.13 |
| 4 | 0.10 ; 0.13 |
| 8 | 0.11 ; 0.13 |

