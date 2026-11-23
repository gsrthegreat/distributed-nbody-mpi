#!/usr/bin/env python3
"""Turn results.csv (from bench.sh) into the tables needed for the report.

Usage: python3 analyze.py [results.csv]      (prints Markdown; stdlib only)

Tables: strong-scaling time / speed-up / efficiency, weak-scaling efficiency
(constant work per rank and fixed particles per rank), communication-to-
computation ratios, benefit of overlap (ring_blocking vs ring), memory.
Rows with oversubscribed=1 are marked with '*' - do not quote them as scaling.
"""
import csv
import sys
from collections import defaultdict

path = sys.argv[1] if len(sys.argv) > 1 else "results.csv"
rows = list(csv.DictReader(open(path)))
for r in rows:
    for k in ("N", "ranks", "steps", "oversubscribed"):
        r[k] = int(r[k])
    for k in ("total_s", "compute_s", "comm_exposed_s", "comm_compute_ratio",
              "recv_bytes_per_rank", "buffer_bytes_per_rank",
              "kinetic_energy", "pos_l1"):
        r[k] = float(r[k])

MODES = ["allgather", "ring_blocking", "ring", "force2d"]
modes = [m for m in MODES if any(r["mode"] == m for r in rows)]
idx = {(r["mode"], r["scenario"], r["N"], r["ranks"]): r for r in rows}
ranks = sorted({r["ranks"] for r in rows})

def mark(r): return "*" if r["oversubscribed"] else ""
def cell(v, fmt): return "-" if v is None else fmt.format(v)

# --- sanity: identical physics across modes / p for the same (N, steps) ---
groups = defaultdict(list)
for r in rows:
    groups[(r["N"], r["steps"])].append(r)
bad = 0
for (n, s), g in groups.items():
    ke0, l10 = g[0]["kinetic_energy"], g[0]["pos_l1"]
    for r in g:
        if abs(r["kinetic_energy"] - ke0) > 1e-8 * abs(ke0) or abs(r["pos_l1"] - l10) > 1e-8 * abs(l10):
            bad += 1
            print(f"WARNING: checksum mismatch N={n}: {r['mode']} p={r['ranks']}")
print("Checksums agree across all modes and rank counts.\n" if not bad else "")

print("## Strong scaling (T_p seconds, speed-up S_p = T_1/T_p, efficiency E_p = S_p/p)\n")
for n in sorted({r["N"] for r in rows if r["scenario"] == "strong"}):
    print(f"### N = {n}\n")
    print("| p | " + " | ".join(f"{m} T / S / E" for m in modes) + " |")
    print("|---|" + "---|" * len(modes))
    for p in ranks:
        cells = []
        for m in modes:
            r, r1 = idx.get((m, "strong", n, p)), idx.get((m, "strong", n, 1))
            if r is None or r1 is None:
                cells.append("-"); continue
            s = r1["total_s"] / r["total_s"]
            cells.append(f"{r['total_s']:.3f} / {s:.2f} / {s / p:.2f}{mark(r)}")
        print(f"| {p} | " + " | ".join(cells) + " |")
    print()

for scen, title in (("weak_work", "Weak scaling, constant work per rank (N = N1*sqrt(p)); E_p = T_1/T_p"),
                    ("weak_n", "Weak scaling, fixed particles per rank (work per rank grows ~p); T_p only")):
    print(f"## {title}\n")
    print("| p | N | " + " | ".join(f"{m} T (s)" + (" / E" if scen == "weak_work" else "") for m in modes) + " |")
    print("|---|---|" + "---|" * len(modes))
    for p in ranks:
        ns = {r["N"] for r in rows if r["scenario"] == scen and r["ranks"] == p}
        if not ns: continue
        n = ns.pop()
        cells = []
        for m in modes:
            r, r1 = idx.get((m, scen, n, p)), None
            r1 = next((x for x in rows if x["mode"] == m and x["scenario"] == scen and x["ranks"] == 1), None)
            if r is None: cells.append("-"); continue
            e = f" / {r1['total_s'] / r['total_s']:.2f}" if (scen == "weak_work" and r1) else ""
            cells.append(f"{r['total_s']:.3f}{e}{mark(r)}")
        print(f"| {p} | {n} | " + " | ".join(cells) + " |")
    print()

big = max((r["N"] for r in rows if r["scenario"] == "strong"), default=None)
if big:
    print(f"## Communication-to-computation ratio, strong scaling, N = {big}\n")
    print("exposed-comm time / compute time (max over ranks). For 'ring' this counts only the part NOT hidden;\n"
          "'ring_blocking' shows the intrinsic ratio of the same data movement.\n")
    print("| p | " + " | ".join(modes) + " |")
    print("|---|" + "---|" * len(modes))
    for p in ranks:
        cells = [cell(idx[(m, "strong", big, p)]["comm_compute_ratio"], "{:.4f}") + mark(idx[(m, "strong", big, p)])
                 if (m, "strong", big, p) in idx else "-" for m in modes]
        print(f"| {p} | " + " | ".join(cells) + " |")
    print()
    if "ring" in modes and "ring_blocking" in modes:
        print(f"## Benefit of overlap, N = {big}: T(ring_blocking) / T(ring)\n")
        print("| p | blocking T (s) | overlapped T (s) | gain | exposed comm blocking -> overlapped (s) |")
        print("|---|---|---|---|---|")
        for p in ranks:
            a, b = idx.get(("ring_blocking", "strong", big, p)), idx.get(("ring", "strong", big, p))
            if a and b:
                print(f"| {p} | {a['total_s']:.3f} | {b['total_s']:.3f} | {a['total_s'] / b['total_s']:.2f}x{mark(a)} "
                      f"| {a['comm_exposed_s']:.4f} -> {b['comm_exposed_s']:.4f} |")
        print()
    print(f"## Per-rank communication volume and buffer memory, N = {big}\n")
    print("| p | " + " | ".join(f"{m} recv MB/step ; buf MB" for m in modes) + " |")
    print("|---|" + "---|" * len(modes))
    for p in ranks:
        cells = []
        for m in modes:
            r = idx.get((m, "strong", big, p))
            cells.append("-" if not r else f"{r['recv_bytes_per_rank'] / r['steps'] / 1e6:.2f} ; {r['buffer_bytes_per_rank'] / 1e6:.2f}")
        print(f"| {p} | " + " | ".join(cells) + " |")
    print()
if any(r["oversubscribed"] for r in rows):
    print("`*` = more ranks than hardware threads (oversubscribed): timings not valid for scaling claims.")
