#!/usr/bin/env python3
"""Independent serial reference for nbody.c (same init, softened gravity, semi-implicit Euler).
Usage: ref_nbody.py N steps dt eps   -> prints kinetic_energy,pos_l1,momentum_rel"""
import sys
import numpy as np
M = (1 << 64) - 1
def splitmix(x):
    x = (x + 0x9e3779b97f4a7c15) & M
    x = ((x ^ (x >> 30)) * 0xbf58476d1ce4e5b9) & M
    x = ((x ^ (x >> 27)) * 0x94d049bb133111eb) & M
    return x ^ (x >> 31)
def u(x): return (x >> 11) * (1.0 / 9007199254740992.0)
N, steps, dt, eps = int(sys.argv[1]), int(sys.argv[2]), float(sys.argv[3]), float(sys.argv[4])
seed = 0x4e424f44595f3236
P = np.zeros((N, 4))
for g in range(N):
    base = seed ^ ((g * 0x9e3779b97f4a7c15) & M)
    P[g] = [2*u(splitmix((base+0) & M))-1, 2*u(splitmix((base+1) & M))-1,
            2*u(splitmix((base+2) & M))-1, 0.5+u(splitmix((base+3) & M))]
pos, m, v = P[:, :3].copy(), P[:, 3].copy(), np.zeros((N, 3))
for _ in range(steps):
    d = pos[None, :, :] - pos[:, None, :]          # d[i,j] = x_j - x_i
    r2 = (d**2).sum(-1) + eps**2
    w = m[None, :] / r2**1.5
    np.fill_diagonal(w, 0.0)                       # explicit self exclusion
    a = (w[:, :, None] * d).sum(1)
    v += a * dt
    pos += v * dt
ke = 0.5 * (m * (v**2).sum(1)).sum()
l1 = np.abs(pos).sum()
mom = np.linalg.norm((m[:, None] * v).sum(0)) / (m * np.linalg.norm(v, axis=1)).sum()
print(f"{ke:.12e},{l1:.12e},{mom:.3e}")
