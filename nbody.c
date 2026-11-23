/*
 * Problem 18: Distributed N-Body with MPI
 *
 * Direct O(N^2) 3-D gravitational N-body simulation with softened gravity.
 * Every rank owns n = N / p particles (global ids rank*n .. rank*n+n-1).
 *
 * Modes (last command-line argument):
 *   allgather     : 1-D. MPI_Allgather of all particle records every step.
 *   ring_blocking : 1-D. p-1 ring shifts with blocking MPI_Sendrecv; the force
 *                   kernel and the transfer do NOT overlap (reference for the
 *                   overlap experiment and for the intrinsic comm cost).
 *   ring          : 1-D. p-1 ring shifts with MPI_Isend/MPI_Irecv; the force
 *                   kernel on the current block runs while the next block is
 *                   in flight. The kernel is cut into chunks and MPI_Testall
 *                   is called between chunks so that MPI libraries without an
 *                   asynchronous progress thread really advance the transfer.
 *   force2d       : 2-D force decomposition on a q x q grid (p = q*q). Rank
 *                   (r,c) computes the forces of target block r due to source
 *                   block c (N/q particles each).
 *
 * Particle record: x, y, z, mass (four doubles; 32 bytes).
 * The program requires N to be divisible by the number of ranks.
 *
 * Self interaction: with softening eps > 0 the self term has dx=dy=dz=0, so it
 * contributes exactly 0 to the acceleration. The kernel therefore has no
 * "skip self" branch (keeps the inner loop branch-free / vectorisable) and
 * evaluates n*N pairs per rank and step.
 *
 * Build:
 *   mpicc -O3 -std=c11 -Wall -Wextra -pedantic nbody.c -lm -o nbody
 *   (optional, faster kernel: add -march=native -fno-math-errno)
 *
 * Run:
 *   mpirun -np 4 ./nbody 4096 10 0.0005 0.01 ring
 */

#include <mpi.h>
#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <errno.h>
#include <limits.h>
#include <stddef.h>

#define DIM 3
#define STRIDE 4
#define G_CONST 1.0
#define MPI_TAG_RING 73
#define APPROX_FLOPS_PER_PAIR 20.0
/* Target pair evaluations between two MPI_Testall progress calls (ring mode). */
#define PROGRESS_PAIRS 16384

typedef enum {
    MODE_ALLGATHER,
    MODE_RING_BLOCKING,
    MODE_RING,
    MODE_FORCE2D
} Mode;

static uint64_t splitmix64(uint64_t x)
{
    x += UINT64_C(0x9e3779b97f4a7c15);
    x = (x ^ (x >> 30)) * UINT64_C(0xbf58476d1ce4e5b9);
    x = (x ^ (x >> 27)) * UINT64_C(0x94d049bb133111eb);
    return x ^ (x >> 31);
}

static double unit_interval(uint64_t x)
{
    /* Use the top 53 bits to create a reproducible value in [0, 1). */
    return (double)(x >> 11) * (1.0 / 9007199254740992.0);
}

static void initialize_particles(double *local, int local_n, int rank)
{
    const uint64_t seed = UINT64_C(0x4e424f44595f3236);

    for (int i = 0; i < local_n; ++i) {
        const uint64_t gid = (uint64_t)rank * (uint64_t)local_n + (uint64_t)i;
        const uint64_t base = seed ^ (gid * UINT64_C(0x9e3779b97f4a7c15));

        local[(size_t)STRIDE * (size_t)i + 0] =
            2.0 * unit_interval(splitmix64(base + 0)) - 1.0;
        local[(size_t)STRIDE * (size_t)i + 1] =
            2.0 * unit_interval(splitmix64(base + 1)) - 1.0;
        local[(size_t)STRIDE * (size_t)i + 2] =
            2.0 * unit_interval(splitmix64(base + 2)) - 1.0;
        local[(size_t)STRIDE * (size_t)i + 3] =
            0.5 + unit_interval(splitmix64(base + 3));
    }
}

static int parse_positive_int(const char *s, const char *name, int *out)
{
    char *end = NULL;
    long value;

    errno = 0;
    value = strtol(s, &end, 10);
    if (errno != 0 || end == s || *end != '\0' || value <= 0 || value > INT_MAX) {
        fprintf(stderr, "Invalid %s: %s\n", name, s);
        return 0;
    }
    *out = (int)value;
    return 1;
}

static int parse_positive_double(const char *s, const char *name, double *out)
{
    char *end = NULL;
    double value;

    errno = 0;
    value = strtod(s, &end);
    if (errno != 0 || end == s || *end != '\0' || !isfinite(value) || value <= 0.0) {
        fprintf(stderr, "Invalid %s: %s\n", name, s);
        return 0;
    }
    *out = value;
    return 1;
}

static void *checked_malloc(size_t bytes, int rank)
{
    void *ptr = malloc(bytes == 0 ? 1 : bytes);
    if (ptr == NULL) {
        fprintf(stderr, "Rank %d: allocation failed for %zu bytes\n", rank, bytes);
        MPI_Abort(MPI_COMM_WORLD, EXIT_FAILURE);
    }
    return ptr;
}

static void *checked_calloc(size_t count, size_t size, int rank)
{
    void *ptr = calloc(count == 0 ? 1 : count, size);
    if (ptr == NULL) {
        fprintf(stderr, "Rank %d: allocation failed for %zu bytes\n", rank, count * size);
        MPI_Abort(MPI_COMM_WORLD, EXIT_FAILURE);
    }
    return ptr;
}

/*
 * Add to acc[i] (i in [i0, i1)) the acceleration of target i due to all
 * source_n particles in `sources`. The self pair gives exactly zero (see the
 * header comment), so no index bookkeeping is needed.
 */
static void accumulate_forces(const double *restrict targets,
                              int i0,
                              int i1,
                              const double *restrict sources,
                              int source_n,
                              double eps2,
                              double *restrict acc)
{
    for (int i = i0; i < i1; ++i) {
        const double tx = targets[(size_t)STRIDE * (size_t)i + 0];
        const double ty = targets[(size_t)STRIDE * (size_t)i + 1];
        const double tz = targets[(size_t)STRIDE * (size_t)i + 2];
        double ax = 0.0, ay = 0.0, az = 0.0;

        for (int j = 0; j < source_n; ++j) {
            const double dx = sources[(size_t)STRIDE * (size_t)j + 0] - tx;
            const double dy = sources[(size_t)STRIDE * (size_t)j + 1] - ty;
            const double dz = sources[(size_t)STRIDE * (size_t)j + 2] - tz;
            const double r2 = dx * dx + dy * dy + dz * dz + eps2;
            const double inv_r = 1.0 / sqrt(r2);
            const double inv_r3 = inv_r * inv_r * inv_r;
            const double scale = G_CONST * sources[(size_t)STRIDE * (size_t)j + 3] * inv_r3;

            ax += scale * dx;
            ay += scale * dy;
            az += scale * dz;
        }
        acc[(size_t)DIM * (size_t)i + 0] += ax;
        acc[(size_t)DIM * (size_t)i + 1] += ay;
        acc[(size_t)DIM * (size_t)i + 2] += az;
    }
}

/* Whole-block force kernel with timing (no communication in flight). */
static void timed_forces(const double *targets, int target_n,
                         const double *sources, int source_n,
                         double eps2, double *acc, double *compute_seconds)
{
    const double t0 = MPI_Wtime();
    accumulate_forces(targets, 0, target_n, sources, source_n, eps2, acc);
    *compute_seconds += MPI_Wtime() - t0;
}

/*
 * Force kernel that overlaps with outstanding requests. The target loop is
 * cut into chunks of about PROGRESS_PAIRS pair evaluations and MPI_Testall is
 * called after each chunk until the requests complete. Time inside
 * MPI_Testall is charged to communication (it is MPI work on the critical
 * path); the rest is compute.
 */
static void overlapped_forces(const double *targets, int target_n,
                              const double *sources, int source_n,
                              double eps2, double *acc,
                              MPI_Request *requests, int nreq,
                              double *compute_seconds, double *comm_seconds)
{
    int rows = PROGRESS_PAIRS / (source_n > 0 ? source_n : 1);
    int done = 0;

    if (rows < 1) {
        rows = 1;
    }
    for (int i0 = 0; i0 < target_n; i0 += rows) {
        const int i1 = (i0 + rows < target_n) ? i0 + rows : target_n;
        double t0 = MPI_Wtime();

        accumulate_forces(targets, i0, i1, sources, source_n, eps2, acc);
        *compute_seconds += MPI_Wtime() - t0;

        if (!done) {
            t0 = MPI_Wtime();
            MPI_Testall(nreq, requests, &done, MPI_STATUSES_IGNORE);
            *comm_seconds += MPI_Wtime() - t0;
        }
    }
}

static void integrate(double *local,
                      double *velocity,
                      const double *acc,
                      int local_n,
                      double dt)
{
    for (int i = 0; i < local_n; ++i) {
        for (int d = 0; d < DIM; ++d) {
            const size_t vindex = (size_t)DIM * (size_t)i + (size_t)d;
            const size_t pindex = (size_t)STRIDE * (size_t)i + (size_t)d;
            velocity[vindex] += acc[vindex] * dt;
            local[pindex] += velocity[vindex] * dt;
        }
    }
}

/* ---------------------------------------------------------------------- */
/* 1-D baseline: allgather                                                 */
/* ---------------------------------------------------------------------- */
static void run_allgather(double *local, double *all_particles,
                          double *velocity, double *acc,
                          int n, int p, int steps, double dt, double eps2,
                          double *compute_seconds, double *comm_seconds)
{
    const int record_count = n * STRIDE;

    for (int step = 0; step < steps; ++step) {
        double t0;
        memset(acc, 0, (size_t)DIM * (size_t)n * sizeof(double));

        t0 = MPI_Wtime();
        MPI_Allgather(local, record_count, MPI_DOUBLE,
                      all_particles, record_count, MPI_DOUBLE, MPI_COMM_WORLD);
        *comm_seconds += MPI_Wtime() - t0;

        timed_forces(local, n, all_particles, p * n, eps2, acc, compute_seconds);
        integrate(local, velocity, acc, n, dt);
    }
}

/* ---------------------------------------------------------------------- */
/* 1-D ring, blocking (no overlap)                                         */
/* ---------------------------------------------------------------------- */
static void run_ring_blocking(double *local, double *block_a, double *block_b,
                              double *velocity, double *acc,
                              int n, int p, int rank, int steps,
                              double dt, double eps2,
                              double *compute_seconds, double *comm_seconds)
{
    const int record_count = n * STRIDE;
    const int next_rank = (rank + 1) % p;
    const int prev_rank = (rank - 1 + p) % p;

    for (int step = 0; step < steps; ++step) {
        double *current = block_a;
        double *next = block_b;

        memset(acc, 0, (size_t)DIM * (size_t)n * sizeof(double));
        memcpy(current, local, (size_t)record_count * sizeof(double));

        for (int stage = 0; stage < p - 1; ++stage) {
            double t0;
            double *tmp;

            timed_forces(local, n, current, n, eps2, acc, compute_seconds);

            t0 = MPI_Wtime();
            MPI_Sendrecv(current, record_count, MPI_DOUBLE, next_rank, MPI_TAG_RING,
                         next, record_count, MPI_DOUBLE, prev_rank, MPI_TAG_RING,
                         MPI_COMM_WORLD, MPI_STATUS_IGNORE);
            *comm_seconds += MPI_Wtime() - t0;

            tmp = current;
            current = next;
            next = tmp;
        }
        timed_forces(local, n, current, n, eps2, acc, compute_seconds);
        integrate(local, velocity, acc, n, dt);
    }
}

/* ---------------------------------------------------------------------- */
/* 1-D ring, non-blocking with computation/communication overlap           */
/* ---------------------------------------------------------------------- */
static void run_ring(double *local, double *block_a, double *block_b,
                     double *velocity, double *acc,
                     int n, int p, int rank, int steps,
                     double dt, double eps2,
                     double *compute_seconds, double *comm_seconds)
{
    const int record_count = n * STRIDE;
    const int next_rank = (rank + 1) % p;
    const int prev_rank = (rank - 1 + p) % p;

    for (int step = 0; step < steps; ++step) {
        double *current = block_a;
        double *next = block_b;

        memset(acc, 0, (size_t)DIM * (size_t)n * sizeof(double));
        memcpy(current, local, (size_t)record_count * sizeof(double));

        /* The local block is stage 0. Each of p-1 shifts supplies one new block. */
        for (int stage = 0; stage < p - 1; ++stage) {
            MPI_Request requests[2];
            double t0;
            double *tmp;

            /* Start moving the current block while the force kernel uses it. */
            t0 = MPI_Wtime();
            MPI_Irecv(next, record_count, MPI_DOUBLE, prev_rank,
                      MPI_TAG_RING, MPI_COMM_WORLD, &requests[0]);
            MPI_Isend(current, record_count, MPI_DOUBLE, next_rank,
                      MPI_TAG_RING, MPI_COMM_WORLD, &requests[1]);
            *comm_seconds += MPI_Wtime() - t0;

            overlapped_forces(local, n, current, n, eps2, acc,
                              requests, 2, compute_seconds, comm_seconds);

            /* Only the wait remaining after computation is exposed communication. */
            t0 = MPI_Wtime();
            MPI_Waitall(2, requests, MPI_STATUSES_IGNORE);
            *comm_seconds += MPI_Wtime() - t0;

            tmp = current;
            current = next;
            next = tmp;
        }

        /* The p-th and final block is already in current; no more rotation needed. */
        timed_forces(local, n, current, n, eps2, acc, compute_seconds);
        integrate(local, velocity, acc, n, dt);
    }
}

/* ---------------------------------------------------------------------- */
/* 2-D force decomposition on a q x q grid                                 */
/*                                                                         */
/* Rank = r*q + c owns particles [rank*n, rank*n + n). Block B is the      */
/* q*n particles [B*q*n, (B+1)*q*n), i.e. the ones owned by grid row B.    */
/* Rank (r,c) evaluates target block r against source block c:             */
/*   1. Allgather in row r   -> target block r (N/q records)               */
/*   2. Bcast in column c    -> source block c, root = diagonal rank (c,c) */
/*      (the diagonal rank already holds block c from step 1)              */
/*   3. local N/q x N/q force tile (N^2/p pairs)                           */
/*   4. Reduce_scatter in row r (sum over c) -> own n accelerations        */
/* ---------------------------------------------------------------------- */
static void run_force2d(double *local, double *tgt, double *src_buf,
                        double *partial_acc, double *velocity, double *acc,
                        int n, int q, int rank, int steps,
                        double dt, double eps2,
                        MPI_Comm row_comm, MPI_Comm col_comm,
                        double *compute_seconds, double *comm_seconds)
{
    const int r = rank / q;
    const int c = rank % q;
    const int block_n = q * n;
    const int record_count = n * STRIDE;
    const int block_records = block_n * STRIDE;
    /* Diagonal ranks already own source block c in `tgt`. */
    double *src = (r == c) ? tgt : src_buf;

    for (int step = 0; step < steps; ++step) {
        double t0;
        memset(partial_acc, 0, (size_t)DIM * (size_t)block_n * sizeof(double));

        t0 = MPI_Wtime();
        MPI_Allgather(local, record_count, MPI_DOUBLE,
                      tgt, record_count, MPI_DOUBLE, row_comm);
        MPI_Bcast(src, block_records, MPI_DOUBLE, c, col_comm);
        *comm_seconds += MPI_Wtime() - t0;

        timed_forces(tgt, block_n, src, block_n, eps2, partial_acc, compute_seconds);

        t0 = MPI_Wtime();
        MPI_Reduce_scatter_block(partial_acc, acc, DIM * n, MPI_DOUBLE, MPI_SUM, row_comm);
        *comm_seconds += MPI_Wtime() - t0;

        integrate(local, velocity, acc, n, dt);
    }
}

/* ---------------------------------------------------------------------- */
/* Validation checksums (outside the timed region)                         */
/*   out[0] kinetic energy   out[1] sum(|x|+|y|+|z|)                       */
/*   out[2..4] momentum sum m*v   out[5] sum m*|v|                         */
/* Total momentum must stay ~0: pair forces are equal and opposite.        */
/* ---------------------------------------------------------------------- */
static void local_checksums(const double *local, const double *velocity,
                            int n, double out[6])
{
    for (int k = 0; k < 6; ++k) {
        out[k] = 0.0;
    }
    for (int i = 0; i < n; ++i) {
        const double m = local[(size_t)STRIDE * (size_t)i + 3];
        const double vx = velocity[(size_t)DIM * (size_t)i + 0];
        const double vy = velocity[(size_t)DIM * (size_t)i + 1];
        const double vz = velocity[(size_t)DIM * (size_t)i + 2];

        out[0] += 0.5 * m * (vx * vx + vy * vy + vz * vz);
        out[1] += fabs(local[(size_t)STRIDE * (size_t)i + 0]) +
                  fabs(local[(size_t)STRIDE * (size_t)i + 1]) +
                  fabs(local[(size_t)STRIDE * (size_t)i + 2]);
        out[2] += m * vx;
        out[3] += m * vy;
        out[4] += m * vz;
        out[5] += m * sqrt(vx * vx + vy * vy + vz * vz);
    }
}

int main(int argc, char **argv)
{
    int rank, p;
    int N = 0, steps = 0;
    double dt = 0.0, eps = 0.0, eps2;
    const char *mode_name;
    Mode mode;
    int local_n;
    int q = 1;
    double *local, *velocity, *acc;
    double *all_particles = NULL;
    double *block_a = NULL, *block_b = NULL;
    double *tgt = NULL, *src_buf = NULL, *partial_acc = NULL;
    MPI_Comm row_comm = MPI_COMM_NULL, col_comm = MPI_COMM_NULL;
    double compute_local = 0.0, comm_local = 0.0, total_local;
    double compute_max = 0.0, comm_max = 0.0, total_max = 0.0;
    double start_total;
    double receive_bytes, local_interactions, bytes_per_flop;
    double buffer_bytes = 0.0;
    double chk_local[6], chk[6];

    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &p);

    if (argc != 6) {
        if (rank == 0) {
            fprintf(stderr,
                "Usage: %s N steps dt epsilon allgather|ring_blocking|ring|force2d\n"
                "Example: mpirun -np 4 %s 4096 10 0.0005 0.01 ring\n",
                argv[0], argv[0]);
        }
        MPI_Finalize();
        return EXIT_FAILURE;
    }

    if (!parse_positive_int(argv[1], "N", &N) ||
        !parse_positive_int(argv[2], "steps", &steps) ||
        !parse_positive_double(argv[3], "dt", &dt) ||
        !parse_positive_double(argv[4], "epsilon", &eps)) {
        MPI_Abort(MPI_COMM_WORLD, EXIT_FAILURE);
    }
    mode_name = argv[5];
    if (strcmp(mode_name, "ring") == 0) {
        mode = MODE_RING;
    } else if (strcmp(mode_name, "ring_blocking") == 0) {
        mode = MODE_RING_BLOCKING;
    } else if (strcmp(mode_name, "allgather") == 0) {
        mode = MODE_ALLGATHER;
    } else if (strcmp(mode_name, "force2d") == 0) {
        mode = MODE_FORCE2D;
    } else {
        if (rank == 0) {
            fprintf(stderr,
                "Mode must be allgather, ring_blocking, ring or force2d (got '%s')\n",
                mode_name);
        }
        MPI_Abort(MPI_COMM_WORLD, EXIT_FAILURE);
        return EXIT_FAILURE;
    }

    if (N % p != 0) {
        if (rank == 0) {
            fprintf(stderr, "N (%d) must be divisible by p (%d).\n", N, p);
        }
        MPI_Abort(MPI_COMM_WORLD, EXIT_FAILURE);
    }
    if (N > INT_MAX / STRIDE) {
        if (rank == 0) {
            fprintf(stderr, "N is too large for MPI int counts; use N <= %d.\n",
                    INT_MAX / STRIDE);
        }
        MPI_Abort(MPI_COMM_WORLD, EXIT_FAILURE);
    }
    if (mode == MODE_FORCE2D) {
        q = (int)(sqrt((double)p) + 0.5);
        if (q * q != p) {
            if (rank == 0) {
                fprintf(stderr, "force2d needs a square rank count (got p=%d).\n", p);
            }
            MPI_Abort(MPI_COMM_WORLD, EXIT_FAILURE);
        }
    }

    local_n = N / p;
    eps2 = eps * eps;

    local = (double *)checked_malloc((size_t)local_n * STRIDE * sizeof(double), rank);
    velocity = (double *)checked_calloc((size_t)local_n * DIM, sizeof(double), rank);
    acc = (double *)checked_malloc((size_t)local_n * DIM * sizeof(double), rank);

    switch (mode) {
    case MODE_RING:
    case MODE_RING_BLOCKING:
        block_a = (double *)checked_malloc((size_t)local_n * STRIDE * sizeof(double), rank);
        block_b = (double *)checked_malloc((size_t)local_n * STRIDE * sizeof(double), rank);
        buffer_bytes = 2.0 * (double)local_n * STRIDE * sizeof(double);
        break;
    case MODE_FORCE2D: {
        const size_t block_n = (size_t)q * (size_t)local_n;
        tgt = (double *)checked_malloc(block_n * STRIDE * sizeof(double), rank);
        src_buf = (double *)checked_malloc(block_n * STRIDE * sizeof(double), rank);
        partial_acc = (double *)checked_malloc(block_n * DIM * sizeof(double), rank);
        MPI_Comm_split(MPI_COMM_WORLD, rank / q, rank % q, &row_comm);
        MPI_Comm_split(MPI_COMM_WORLD, rank % q, rank / q, &col_comm);
        buffer_bytes = (double)block_n * (2.0 * STRIDE + DIM) * sizeof(double);
        break;
    }
    default: /* MODE_ALLGATHER */
        all_particles = (double *)checked_malloc((size_t)N * STRIDE * sizeof(double), rank);
        buffer_bytes = (double)N * STRIDE * sizeof(double);
        break;
    }

    initialize_particles(local, local_n, rank);
    MPI_Barrier(MPI_COMM_WORLD);
    start_total = MPI_Wtime();

    switch (mode) {
    case MODE_RING:
        run_ring(local, block_a, block_b, velocity, acc, local_n, p, rank,
                 steps, dt, eps2, &compute_local, &comm_local);
        break;
    case MODE_RING_BLOCKING:
        run_ring_blocking(local, block_a, block_b, velocity, acc, local_n, p, rank,
                          steps, dt, eps2, &compute_local, &comm_local);
        break;
    case MODE_FORCE2D:
        run_force2d(local, tgt, src_buf, partial_acc, velocity, acc, local_n, q, rank,
                    steps, dt, eps2, row_comm, col_comm, &compute_local, &comm_local);
        break;
    default:
        run_allgather(local, all_particles, velocity, acc, local_n, p,
                      steps, dt, eps2, &compute_local, &comm_local);
        break;
    }

    total_local = MPI_Wtime() - start_total;
    MPI_Reduce(&total_local, &total_max, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
    MPI_Reduce(&compute_local, &compute_max, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
    MPI_Reduce(&comm_local, &comm_max, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);

    local_checksums(local, velocity, local_n, chk_local);
    MPI_Reduce(chk_local, chk, 6, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);

    /* Logical per-rank incoming payload over all steps (worst-case rank).
     * 1-D modes: p-1 foreign blocks of n records per step.
     * force2d  : row allgather (q-1 blocks) + column bcast (q*n records, not on
     *            the diagonal) + reduce-scatter (q-1 segments of 3n doubles;
     *            a lower bound, the library algorithm may move more). */
    if (mode == MODE_FORCE2D) {
        const double rec = (double)local_n * STRIDE * sizeof(double);
        const double seg = (double)local_n * DIM * sizeof(double);
        receive_bytes = (double)steps *
            ((double)(q - 1) * rec + (q > 1 ? (double)q * rec : 0.0) + (double)(q - 1) * seg);
    } else {
        receive_bytes = (double)steps * (double)(p - 1) * (double)local_n *
                        (double)STRIDE * (double)sizeof(double);
    }
    local_interactions = (double)steps * (double)local_n * (double)N;
    bytes_per_flop = receive_bytes / (local_interactions * APPROX_FLOPS_PER_PAIR);

    if (rank == 0) {
        const double comm_comp_ratio = (compute_max > 0.0) ? comm_max / compute_max : 0.0;
        const double mom = sqrt(chk[2] * chk[2] + chk[3] * chk[3] + chk[4] * chk[4]);
        const double mom_rel = (chk[5] > 0.0) ? mom / chk[5] : 0.0;
        printf("Mode: %s | N=%d | ranks=%d | local_particles=%d | steps=%d\n",
               mode_name, N, p, local_n, steps);
        printf("Timing uses the maximum rank time; compute time covers force kernels.\n");
        printf("RESULT,%s,%d,%d,%d,%.9f,%.9f,%.9f,%.6f,%.0f,%.0f,%.9e,%.0f,%.12e,%.12e,%.3e\n",
               mode_name, N, p, steps, total_max, compute_max, comm_max,
               comm_comp_ratio, receive_bytes, local_interactions, bytes_per_flop,
               buffer_bytes, chk[0], chk[1], mom_rel);
        printf("CSV columns: mode,N,ranks,steps,total_s,compute_s,comm_exposed_s,"
               "comm_compute_ratio,recv_bytes_per_rank,interactions_per_rank,"
               "recv_bytes_per_flop,buffer_bytes_per_rank,kinetic_energy,pos_l1,"
               "momentum_rel\n");
    }

    if (row_comm != MPI_COMM_NULL) {
        MPI_Comm_free(&row_comm);
    }
    if (col_comm != MPI_COMM_NULL) {
        MPI_Comm_free(&col_comm);
    }
    free(local);
    free(velocity);
    free(acc);
    free(all_particles);
    free(block_a);
    free(block_b);
    free(tgt);
    free(src_buf);
    free(partial_acc);
    MPI_Finalize();
    return EXIT_SUCCESS;
}
