"""
util.py
=======
Helper / glue layer for reproducing the figures of

    S. Lamata-Otin, F. Malizia, V. Latora, M. Frasca & J. Gomez-Gardenes,
    "Hyperedge overlap drives synchronizability of systems with higher-order
    interactions", Phys. Rev. E 111, 034302 (2025).

DESIGN RULES (per assignment):
  * We DO NOT modify any file in the repository.
  * Wherever a repo function is correct and importable, we re-use it.
  * Wherever a repo function has a bug that prevents import/execution, we
    provide a fixed re-implementation here, clearly labelled with a
    "REPO-FIX" note explaining what was wrong.
  * Missing helpers referenced by the repo (e.g. `sort_edge` from the
    non-existent `misc` module) are supplied here.

This module is grown figure-by-figure. Sections are labelled so you can see
which figure introduced which helper.

Author: (your name) — course re-implementation
"""

import os
import math
import itertools
import numpy as np


# ---------------------------------------------------------------------------
# Repo path handling
# ---------------------------------------------------------------------------
# All notebooks are expected to live in the repository root (next to this
# file), so the repo sub-folders are addressed relative to REPO_ROOT.
REPO_ROOT = os.path.dirname(os.path.abspath(__file__))

CHAOS_DIR   = os.path.join(REPO_ROOT, "HO-Chaotic-Oscillators")
REAL_DIR    = os.path.join(CHAOS_DIR, "Real_structures")
OVERLAP_DIR = os.path.join(REPO_ROOT, "Hyperedge overlap matrix")
HS_DIR      = os.path.join(OVERLAP_DIR, "Hs")
STRUCTGEN_DIR = os.path.join(REPO_ROOT, "Structure generation")
NATCOMMS_DIR  = os.path.join(REPO_ROOT, "Hyperedge Overlap")


# ===========================================================================
#  SECTION A — Generalized Laplacians & effective Laplacian spectrum
#  (used by Fig 3, 4, 5, 6, 7, 8)
# ===========================================================================
#
# REPO-FIX. The repo file
#     HO-Chaotic-Oscillators/Synchronizability_analysis.py
# cannot be imported as-is because:
#   (1) line `from itertools import combination`  -> should be `combinations`
#       (and is in fact unused);
#   (2) it uses `np.math.factorial`, removed in NumPy >= 1.25 (we are on 2.x).
# The maths of `compute_laplacian` / `compute_effective_laplacian` /
# `All_Eigenvalues_Computation` are otherwise exactly Eqs. (1),(2),(14).
# We reproduce them here verbatim except for those two fixes.

def compute_laplacian(edges_array, order):
    """Generalized Laplacian L^(m) of a single order (Eqs. 1-2).

    Parameters
    ----------
    edges_array : (E, order+1) int ndarray
        The hyperedges of the given order. Node indices must be contiguous
        0..N-1 over the WHOLE hypergraph; see `build_order_laplacian` which
        pins N explicitly so different orders share one node set.
    order : int
        1 for pairwise, 2 for triangles (3-body), 3 for 4-body, ...

    Returns
    -------
    L : (N, N) float ndarray
    k_array : (N,) float ndarray  generalized degree k^(m)_i
    """
    edges_array = np.asarray(edges_array)
    N = int(edges_array.max()) + 1
    return _laplacian_fixed_N(edges_array, order, N)


def _laplacian_fixed_N(edges_array, order, N):
    """Same as compute_laplacian but with N supplied explicitly.

    This is what the notebooks actually call, because for a multi-order
    hypergraph every order must be embedded in the *same* NxN space.
    Implementation mirrors the repo's adjacency-tensor construction but
    uses math.factorial (NumPy 2.x safe).
    """
    edges_array = np.asarray(edges_array)
    fact_order   = math.factorial(order)
    fact_orderm1 = math.factorial(order - 1)

    tensor_shape = tuple([N] * (order + 1))
    A = np.zeros(tensor_shape)
    for row in edges_array:
        for perm in set(itertools.permutations(row)):
            A[perm] = 1

    k_array = A.sum(axis=tuple(range(1, order + 1))) / fact_order

    L = np.zeros((N, N))
    for i in range(N):
        for j in range(N):
            if i != j:
                L[i, j] = -np.sum(A[i, j, ...]) / fact_orderm1
            else:
                L[i, j] = order * k_array[i]
    return L, k_array


def compute_effective_laplacian(L_matrices, sigma_values):
    """Effective Laplacian, Eq. (14): L_eff = sum_m sigma^(m) L^(m).

    Verbatim re-use of the repo logic (that function is correct); reproduced
    here so notebooks need only import `util`.
    """
    if len(L_matrices) != len(sigma_values):
        raise ValueError("The number of Laplacians and sigma values must match.")
    L_eff = np.zeros_like(L_matrices[0], dtype=float)
    for L, s in zip(L_matrices, sigma_values):
        if L.shape[0] != L.shape[1]:
            raise ValueError("Each Laplacian matrix must be square.")
        L_eff += s * L
    return L_eff


def sorted_eigenvalues(L_eff):
    """Ascending real eigenvalues of a (symmetric, PSD) effective Laplacian.

    The repo uses np.linalg.eig; the effective Laplacian is symmetric so we
    use eigvalsh (faster, guaranteed real, correct ordering).
    """
    w = np.linalg.eigvalsh((L_eff + L_eff.T) / 2.0)
    return np.sort(w.real)


def effective_laplacian_spectrum(edges_by_order, orders, sigmas, N=None):
    """Convenience wrapper: build L^(m) for each order on a common N, weight
    by sigma^(m), and return the sorted spectrum of the effective Laplacian.

    Parameters
    ----------
    edges_by_order : list of (E_m, order_m+1) int ndarrays
    orders         : list of int, same length
    sigmas         : list of float sigma^(m), same length
    N              : int or None. If None, inferred as max node index + 1
                     across all orders.

    Returns
    -------
    eigs : (N,) ndarray, ascending
    L_eff : (N, N) ndarray
    Ls    : list of per-order Laplacians
    ks    : list of per-order generalized-degree arrays
    """
    if N is None:
        N = 0
        for e in edges_by_order:
            e = np.asarray(e)
            if e.size:
                N = max(N, int(e.max()) + 1)
    Ls, ks = [], []
    for e, m in zip(edges_by_order, orders):
        L, k = _laplacian_fixed_N(np.asarray(e), m, N)
        Ls.append(L)
        ks.append(k)
    L_eff = compute_effective_laplacian(Ls, sigmas)
    eigs = sorted_eigenvalues(L_eff)
    return eigs, L_eff, Ls, ks


# ===========================================================================
#  SECTION B — Rossler dynamics on a hypergraph (Figs 3, 7, 8)
# ===========================================================================
#
# REPO-FIX. The repo file
#     HO-Chaotic-Oscillators/Numerical_integration_rossler.py
# does not import at all:
#   (1) `rossler_fun_natural_coupling_type2` is missing a closing ')' on the
#       Y_hist update line -> SyntaxError;
#   (2) the file ends with '""""' (four quotes) -> SyntaxError;
#   (3) in that same type-2 function the SECOND-order term couples on X
#       instead of Y (copy-paste from the type-3 body). Per Eq. (16) the
#       class-II system couples through the y variable at BOTH orders.
# Below we re-implement the three integrators + grid drivers cleanly, keeping
# the repo's Euler scheme, resampling, and error/order-parameter definitions
# identical so results match. numba is optional: if unavailable we fall back
# to a pure-python/numpy path (slower, but the assignment env has no numba).

try:
    from numba import njit, prange          # noqa: F401
    _HAVE_NUMBA = True
except Exception:                            # pragma: no cover
    _HAVE_NUMBA = False
    def njit(*args, **kwargs):
        def _wrap(f):
            return f
        if args and callable(args[0]):
            return args[0]
        return _wrap
    prange = range


# --- integrators -----------------------------------------------------------
# All three integrate N Rossler oscillators (Eq. 15, a=0.2,b=0.2,c=9) with
# 1-body (pairwise) and 2-body (triangle) diffusive coupling. They differ
# only in WHICH state variable carries the coupling and the coupling kernel:
#   type3 : couple x with h1(x)=x^3,  h2(x_j,x_k)=x_j^2 x_k   (class III, Eq.17)
#   type2 : couple y with h1(y)=y^3,  h2(y_j,y_k)=y_j^2 y_k   (class II,  Eq.16)
#
# `list_neighbors` : list of int arrays, pairwise neighbours of each node.
# `triangles_list` : (T,3) int array of 2-hyperedges.

@njit
def _rossler_type3(N, list_neighbors, triangles_list, a, b, c, s1, s2, h, numsteps):
    X = np.random.random(N); Y = np.random.random(N); Z = np.random.random(N)
    Xh = np.zeros((N, numsteps + 1)); Yh = np.zeros((N, numsteps + 1)); Zh = np.zeros((N, numsteps + 1))
    theta = np.zeros((N, numsteps + 1))
    Xh[:, 0] = X; Yh[:, 0] = Y; Zh[:, 0] = Z
    for i in range(N):
        theta[i, 0] = np.arctan2(Y[i], X[i])
    for t in range(numsteps):
        for i in range(N):
            fo = 0.0
            for j in list_neighbors[i]:
                fo += Xh[j, t] ** 3 - Xh[i, t] ** 3
            so = 0.0
            for tri in triangles_list:
                n0 = tri[0]; j = tri[1]; k = tri[2]
                if i == n0 or i == j or i == k:
                    so += ((Xh[j, t] ** 2) * Xh[k, t] - Xh[i, t] ** 3) \
                        + ((Xh[k, t] ** 2) * Xh[j, t] - Xh[i, t] ** 3)
            Xh[i, t + 1] = Xh[i, t] + h * (-Yh[i, t] - Zh[i, t] + s1 * fo + s2 * so)
            Yh[i, t + 1] = Yh[i, t] + h * (Xh[i, t] + a * Yh[i, t])
            Zh[i, t + 1] = Zh[i, t] + h * (b + Zh[i, t] * (Xh[i, t] - c))
            theta[i, t + 1] = np.arctan2(Yh[i, t + 1], Xh[i, t + 1])
    return Xh, Yh, Zh, theta


@njit
def _rossler_type2(N, list_neighbors, triangles_list, a, b, c, s1, s2, h, numsteps):
    X = np.random.random(N); Y = np.random.random(N); Z = np.random.random(N)
    Xh = np.zeros((N, numsteps + 1)); Yh = np.zeros((N, numsteps + 1)); Zh = np.zeros((N, numsteps + 1))
    theta = np.zeros((N, numsteps + 1))
    Xh[:, 0] = X; Yh[:, 0] = Y; Zh[:, 0] = Z
    for i in range(N):
        theta[i, 0] = np.arctan2(Y[i], X[i])
    for t in range(numsteps):
        for i in range(N):
            fo = 0.0
            for j in list_neighbors[i]:
                fo += Yh[j, t] ** 3 - Yh[i, t] ** 3          # REPO-FIX: couple on Y
            so = 0.0
            for tri in triangles_list:
                n0 = tri[0]; j = tri[1]; k = tri[2]
                if i == n0 or i == j or i == k:
                    so += ((Yh[j, t] ** 2) * Yh[k, t] - Yh[i, t] ** 3) \
                        + ((Yh[k, t] ** 2) * Yh[j, t] - Yh[i, t] ** 3)  # REPO-FIX: Y not X
            Xh[i, t + 1] = Xh[i, t] + h * (-Yh[i, t] - Zh[i, t])
            Yh[i, t + 1] = Yh[i, t] + h * (Xh[i, t] + a * Yh[i, t] + s1 * fo + s2 * so)
            Zh[i, t + 1] = Zh[i, t] + h * (b + Zh[i, t] * (Xh[i, t] - c))
            theta[i, t + 1] = np.arctan2(Yh[i, t + 1], Xh[i, t + 1])
    return Xh, Yh, Zh, theta


@njit
def _sync_error(N, X, Y, Z, nsteps):
    err = np.zeros(nsteps)
    for t in range(nsteps):
        s = 0.0
        cnt = 0
        for i in range(N):
            for j in range(N):
                dx = X[i, t] - X[j, t]
                dy = Y[i, t] - Y[j, t]
                dz = Z[i, t] - Z[j, t]
                s += np.sqrt(dx * dx + dy * dy + dz * dz)
                cnt += 1
        err[t] = s / cnt
    return err


@njit
def _r_order(N, theta, nsteps):
    r = np.zeros(nsteps)
    for t in range(nsteps):
        rs = 0.0; ims = 0.0
        for i in range(N):
            rs += np.cos(theta[i, t]); ims += np.sin(theta[i, t])
        r[t] = np.sqrt(rs * rs + ims * ims) / N
    return r


def _resample(X, Y, Z, theta, n_resampling, numsteps):
    idx = np.arange(0, numsteps, n_resampling)
    return X[:, idx], Y[:, idx], Z[:, idx], theta[:, idx]


def rossler_error_grid(N, list_neighbors, triangles_list, s1_, s2_,
                       a=0.2, b=0.2, c=9.0, h=1e-3, n_steps=1500,
                       n_resampling=200, coupling="type3", seed=None,
                       burn_frac_index=20, verbose=True):
    """Sweep (s1,s2) and return the mean synchronization-error grid <E>.

    This reproduces the repo's `rossler_errors_grid_natural_coupling_type{2,3}`
    driver. Returns (errors_grid, order_grid), both shape (len(s2_), len(s1_)),
    matching the paper's convention (rows = s2, cols = s1).

    NOTE ON RUNTIME: with the paper settings (h=1e-3, n_steps=1500 ->
    1.5M Euler steps per (s1,s2) cell, 25x25 grid) this is extremely heavy in
    pure python. Without numba, start with a COARSE grid / shorter n_steps to
    sanity-check, then scale up. See the notebook for guidance.
    """
    if seed is not None:
        np.random.seed(seed)
    numsteps = int(n_steps / h)
    integ = _rossler_type3 if coupling == "type3" else _rossler_type2

    # numba wants a typed list of int arrays for list_neighbors
    if _HAVE_NUMBA:
        from numba.typed import List
        nb_neighbors = List()
        for arr in list_neighbors:
            nb_neighbors.append(np.asarray(arr, dtype=np.int64))
        neighbors = nb_neighbors
    else:
        neighbors = [np.asarray(a_, dtype=np.int64) for a_ in list_neighbors]

    tri = np.asarray(triangles_list, dtype=np.int64)
    ns2, ns1 = len(s2_), len(s1_)
    errors = np.zeros((ns2, ns1))
    order = np.zeros((ns2, ns1))

    for i2, s2 in enumerate(s2_):
        for i1, s1 in enumerate(s1_):
            Xh, Yh, Zh, th = integ(N, neighbors, tri, a, b, c,
                                    float(s1), float(s2), h, numsteps)
            Xr, Yr, Zr, thr = _resample(Xh, Yh, Zh, th, n_resampling, numsteps)
            nrs = Xr.shape[1]
            se = _sync_error(N, Xr, Yr, Zr, nrs)
            r = _r_order(N, thr, nrs)
            errors[i2, i1] = np.mean(se[burn_frac_index:])
            order[i2, i1] = np.mean(r[burn_frac_index:])
        if verbose:
            print(f"  s2 row {i2 + 1}/{ns2} done (s2={s2:.3e})")
    return errors, order


# ===========================================================================
#  SECTION C — Hypergraph I/O and neighbour lists (Figs 3, 8)
# ===========================================================================

def load_edges(path):
    """Load a whitespace/tab separated edge or triplet list as int ndarray."""
    return np.loadtxt(path, dtype=np.int64)


def neighbors_from_pairs(pairs, N=None):
    """Build a list-of-arrays adjacency (undirected) from a pair list.

    Returns `list_neighbors` such that list_neighbors[i] is an int array of
    the pairwise neighbours of node i — the format the integrators expect.
    """
    pairs = np.asarray(pairs, dtype=np.int64)
    if N is None:
        N = int(pairs.max()) + 1
    adj = [set() for _ in range(N)]
    for a_, b_ in pairs:
        adj[a_].add(b_)
        adj[b_].add(a_)
    return [np.array(sorted(s), dtype=np.int64) for s in adj], N


def load_real_structure(name):
    """Load one of the shipped real-world hypergraphs used in Fig 8.

    name in {'Karate_club_original','Karate_club_modified',
             'Cat_brain_original','Cat_brain_modified'}.

    Returns dict with pairs, triplets, list_neighbors, N.
    """
    pairs = load_edges(os.path.join(REAL_DIR, f"{name}_Pairs.txt"))
    trips = load_edges(os.path.join(REAL_DIR, f"{name}_Triplets.txt"))
    N = int(max(pairs.max(), trips.max())) + 1
    list_neighbors, _ = neighbors_from_pairs(pairs, N)
    return dict(pairs=pairs, triplets=trips, list_neighbors=list_neighbors, N=N)


# ---------------------------------------------------------------------------
# Missing helper referenced by Structure generation/Minimization-algorithm.py
# via `from misc import *`. The `misc` module is absent from the repo, so we
# provide the obvious implementation (sort a 2-tuple into a canonical order).
# ---------------------------------------------------------------------------
try:
    from numba import njit as _njit_se
    @_njit_se
    def sort_edge(pair):
        a, b = pair
        if a <= b:
            return (a, b)
        return (b, a)
except Exception:
    def sort_edge(pair):
        a, b = pair
        return (a, b) if a <= b else (b, a)


# ===========================================================================
#  SECTION D — Structure generators for M>=2 (Figs 4, 5, 6, 7)
# ===========================================================================
#
# The repo ships generation code only for M<=2 as turnkey functions, and those
# files hard-import numba (`from numba import jit`) so they will not import in a
# numba-less environment. Below we port the algorithms we need with a
# numba-optional path, and add the M=3 constructions that the PAPER describes
# explicitly (Fig. 4 / Fig. 5 captions) but that the repo does not ship as a
# single driver.
#
# Every generator returns node indices contiguous 0..N-1 so the Laplacian
# helpers in Section A can embed all orders on a common N.

import networkx as _nx
import random as _random
from itertools import combinations as _combinations


# --- D.1 random regular structures per order (I~0, T~0) --------------------
def random_regular_pairs(N, k1, seed=0):
    """Random k1-regular pairwise backbone (independent of higher orders)."""
    G = _nx.random_regular_graph(k1, N, seed=seed)
    return np.array(sorted(tuple(sorted(e)) for e in G.edges()), dtype=np.int64)


def random_regular_mbody(N, km, order, seed=0, max_tries=1_000_000):
    """Random (order+1)-uniform hypergraph, each node in ~km hyperedges.

    Random placement => intra-order overlap T^(order) ~ 0, and being drawn
    independently of other orders => inter-order overlap ~ 0.
    order=1 -> pairs, order=2 -> triangles, order=3 -> tetrahedra, ...
    """
    rng = np.random.default_rng(seed)
    size = order + 1
    target = km * N // size
    deg = np.zeros(N, dtype=int)
    groups = set()
    tries = 0
    while len(groups) < target and tries < max_tries:
        tries += 1
        avail = np.where(deg < km)[0]
        if len(avail) < size:
            break
        g = tuple(sorted(rng.choice(avail, size, replace=False)))
        if g not in groups:
            groups.add(g)
            for x in g:
                deg[x] += 1
    return np.array(sorted(groups), dtype=np.int64), deg


# --- D.2 max intra-order overlap via disjoint fully-connected blocks -------
#     This is the mesoscale organization the paper invokes for T=1
#     ("N/size subsets of `size` nodes, each fully connected").
def clustered_blocks(N, size, orders, seed=0):
    """Partition N nodes into disjoint blocks of `size`, each block fully
    connected by hyperedges of every order in `orders`.

    For a block of B nodes and order m, we add all C(B, m+1) hyperedges.
    Each node then has generalized degree k^(m) = C(B-1, m) within its block,
    and the set of m-hyperedges is DISCONNECTED (one component per block) =>
    intra-order overlap T^(m) = 1 for every m (maximal), while inter-order
    overlap between the orders is maximal too (downward closure inside blocks).

    Returns dict: {m: (E_m, m+1) int ndarray}.  N is truncated to a multiple
    of `size`.
    """
    _random.seed(seed)
    nblocks = N // size
    Neff = nblocks * size
    perm = list(range(Neff))
    _random.shuffle(perm)
    blocks = [perm[b * size:(b + 1) * size] for b in range(nblocks)]
    out = {}
    for m in orders:
        edges = []
        for blk in blocks:
            for combo in _combinations(sorted(blk), m + 1):
                edges.append(combo)
        out[m] = np.array(sorted(edges), dtype=np.int64)
    return out, Neff


def block_size_for_degrees(target_degrees):
    """Given desired {order: k^(order)} inside a fully-connected block, find
    the block size B with C(B-1, order) == k for all requested orders.

    Example: {2:6, 3:4} -> B=5  (C(4,2)=6, C(4,3)=4).
    Raises ValueError if no single B satisfies all constraints.
    """
    from math import comb
    candidate = None
    for B in range(2, 60):
        ok = all(comb(B - 1, m) == k for m, k in target_degrees.items())
        if ok:
            candidate = B
            break
    if candidate is None:
        raise ValueError(f"No block size B fits degrees {target_degrees}")
    return candidate


# --- D.3 NatComms max-intra simplicial complex + intra minimization --------
#     Ported from `Hyperedge Overlap/Hyperedge-overlap.py` (numba-optional).
#     Builds an order-2 (triangles) structure with T^(2)=1 whose k^(2) is
#     fixed by k1 as (k1-1)(k1-2)/2, then rewires to LOWER T^(2) toward 0.
def natcomms_max_overlap_simplicial_complex(n, k, seed=None):
    """Regular order-2 simplicial complex with maximal intra-order overlap.
    Node count of the RESULT is N = n*k. Each node: k^(1)=k,
    k^(2)=(k-1)(k-2)/2. (Verbatim port of the repo function.)"""
    if seed is not None:
        _random.seed(seed)
        np.random.seed(seed)
    M = _nx.random_regular_graph(k, n, seed=seed)
    graph_to_simplex = {ii: [ii * k + r for r in range(k)] for ii in range(n)}
    G = _nx.disjoint_union_all([_nx.complete_graph(k) for _ in range(n)])
    triangle_list = [c for c in _nx.enumerate_all_cliques(G) if len(c) == 3]
    for edge in M.edges():
        m1, m2 = edge
        n1 = _random.choice(graph_to_simplex[m1])
        n2 = _random.choice(graph_to_simplex[m2])
        G.add_edge(n1, n2)
        graph_to_simplex[m1].remove(n1)
        graph_to_simplex[m2].remove(n2)
    N = n * k
    return N, np.array(sorted(tuple(sorted(e)) for e in G.edges()), dtype=np.int64), \
        np.array(triangle_list, dtype=np.int64)
