from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np
import scipy.sparse as sp
from scipy import linalg
from scipy.spatial import KDTree
from scipy.stats import norm
from tqdm import tqdm


@dataclass
class NNGPFactors:
    # For each target point i: indices of neighbors in prior set
    neighbor_idx: List[np.ndarray]
    # For each target point i: coefficients a_i of length m_i (so that mu_i = a_i^T z(N_i))
    a_rows: List[np.ndarray]
    # For each target point i: conditional variance d_i (scalar)
    d: np.ndarray


def _ensure_2d(X: np.ndarray) -> np.ndarray:
    X = np.asarray(X, float)
    if X.ndim == 1:
        X = X[None, :]
    return X


def build_nngp_factors(
    coords: np.ndarray,
    M: int,
    kernel,
    jitter: float = 1e-8,
    order: Optional[np.ndarray] = None,
    show_progress: bool = False,
) -> Tuple[NNGPFactors, np.ndarray]:
    """
    Build Vecchia/NNGP factors (A,D) for a single set of points with an ordering.
    For each i, pick neighbors N_i from predecessors {0..i-1} by proximity (KDTree),
    then compute a_i = K_{iN} K_{NN}^{-1} and d_i = K_{ii} - K_{iN} K_{NN}^{-1} K_{Ni}.

    Returns factors in original index order along with the order array (perm indices).
    """
    X = _ensure_2d(coords)
    n = X.shape[0]
    if n == 0:
        return NNGPFactors([], [], np.zeros(0)), np.arange(0)
    if order is None:
        order = np.arange(n)
    order = np.asarray(order, dtype=int)
    Xo = X[order]

    tree = KDTree(Xo)
    neighbor_idx_ord: List[np.ndarray] = []
    a_rows_ord: List[np.ndarray] = []
    d_ord = np.zeros(n, dtype=float)

    for i in tqdm(range(n), disable=not show_progress):
        if i == 0:
            neighbor_idx_ord.append(np.zeros(0, dtype=int))
            a_rows_ord.append(np.zeros(0, dtype=float))
            # K_ii
            d_ord[i] = float(kernel.K(Xo[i : i + 1], Xo[i : i + 1])[0, 0])
            continue

        # Get candidate neighbors and keep predecessors only
        # Start with 2*M and increase if needed
        k_query = min(max(2 * M, M + 5), i + 1)
        idx = tree.query(Xo[i], k=k_query)[1]
        idx = np.atleast_1d(idx)
        # Remove self and keep predecessors
        mask = idx < i
        idx = idx[mask]
        # If insufficient predecessors, take all predecessors and select closest M
        if idx.size < min(M, i):
            prev_idx = np.arange(i)
            # Compute distances to all predecessors
            dists = np.sum((Xo[prev_idx] - Xo[i]) ** 2, axis=1)
            take = np.argsort(dists)[: min(M, i)]
            idx = prev_idx[take]
        else:
            # Already sorted by distance from KDTree
            idx = idx[:M]

        m = idx.size
        if m == 0:
            neighbor_idx_ord.append(np.zeros(0, dtype=int))
            a_rows_ord.append(np.zeros(0, dtype=float))
            d_ord[i] = float(kernel.K(Xo[i : i + 1], Xo[i : i + 1])[0, 0])
            continue

        XN = Xo[idx]
        xi = Xo[i : i + 1]
        K_NN = kernel.K(XN, XN).copy()
        # jitter for stability
        K_NN[np.diag_indices_from(K_NN)] += jitter
        k_iN = kernel.K(xi, XN).ravel()
        k_ii = float(kernel.K(xi, xi)[0, 0])

        # Solve K_NN w = k_Ni (where k_Ni = k_iN^T)
        try:
            c, lower = linalg.cho_factor(K_NN)
            w = linalg.cho_solve((c, lower), k_iN)
        except linalg.LinAlgError:
            w = linalg.solve(K_NN, k_iN)

        a = w  # length m
        d = max(k_ii - float(k_iN @ w), 0.0)

        neighbor_idx_ord.append(idx)
        a_rows_ord.append(a)
        d_ord[i] = d

    # Map back to original index order
    inv = np.empty(n, dtype=int)
    inv[order] = np.arange(n)

    neighbor_idx: List[np.ndarray] = [None] * n
    a_rows: List[np.ndarray] = [None] * n
    d = np.zeros(n, dtype=float)
    for i_ord in range(n):
        i_orig = order[i_ord]
        # neighbor indices need to be mapped back to original indices
        idx_ord = neighbor_idx_ord[i_ord]
        if idx_ord.size:
            idx_orig = order[idx_ord]
            neighbor_idx[i_orig] = idx_orig
        else:
            neighbor_idx[i_orig] = np.zeros(0, dtype=int)
        a_rows[i_orig] = a_rows_ord[i_ord]
        d[i_orig] = d_ord[i_ord]

    return NNGPFactors(neighbor_idx=neighbor_idx, a_rows=a_rows, d=d), order


def build_cross_factors(
    prior_coords: np.ndarray,
    target_coords: np.ndarray,
    M: int,
    kernel,
    jitter: float = 1e-8,
    show_progress: bool = False,
) -> NNGPFactors:
    """
    Build NNGP-like factors for targets conditioning only on the prior set.
    For each target u, pick M nearest neighbors among prior points and compute
    a_u and d_u so that beta(u) | beta(prior) ~ N(a_u^T beta(prior_N), d_u).
    """
    S = _ensure_2d(prior_coords)
    U = _ensure_2d(target_coords)
    nS = S.shape[0]
    nU = U.shape[0]
    if nU == 0:
        return NNGPFactors([], [], np.zeros(0))
    tree = KDTree(S)

    neighbor_idx: List[np.ndarray] = []
    a_rows: List[np.ndarray] = []
    d = np.zeros(nU, dtype=float)

    for i in tqdm(range(nU), disable=not show_progress):
        k = min(M, nS)
        dists, idx = tree.query(U[i], k=k)
        idx = np.atleast_1d(idx)
        XN = S[idx]
        xi = U[i : i + 1]
        K_NN = kernel.K(XN, XN).copy()
        K_NN[np.diag_indices_from(K_NN)] += jitter
        k_iN = kernel.K(xi, XN).ravel()
        k_ii = float(kernel.K(xi, xi)[0, 0])
        try:
            c, lower = linalg.cho_factor(K_NN)
            w = linalg.cho_solve((c, lower), k_iN)
        except linalg.LinAlgError:
            w = linalg.solve(K_NN, k_iN)
        a_rows.append(w)
        neighbor_idx.append(idx)
        d[i] = max(k_ii - float(k_iN @ w), 0.0)

    return NNGPFactors(neighbor_idx=neighbor_idx, a_rows=a_rows, d=d)


def conditional_mean_var_from_factors(
    factors: NNGPFactors,
    values: np.ndarray,
    indices: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Compute conditional means and variances for a set of target indices
    using precomputed factors and training values.

    values: (n_train,) array for a single coefficient evaluated at prior set
    indices: array of target indices aligned with factors lists
    Returns: mu (len(indices),), var (len(indices),)
    """
    mu = np.zeros(len(indices), dtype=float)
    var = np.zeros(len(indices), dtype=float)
    for i, t in enumerate(indices):
        idx = factors.neighbor_idx[t]
        a = factors.a_rows[t]
        if idx.size:
            mu[i] = float(a @ values[idx])
        else:
            mu[i] = 0.0
        var[i] = float(max(factors.d[t], 0.0))
    return mu, var


__all__ = [
    "NNGPFactors",
    "build_nngp_factors",
    "build_cross_factors",
    "conditional_mean_var_from_factors",
    "compute_mf_vf_for_targets",
    "alphas_from_factors",
    "lambda_from_factors",
    "order_points_morton",
]


def compute_mf_vf_for_targets(
    factors: NNGPFactors,
    beta_S: np.ndarray,  # (p+1, nS)
    W_pred: Optional[np.ndarray],  # (nT, p) or None if p==0
    target_indices: np.ndarray,  # indices into factors lists
) -> Tuple[np.ndarray, np.ndarray]:
    """Compute m_f and v_f for targets using precomputed factors and beta at S.

    For coefficient j, beta_j(target) | beta_j(S) ~ N(mu_j, var_j), where
    mu_j, var_j are obtained via conditional_mean_var_from_factors.
    Then m_f = sum_j w_j mu_j, v_f = sum_j w_j^2 var_j with w_0=1 and
    w_j=W_pred[:, j-1] for j>=1.
    """
    beta_S = np.asarray(beta_S, float)
    p_plus_1, nS = beta_S.shape
    nT = len(target_indices)
    m_f = np.zeros(nT, dtype=float)
    v_f = np.zeros(nT, dtype=float)
    for j in range(p_plus_1):
        mu_j, var_j = conditional_mean_var_from_factors(
            factors, beta_S[j, :], target_indices
        )
        if j == 0:
            wj = 1.0
            m_f += mu_j
            v_f += var_j
        else:
            if W_pred is None:
                # no covariates provided; treat as zeros
                continue
            w = W_pred[:, j - 1]
            m_f += w * mu_j
            v_f += (w**2) * var_j
    return m_f, np.maximum(v_f, 0.0)


def alphas_from_factors(
    factors: NNGPFactors,
    beta_S: np.ndarray,
    cand_covs: Optional[np.ndarray],
    cand_idx: np.ndarray,
) -> np.ndarray:
    """Compute thinning acceptance alpha for candidate targets.
    alpha = Phi( - m_f / sqrt(1+v_f) )."""
    m_f, v_f = compute_mf_vf_for_targets(factors, beta_S, cand_covs, cand_idx)
    z = -m_f / np.sqrt(1.0 + v_f)
    alpha = norm.cdf(z)
    return np.clip(alpha, 0.0, 1.0)


def lambda_from_factors(
    factors: NNGPFactors,
    beta_S: np.ndarray,
    W_pred: Optional[np.ndarray],
    lambda_star: float,
    target_indices: np.ndarray,
) -> np.ndarray:
    """Compute lambda(s) = lambda_star * Phi(m_f / sqrt(1+v_f)) for targets."""
    m_f, v_f = compute_mf_vf_for_targets(factors, beta_S, W_pred, target_indices)
    return lambda_star * norm.cdf(m_f / np.sqrt(1.0 + v_f))


class FactorCache:
    """Cache and reuse NNGP factors for S and Grid under a given kernel and M.

    Keeps DRY by centralizing factor builds and common computations (alphas/lambda).
    """

    def __init__(
        self,
        s_coords: np.ndarray,
        grid_points: np.ndarray,
        M: int,
        kernels: Sequence[Any],
    ) -> None:
        self.s_coords = _ensure_2d(s_coords)
        self.grid_points = _ensure_2d(grid_points)
        self.M = int(M)
        kernels_list = list(kernels)
        if not kernels_list:
            raise ValueError("kernels must contain at least one element")
        self.kernels = kernels_list
        self.n_features = len(kernels_list)
        # Use Morton order for S to improve neighbor locality
        self.order_S = order_points_morton(self.s_coords)
        self._build_all_factors()

    def update_if_needed(
        self,
        M: Optional[int] = None,
        kernels: Optional[Sequence[Any]] = None,
    ) -> None:
        changed = False
        if M is not None and int(M) != self.M:
            self.M = int(M)
            changed = True
        if kernels is not None:
            new_kernels = list(kernels)
            if len(new_kernels) != self.n_features:
                raise ValueError(
                    f"kernels must have length {self.n_features}, got {len(new_kernels)}"
                )
            if not self._kernel_specs_equal(new_kernels):
                self.kernels = new_kernels
                changed = True
        if changed:
            self.order_S = order_points_morton(self.s_coords)
            self._build_all_factors()

    def _kernel_specs_equal(self, other_kernels: Sequence[Any]) -> bool:
        if len(other_kernels) != self.n_features:
            return False
        for k_curr, k_new in zip(self.kernels, other_kernels):
            if getattr(k_curr, "lengthscale", None) != getattr(
                k_new, "lengthscale", None
            ) or getattr(k_curr, "variance", None) != getattr(k_new, "variance", None):
                return False
        return True

    def _build_all_factors(self) -> None:
        self._factors_S: List[NNGPFactors] = []
        self._factors_grid: List[NNGPFactors] = []
        self._A_S: List[sp.csr_matrix] = []
        self._A_grid: List[sp.csr_matrix] = []
        self._d_S: List[np.ndarray] = []
        self._d_grid: List[np.ndarray] = []
        cache: Dict[
            Tuple[float, float],
            Tuple[
                NNGPFactors,
                NNGPFactors,
                sp.csr_matrix,
                sp.csr_matrix,
                np.ndarray,
                np.ndarray,
            ],
        ] = {}
        for kernel in self.kernels:
            key = (
                float(getattr(kernel, "lengthscale", 0.0)),
                float(getattr(kernel, "variance", 0.0)),
            )
            if key not in cache:
                factors_S, _ = build_nngp_factors(
                    self.s_coords, self.M, kernel, order=self.order_S
                )
                factors_grid = build_cross_factors(
                    self.s_coords, self.grid_points, self.M, kernel
                )
                A_S = factors_to_csr(factors_S, n_cols=self.s_coords.shape[0])
                A_grid = factors_to_csr(factors_grid, n_cols=self.s_coords.shape[0])
                d_S = np.maximum(factors_S.d, 1e-12)
                d_grid = np.maximum(factors_grid.d, 0.0)
                cache[key] = (factors_S, factors_grid, A_S, A_grid, d_S, d_grid)
            stored = cache[key]
            self._factors_S.append(stored[0])
            self._factors_grid.append(stored[1])
            self._A_S.append(stored[2])
            self._A_grid.append(stored[3])
            self._d_S.append(stored[4])
            self._d_grid.append(stored[5])

    def factors_S_for_feature(self, feature_idx: int) -> NNGPFactors:
        return self._factors_S[feature_idx]

    def factors_grid_for_feature(self, feature_idx: int) -> NNGPFactors:
        return self._factors_grid[feature_idx]

    def A_grid_for_feature(self, feature_idx: int) -> sp.csr_matrix:
        return self._A_grid[feature_idx]

    def d_grid_for_feature(self, feature_idx: int) -> np.ndarray:
        return self._d_grid[feature_idx]

    def sqrt_d_grid_for_feature(self, feature_idx: int) -> np.ndarray:
        return np.sqrt(self._d_grid[feature_idx])

    def factors_S_all(self) -> List[NNGPFactors]:
        return self._factors_S

    def factors_grid_all(self) -> List[NNGPFactors]:
        return self._factors_grid

    def A_grid_all(self) -> List[sp.csr_matrix]:
        return self._A_grid

    def d_grid_all(self) -> List[np.ndarray]:
        return self._d_grid

    # Convenience wrappers (DRY)
    def alphas_for_candidates(
        self,
        beta_S: np.ndarray,
        cand_covs: Optional[np.ndarray],
        cand_idx: np.ndarray,
    ) -> np.ndarray:
        subA_list = [A[cand_idx] for A in self._A_grid]
        p_plus_1, _ = beta_S.shape
        if p_plus_1 != len(subA_list):
            raise ValueError("beta_S feature dimension does not match cached kernels")
        m_f = np.zeros(subA_list[0].shape[0], dtype=float)
        v_f = np.zeros_like(m_f)
        for j in range(p_plus_1):
            mu_j = subA_list[j] @ beta_S[j, :]
            if j == 0:
                m_f += mu_j
                v_f += self._d_grid[j][cand_idx]
            else:
                if cand_covs is None:
                    continue
                w = cand_covs[:, j - 1]
                m_f += w * mu_j
                v_f += (w**2) * self._d_grid[j][cand_idx]
        z = -m_f / np.sqrt(1.0 + v_f)
        return np.clip(norm.cdf(z), 0.0, 1.0)

    def lambda_for_indices(
        self,
        beta_S: np.ndarray,
        W_pred: Optional[np.ndarray],
        lambda_star: float,
        indices: np.ndarray,
    ) -> np.ndarray:
        subA_list = [A[indices] for A in self._A_grid]
        p_plus_1, _ = beta_S.shape
        if p_plus_1 != len(subA_list):
            raise ValueError("beta_S feature dimension does not match cached kernels")
        m_f = np.zeros(subA_list[0].shape[0], dtype=float)
        v_f = np.zeros_like(m_f)
        for j in range(p_plus_1):
            mu_j = subA_list[j] @ beta_S[j, :]
            if j == 0:
                m_f += mu_j
                v_f += self._d_grid[j][indices]
            else:
                if W_pred is None:
                    continue
                w = W_pred[:, j - 1]
                m_f += w * mu_j
                v_f += (w**2) * self._d_grid[j][indices]
        return lambda_star * norm.cdf(m_f / np.sqrt(1.0 + v_f))


def build_grid_interpolation_matrix(
    site_coords: np.ndarray,
    grid_coords: np.ndarray,
    kernel,
    M: int = 10,
    jitter: float = 1e-8,
) -> sp.csr_matrix:
    """Build sparse interpolation matrix from sites to grid using NNGP.

    For each grid point g, find M nearest sites and compute NNGP weights:
        a_g = K(g, neighbors) @ inv(K(neighbors, neighbors))

    Then the conditional mean at grid point g is: a_g @ beta_sites[neighbors]

    Parameters
    ----------
    site_coords : np.ndarray
        Site coordinates, shape (n_sites, 2).
    grid_coords : np.ndarray
        Grid coordinates, shape (n_grid, 2).
    kernel
        Kernel with K(X1, X2) method.
    M : int
        Number of nearest neighbors.
    jitter : float
        Jitter for numerical stability.

    Returns
    -------
    sp.csr_matrix
        Sparse interpolation matrix, shape (n_grid, n_sites).
        A_grid @ beta_sites gives interpolated values at grid points.
    """
    site_coords = _ensure_2d(site_coords)
    grid_coords = _ensure_2d(grid_coords)
    n_sites = site_coords.shape[0]
    n_grid = grid_coords.shape[0]

    if n_sites == 0 or n_grid == 0:
        return sp.csr_matrix((n_grid, n_sites))

    M = min(M, n_sites)

    # Build KDTree on sites
    tree = KDTree(site_coords)

    rows = []
    cols = []
    data = []

    for g in range(n_grid):
        g_coord = grid_coords[g : g + 1]

        # Find M nearest sites
        _, neighbor_idx = tree.query(g_coord, k=M)
        neighbor_idx = np.atleast_1d(neighbor_idx.ravel())

        if neighbor_idx.size == 0:
            continue

        # Compute NNGP weights
        neighbor_coords = site_coords[neighbor_idx]
        K_gN = kernel.K(g_coord, neighbor_coords).ravel()
        K_NN = kernel.K(neighbor_coords, neighbor_coords)
        K_NN += jitter * np.eye(K_NN.shape[0])

        try:
            c, lower = linalg.cho_factor(K_NN)
            a_g = linalg.cho_solve((c, lower), K_gN)
        except linalg.LinAlgError:
            # Fallback: uniform weights
            a_g = np.ones(neighbor_idx.size) / neighbor_idx.size

        rows.extend([g] * neighbor_idx.size)
        cols.extend(neighbor_idx.tolist())
        data.extend(a_g.tolist())

    if len(rows) == 0:
        return sp.csr_matrix((n_grid, n_sites))

    return sp.csr_matrix(
        (np.array(data), (np.array(rows), np.array(cols))),
        shape=(n_grid, n_sites),
    )


def factors_to_csr(factors: NNGPFactors, n_cols: int) -> sp.csr_matrix:
    """Build CSR sparse matrix A whose row i has a_i at columns neighbor_idx[i]."""
    rows = []
    cols = []
    data = []
    n_rows = len(factors.neighbor_idx)
    for i in range(n_rows):
        idx = factors.neighbor_idx[i]
        if idx.size == 0:
            continue
        a = factors.a_rows[i]
        k = idx.size
        rows.extend([i] * k)
        cols.extend(idx.tolist())
        data.extend(a.tolist())
    if len(rows) == 0:
        return sp.csr_matrix((n_rows, n_cols))
    return sp.csr_matrix(
        (np.array(data), (np.array(rows), np.array(cols))), shape=(n_rows, n_cols)
    )


def _interleave_bits_16(x: np.ndarray) -> np.ndarray:
    x = x.astype(np.uint32)
    x &= np.uint32(0xFFFF)
    x = (x | (x << 8)) & np.uint32(0x00FF00FF)
    x = (x | (x << 4)) & np.uint32(0x0F0F0F0F)
    x = (x | (x << 2)) & np.uint32(0x33333333)
    x = (x | (x << 1)) & np.uint32(0x55555555)
    return x


def order_points_morton(coords: np.ndarray, bits: int = 16) -> np.ndarray:
    """Return indices that sort points by Morton (Z-order)."""
    X = _ensure_2d(coords)
    n = X.shape[0]
    if n == 0:
        return np.arange(0)
    minv = X.min(axis=0)
    maxv = X.max(axis=0)
    span = np.maximum(maxv - minv, 1e-12)
    scaled = (X - minv) / span
    maxint = (1 << bits) - 1
    xi = np.clip((scaled[:, 0] * maxint).astype(np.uint32), 0, np.uint32(maxint))
    yi = np.clip((scaled[:, 1] * maxint).astype(np.uint32), 0, np.uint32(maxint))
    xbits = _interleave_bits_16(xi)
    ybits = _interleave_bits_16(yi)
    morton = (xbits << 1) | ybits
    order = np.argsort(morton)
    return order


def log_prior_beta_vecchia(beta_S: np.ndarray, factors_S: NNGPFactors) -> float:
    """Compute -log prior up to a constant for beta under Vecchia on S.
    Sum over coefficients j and locations i:
        0.5 * ( (beta_j[i] - a_i^T beta_j[N_i])^2 / d_i + log d_i )
    (drop 0.5*log(2π) constant).
    """
    beta_S = np.asarray(beta_S, float)
    p_plus_1, nS = beta_S.shape
    d = np.maximum(factors_S.d, 1e-12)
    val = 0.0
    for j in range(p_plus_1):
        bj = beta_S[j, :]
        # loop over locations
        for i in range(nS):
            idx = factors_S.neighbor_idx[i]
            ai = factors_S.a_rows[i]
            mu = float(ai @ bj[idx]) if idx.size else 0.0
            r = bj[i] - mu
            val += 0.5 * (r * r / d[i] + np.log(d[i]))
    return float(val)
