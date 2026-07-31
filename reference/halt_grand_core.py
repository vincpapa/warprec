"""
HALT-GRAND — Standalone reference implementation (framework-agnostic).

Purpose: a small, dependency-light, *verified* reference for the core math of
HALT-GRAND, to be used as a CORRECTNESS ORACLE for the WarpRec integration.
On a toy graph, the WarpRec model must match this module numerically.

This file is intentionally NOT optimized and NOT a WarpRec model. It implements
the equations of the final ("sound", discrete) method and checks the
propositions numerically:

  - Property 1 : Euler stability, T multipliers in [0,1] for h <= 1/(2 rho),
                 Perron component conserved.
  - Property 3 : discrete exploitability E_v^{(k)} starts at 0, is non-decreasing,
                 has non-increasing increments; probe estimator is unbiased and
                 converges to the exact diagonal as P grows.
  - Property 4 : survival weights sum to 1.
  - Remark 2   : with alpha=0, b_tilde=0, T = A_hat (standard norm), the model
                 reproduces LightGCN's uniform layer combination.

Run:  python halt_grand_core.py
Requires: torch, numpy.
"""

from __future__ import annotations
import math
import torch
import numpy as np

torch.set_default_dtype(torch.float64)  # oracle: prioritize numerical accuracy


# --------------------------------------------------------------------------- #
# Graph utilities
# --------------------------------------------------------------------------- #
def build_bipartite_adjacency(n_users: int, n_items: int, edges: list[tuple[int, int]]) -> torch.Tensor:
    """Symmetric (N x N) adjacency, N = n_users + n_items. Items indexed after users."""
    n = n_users + n_items
    A = torch.zeros(n, n)
    for u, i in edges:
        iu, ii = u, n_users + i
        A[iu, ii] = 1.0
        A[ii, iu] = 1.0
    return A


def degrees(A: torch.Tensor) -> torch.Tensor:
    return A.sum(dim=1)


def excess_degree_operator(A: torch.Tensor) -> torch.Tensor:
    """B_Delta = Dtilde^{-1/2} A Dtilde^{-1/2}, Dtilde = diag(nu - 1). Requires nu >= 2."""
    nu = degrees(A)
    assert torch.all(nu >= 2), "k-core with k>=2 required (excess degree nu-1 must be >= 1)."
    dtilde = nu - 1.0
    dinv = dtilde.pow(-0.5)
    return dinv.unsqueeze(1) * A * dinv.unsqueeze(0)


def standard_norm_adjacency(A: torch.Tensor) -> torch.Tensor:
    """A_hat = D^{-1/2} A D^{-1/2} (LightGCN operator)."""
    nu = degrees(A)
    dinv = nu.pow(-0.5)
    return dinv.unsqueeze(1) * A * dinv.unsqueeze(0)


def perron(Bop: torch.Tensor) -> tuple[float, torch.Tensor]:
    """Top eigenvalue rho and unit-norm positive Perron vector of a symmetric nonneg. op.
    Power iteration on (I + Bop) to make the dominant eigenvalue unique/positive."""
    n = Bop.shape[0]
    M = torch.eye(n) + Bop
    v = torch.ones(n) / math.sqrt(n)
    for _ in range(5000):
        w = M @ v
        w = w / w.norm()
        if (w - v).norm() < 1e-12:
            v = w
            break
        v = w
    # rho from Rayleigh quotient on Bop
    rho = float((v @ (Bop @ v)) / (v @ v))
    if (v < 0).sum() > (v > 0).sum():   # fix sign so Perron vector is positive
        v = -v
    v = v / v.norm()
    return rho, v


# --------------------------------------------------------------------------- #
# HALT-GRAND core
# --------------------------------------------------------------------------- #
class HaltGrandCore:
    def __init__(self, A: torch.Tensor, K: int = 4, h: float | None = None,
                 operator: str = "bcd"):
        self.A = A
        self.n = A.shape[0]
        self.K = K
        if operator == "bcd":
            self.B = excess_degree_operator(A)
        elif operator == "lightgcn":
            self.B = standard_norm_adjacency(A)
        else:
            raise ValueError(operator)
        self.rho, self.phi0 = perron(self.B)
        # Stability bound: h <= 1/(2 rho). Use the strict bound.
        self.h = h if h is not None else 1.0 / (2.0 * self.rho)
        self.T = torch.eye(self.n) - self.h * (self.rho * torch.eye(self.n) - self.B)
        # For the LightGCN special case we want T = A_hat exactly:
        if operator == "lightgcn":
            self.T = self.B  # A_hat itself (h=1, L=I-A_hat)

    # ---- propagation ----
    def propagate(self, X: torch.Tensor) -> list[torch.Tensor]:
        """Return [z^{(0)},...,z^{(K)}]."""
        Z = [X]
        for _ in range(self.K):
            Z.append(self.T @ Z[-1])
        return Z

    def consensus(self, X: torch.Tensor) -> torch.Tensor:
        """mu = phi0 phi0^T X  (recomputed each forward)."""
        return torch.outer(self.phi0, self.phi0) @ X

    # ---- E: exact and probe estimate (graph-only) ----
    def exploitability_exact(self) -> torch.Tensor:
        """E_v^{(k)} exact via [T^{2k}]_vv, shape (n, K+1)."""
        phi0sq = self.phi0 ** 2
        E = torch.zeros(self.n, self.K + 1)
        Tk = torch.eye(self.n)
        diag0 = torch.diagonal(Tk @ Tk)  # k=0 -> T^0 = I, diag = 1
        for k in range(self.K + 1):
            T2k = torch.matrix_power(self.T, 2 * k)
            fk = torch.diagonal(T2k) - phi0sq
            f0 = 1.0 - phi0sq
            E[:, k] = 1.0 - fk / f0
        return E

    def exploitability_probe(self, P: int = 20, seed: int = 0) -> torch.Tensor:
        """Hutchinson estimate of E_v^{(k)}; probes projected off stationary mode."""
        g = torch.Generator().manual_seed(seed)
        Xi = (torch.randint(0, 2, (self.n, P), generator=g).double() * 2 - 1)  # Rademacher
        Q = torch.eye(self.n) - torch.outer(self.phi0, self.phi0)
        Y = Q @ Xi                       # y_p^{(0)}
        phi0sq = self.phi0 ** 2
        f0 = 1.0 - phi0sq
        E = torch.zeros(self.n, self.K + 1)
        for k in range(self.K + 1):
            fk = (Y ** 2).mean(dim=1) if k == 0 else (Yk ** 2).mean(dim=1)  # noqa
            if k == 0:
                Yk = Y.clone()
                fk = (Yk ** 2).mean(dim=1)
            E[:, k] = 1.0 - fk / f0
            Yk = self.T @ Yk
        return E

    # ---- S: specificity on L2-normalized residuals ----
    def specificity(self, Z: list[torch.Tensor], anchors: torch.Tensor,
                    theta: float = 1.0, eps: float = 1e-9) -> torch.Tensor:
        """S_v^{(k)} for all nodes vs a shared anchor set (toy: global anchors).
        Returns (n, K+1). log base = natural; normalized by log(R)."""
        mu = self.consensus(Z[0])
        R = anchors.shape[0]
        S = torch.zeros(self.n, self.K + 1)
        for k in range(self.K + 1):
            r = Z[k] - mu
            r = r / (r.norm(dim=1, keepdim=True) + eps)
            ra = r[anchors]                       # (R, d)
            logits = (r @ ra.T) / theta           # (n, R)
            q = torch.softmax(logits, dim=1)
            Hq = -(q * (q + eps).log()).sum(dim=1)
            S[:, k] = 1.0 - Hq / math.log(R)
        return S

    # ---- controller ----
    def bias_schedule(self, c: torch.Tensor) -> torch.Tensor:
        """b_k = b_uni_k + (c_k - mean(c)); b_uni_k = -log(K-k), k=0..K-1."""
        k = torch.arange(self.K)
        b_uni = -torch.log((self.K - k).double())
        b_tilde = c - c.mean()
        return b_uni + b_tilde

    def survival_weights(self, m: torch.Tensor, b: torch.Tensor, alpha: float) -> torch.Tensor:
        """m: (n, K) increments; b: (K,) bias; returns w: (n, K+1) summing to 1 per row."""
        n = m.shape[0]
        rho_rate = torch.nn.functional.softplus(b.unsqueeze(0) - alpha * m)  # (n,K)
        g = 1.0 - torch.exp(-rho_rate)                                       # (n,K)
        w = torch.zeros(n, self.K + 1)
        surv = torch.ones(n)
        for k in range(self.K):
            w[:, k] = g[:, k] * surv
            surv = surv * (1.0 - g[:, k])
        w[:, self.K] = surv
        return w

    def forward(self, X: torch.Tensor, anchors: torch.Tensor, c: torch.Tensor,
                alpha: float = 1.0, gamma: float = 1.0, lambda0: float = 0.05,
                theta: float = 1.0, E: torch.Tensor | None = None):
        Z = self.propagate(X)
        if E is None:
            E = self.exploitability_exact()
        S = self.specificity(Z, anchors, theta=theta)
        dE = E[:, 1:] - E[:, :-1]                 # (n,K)
        dS = S[:, 1:] - S[:, :-1]                 # (n,K) signed
        m = dE + gamma * dS - lambda0             # (n,K)
        b = self.bias_schedule(c)
        w = self.survival_weights(m, b, alpha)    # (n,K+1)
        Zstack = torch.stack(Z, dim=0)            # (K+1, n, d)
        zstar = (w.T.unsqueeze(-1) * Zstack).sum(dim=0)  # (n,d)
        kbar = (w * torch.arange(self.K + 1).double()).sum(dim=1)
        return zstar, w, kbar, E, S


def lightgcn_reference(A: torch.Tensor, X: torch.Tensor, K: int) -> torch.Tensor:
    """LightGCN: z = mean_{k=0..K} A_hat^k X."""
    Ah = standard_norm_adjacency(A)
    z = X.clone()
    acc = X.clone()
    for _ in range(K):
        z = Ah @ z
        acc = acc + z
    return acc / (K + 1)


# --------------------------------------------------------------------------- #
# Tests
# --------------------------------------------------------------------------- #
def toy_graph():
    # bipartite, every node degree >= 2. 4 users, 3 items.
    edges = [(0, 0), (0, 1), (1, 0), (1, 2), (2, 1), (2, 2), (3, 0), (3, 1), (3, 2)]
    A = build_bipartite_adjacency(4, 3, edges)
    assert torch.all(degrees(A) >= 2)
    return A


def test_stability(core: HaltGrandCore):
    # eigenvalues of T = I - h*(rho I - B) should be in [0,1]
    L = core.rho * torch.eye(core.n) - core.B
    Tm = torch.eye(core.n) - core.h * L
    ev = torch.linalg.eigvalsh(Tm)
    assert ev.min() > -1e-9 and ev.max() < 1 + 1e-9, f"multipliers out of [0,1]: {ev.min()},{ev.max()}"
    # Perron component conserved: T phi0 = phi0
    assert (Tm @ core.phi0 - core.phi0).norm() < 1e-8
    print(f"[OK] Property 1 (stability): T eigenvalues in [{ev.min():.4f}, {ev.max():.4f}], "
          f"Perron conserved. h={core.h:.4f}, rho={core.rho:.4f}")


def test_exploitability_shape(core: HaltGrandCore):
    E = core.exploitability_exact()
    assert torch.allclose(E[:, 0], torch.zeros(core.n), atol=1e-9), "E^{(0)} must be 0"
    dE = E[:, 1:] - E[:, :-1]
    assert (dE > -1e-9).all(), "E must be non-decreasing"
    ddE = dE[:, 1:] - dE[:, :-1]
    assert (ddE < 1e-9).all(), "E increments must be non-increasing (diminishing returns)"
    assert (E[:, -1] <= 1 + 1e-9).all() and (E[:, -1] > E[:, 0]).all()
    print(f"[OK] Property 3 (E shape): E starts at 0, non-decreasing, diminishing returns. "
          f"E[:, -1] mean = {E[:,-1].mean():.4f}")


def test_probe_convergence(core: HaltGrandCore):
    Eex = core.exploitability_exact()
    errs = []
    for P in [10, 50, 200, 1000, 5000]:
        Ep = core.exploitability_probe(P=P, seed=1)
        errs.append(float((Ep - Eex).abs().mean()))
    # error should decrease overall with P
    assert errs[-1] < errs[0], f"probe error did not decrease: {errs}"
    assert errs[-1] < 0.05, f"probe error too high at P=5000: {errs[-1]}"
    print(f"[OK] Property 3 (probe): mean|E_hat - E_exact| over P=[10,50,200,1000,5000] = "
          f"{[round(e,4) for e in errs]}")


def test_survival_sums_to_one(core: HaltGrandCore):
    X = torch.randn(core.n, 8)
    anchors = torch.arange(4)  # user anchors (toy: just use some)
    c = torch.zeros(core.K)
    _, w, kbar, _, _ = core.forward(X, anchors, c, alpha=2.0, gamma=1.0, lambda0=0.05)
    s = w.sum(dim=1)
    assert torch.allclose(s, torch.ones(core.n), atol=1e-9), f"weights sum != 1: {s}"
    assert (w >= -1e-12).all(), "weights must be non-negative"
    print(f"[OK] Property 4 (mixture): survival weights sum to 1 for all nodes "
          f"(max dev {float((s-1).abs().max()):.2e}); mean depth kbar={kbar.mean():.3f}")


def test_lightgcn_recovery():
    A = toy_graph()
    X = torch.randn(A.shape[0], 8)
    K = 4
    # HALT-GRAND in LightGCN mode: operator=lightgcn (T=A_hat), alpha=0, b_tilde=0.
    core = HaltGrandCore(A, K=K, operator="lightgcn")
    anchors = torch.arange(4)
    c = torch.zeros(K)  # b_tilde = 0
    zstar, w, kbar, _, _ = core.forward(X, anchors, c, alpha=0.0, gamma=1.0, lambda0=0.0)
    # With alpha=0 and b_uni_k=-log(K-k), weights should be uniform 1/(K+1)
    w_uniform_target = torch.full((A.shape[0], K + 1), 1.0 / (K + 1))
    assert torch.allclose(w, w_uniform_target, atol=1e-9), \
        f"weights not uniform: {w[0]}"
    # And zstar should equal LightGCN's mean-over-layers
    z_lgcn = lightgcn_reference(A, X, K)
    assert torch.allclose(zstar, z_lgcn, atol=1e-8), \
        f"LightGCN mismatch, max err {float((zstar-z_lgcn).abs().max()):.2e}"
    print(f"[OK] Remark 2 (LightGCN recovery): alpha=0 + b_uni gives uniform 1/(K+1) weights, "
          f"zstar matches LightGCN (max err {float((zstar-z_lgcn).abs().max()):.2e})")


def test_bias_schedule_uniform():
    """b_uni_k = -log(K-k) must induce g^{(k)} = 1/(K+1-k) and uniform weights."""
    A = toy_graph()
    K = 4
    core = HaltGrandCore(A, K=K, operator="lightgcn")
    c = torch.zeros(K)
    b = core.bias_schedule(c)
    g = 1.0 - torch.exp(-torch.nn.functional.softplus(b))  # alpha=0 => rate=softplus(b)
    target = torch.tensor([1.0 / (K + 1 - k) for k in range(K)])
    assert torch.allclose(g, target, atol=1e-9), f"g={g}, target={target}"
    print(f"[OK] Bias schedule: b_uni induces g^(k)=1/(K+1-k) = {[round(float(x),4) for x in g]}")


def main():
    print("=" * 70)
    print("HALT-GRAND reference — numerical verification of the propositions")
    print("=" * 70)
    A = toy_graph()
    core = HaltGrandCore(A, K=4, operator="bcd")
    test_stability(core)
    test_exploitability_shape(core)
    test_probe_convergence(core)
    test_survival_sums_to_one(core)
    test_bias_schedule_uniform()
    test_lightgcn_recovery()
    print("=" * 70)
    print("ALL TESTS PASSED")
    print("=" * 70)


if __name__ == "__main__":
    main()
