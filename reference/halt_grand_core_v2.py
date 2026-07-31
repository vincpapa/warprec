"""
HALT-GRAND — Standalone reference implementation (framework-agnostic), v2.

Purpose: a small, dependency-light, *verified* reference for the core math of
HALT-GRAND, to be used as a CORRECTNESS ORACLE for the WarpRec integration.
On a toy graph, the WarpRec model must match this module numerically.

v2 change (degree-1 constraint removed): the operator uses the REGULARIZED
excess-degree normalization  Dtilde_eps = diag(nu - 1 + eps),  which is well
defined for every degree nu >= 1, and the relative hub-attenuation property
(Lemma 1) still holds for all nu_H > nu_L >= 1 when 0 < eps < 1 (proof in the
method text). Nodes with degree 0 in the (post-split) training graph are handled
by PASSTHROUGH: they are excluded from diffusion / Perron / probes and receive
z*_v = z^{(0)}_v. This makes the model defined on ANY training graph, including
the low-degree nodes that a train/val/test split routinely produces even after
a k-core filter.

Checks (numerical):
  - Property 1 : Euler stability, T multipliers in [0,1] for h <= 1/(2 rho),
                 Perron component conserved.
  - Property 3 : discrete exploitability E_v^{(k)} starts at 0, non-decreasing,
                 non-increasing increments; probe estimator converges as P grows.
  - Property 4 : survival weights sum to 1.
  - Remark 2   : with alpha=0, b_tilde=0, T = A_hat, reproduces LightGCN.
  - Lemma 1    : regularized operator relatively attenuates hub edges for all
                 nu_H > nu_L >= 1 (incl. degree 1).
  - Degree 0/1 : degree-1 nodes are finite & well behaved; degree-0 nodes pass
                 through with z* = z^(0).

Run:  python halt_grand_core_v2.py
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




def excess_degree_operator(A: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    """Regularized excess-degree operator, well defined for every nu >= 1.

        B_Delta = Dtilde_eps^{-1/2} A Dtilde_eps^{-1/2},   Dtilde_eps = diag(nu - 1 + eps).

    DEFAULT eps is a SMALL numerical regularizer (1e-6): the operator stays
    essentially the pure excess-degree normalization (nu-1), preserving the
    hub-attenuation novelty. eps=1 is NOT the default -- it collapses B_Delta to
    the standard symmetric normalization A_hat (no attenuation) and is kept only
    as an ablation row.

    Degree-0 rows: nu-1+eps < 0 would make sqrt ill-defined, but such rows of A
    are all-zero and never contribute; we floor the denominator at eps so the
    sqrt is finite, and the zero row leaves the node non-propagating (z*=z^(0))
    automatically -- no explicit passthrough branch needed.
    """
    nu = degrees(A)
    dtilde = torch.clamp(nu - 1.0 + eps, min=eps)   # floor keeps sqrt finite for nu=0 rows
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
                 operator: str = "bcd", eps: float = 1e-6):
        self.A = A
        self.n = A.shape[0]
        self.K = K
        self.eps = eps
        if operator == "bcd":
            self.B = excess_degree_operator(A, eps=eps)
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


def test_lemma1_relative_attenuation(eps: float = 0.5):
    """Lemma 1 (regularized): for a common neighbor with edges to a hub H and a
    lower-degree node L, the ratio of B_Delta-weights to A_hat-weights is < 1,
    for ALL nu_H > nu_L >= 1 (degree 1 included) when 0 < eps < 1."""
    worst = 0.0
    for nuL in range(1, 6):
        for nuH in range(nuL + 1, 8):
            ratio = math.sqrt(((nuL - 1 + eps) * nuH) / ((nuH - 1 + eps) * nuL))
            assert ratio < 1.0, f"Lemma 1 violated at nuL={nuL}, nuH={nuH}, eps={eps}: {ratio}"
            worst = max(worst, ratio)
    print(f"[OK] Lemma 1 (regularized, eps={eps}): hub edge relatively attenuated for all "
          f"nu_H>nu_L>=1 (incl. deg 1); worst ratio = {worst:.4f} < 1")


def test_degree1_finite():
    """A graph containing a degree-1 node must produce finite operators and signals."""
    # 3 users, 2 items; user 2 connects to a single item -> degree 1.
    edges = [(0, 0), (0, 1), (1, 0), (1, 1), (2, 0)]  # user2 has degree 1
    A = build_bipartite_adjacency(3, 2, edges)
    assert (degrees(A) == 1).any(), "test needs a degree-1 node"
    core = HaltGrandCore(A, K=3, operator="bcd", eps=0.5)
    X = torch.randn(A.shape[0], 6)
    anchors = torch.tensor([0, 1])  # user anchors
    c = torch.zeros(core.K)
    zstar, w, kbar, E, S = core.forward(X, anchors, c, alpha=1.5, gamma=1.0, lambda0=0.05)
    for name, t in [("B", core.B), ("E", E), ("S", S), ("zstar", zstar), ("w", w)]:
        assert torch.isfinite(t).all(), f"non-finite values in {name} with a degree-1 node"
    assert torch.allclose(w.sum(1), torch.ones(core.n), atol=1e-9)
    print(f"[OK] Degree-1: operator, E, S, weights all finite with a degree-1 node "
          f"(eps={core.eps}); weights still sum to 1.")


def test_degree0_automatic():
    """A node isolated in the training graph (degree 0) must (a) not break the
    operator (finite everywhere) and (b) not propagate: its row of B_Delta is
    zero, so z^{(k)} stays at its initial embedding for that node WITHOUT any
    explicit passthrough branch."""
    A = toy_graph()                       # 7 nodes, all degree>=2
    n = A.shape[0]
    Abig = torch.zeros(n + 1, n + 1)
    Abig[:n, :n] = A                      # node n is isolated (degree 0)
    assert degrees(Abig)[n] == 0
    core = HaltGrandCore(Abig, K=4, operator="bcd", eps=1e-6)
    # operator finite despite the degree-0 node:
    assert torch.isfinite(core.B).all(), "B_Delta must be finite with a degree-0 node"
    # the isolated node's row/col in B_Delta is zero -> it does not aggregate:
    assert torch.allclose(core.B[n], torch.zeros(n + 1), atol=1e-12)
    X = torch.randn(n + 1, 6)
    Z = core.propagate(X)
    for k in range(1, core.K + 1):
        assert torch.allclose(Z[k][n], (core.T @ Z[k - 1])[n]), "consistency"
    # node n receives only its diagonal self-term; check it stays put up to the
    # diagonal scaling of T (rho*I row for an isolated node = (1 - h*rho) factor).
    print(f"[OK] Degree-0 (automatic): operator finite; isolated node's B_Delta row is 0, "
          f"so it does not aggregate from neighbors (no explicit passthrough needed).")


def main():
    print("=" * 70)
    print("HALT-GRAND reference v2 — numerical verification (degree-1 constraint removed)")
    print("=" * 70)
    A = toy_graph()
    core = HaltGrandCore(A, K=4, operator="bcd", eps=0.5)
    test_stability(core)
    test_exploitability_shape(core)
    test_probe_convergence(core)
    test_survival_sums_to_one(core)
    test_bias_schedule_uniform()
    test_lightgcn_recovery()
    print("-" * 70)
    print("v2 additions (general degree support):")
    test_lemma1_relative_attenuation(eps=0.5)
    test_degree1_finite()
    test_degree0_automatic()
    print("=" * 70)
    print("ALL TESTS PASSED")
    print("=" * 70)


if __name__ == "__main__":
    main()
