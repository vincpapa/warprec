# pylint: disable = R0801, E1102, R0902, R0914, W0201, C0302, R0915
# W0201 (attribute-defined-outside-init): the one-time graph precompute is
# factored into `_precompute_graph_signals` (called from `__init__`), not
# inlined into `__init__`'s own body, for readability and testability; pylint
# doesn't trace attribute assignments through called helper methods.
# C0302 (too-many-lines): the model, its k_bar, dE and dS diagnostics, and
# their docstrings are all kept in this one file rather than split
# arbitrarily across files for a single model class.
# R0915 (too-many-statements): `diagnose_delta_s`/`diagnose_delta_e` are each
# one linear diagnostic routine (build signal, print representative-node
# table, print aggregate stats/comparisons); splitting them further would
# hurt readability more than the statement count helps.
import csv
import math
import os
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from scipy.stats import spearmanr
from torch import nn, Tensor
from torch_sparse import SparseTensor, matmul as sparse_matmul

from warprec.data.entities import Interactions, Sessions
from warprec.recommenders.base_recommender import IterativeRecommender
from warprec.recommenders.collaborative_filtering_recommender.graph_based import (
    GraphRecommenderUtils,
)
from warprec.recommenders.losses import BPRLoss, EmbLoss
from warprec.utils.enums import DataLoaderType
from warprec.utils.logger import logger
from warprec.utils.registry import model_registry


@model_registry.register(name="HALTGRAND")
class HALTGRAND(GraphRecommenderUtils, IterativeRecommender):
    """Implementation of HALT-GRAND, a generalization of LightGCN that aggregates
    propagation layers with per-node adaptive (survival-based) weights instead of a
    uniform average.

    The diffusion operator uses a regularized excess-degree normalization
    (``Dtilde = clamp(nu - 1 + eps_delta, min=eps_delta)``), well defined for every
    node degree nu >= 0 -- no k-core filtering is required, since train/val/test
    splitting routinely produces degree-0/1 nodes even after a k-core filter on
    the full dataset. A degree-0 node has an all-zero adjacency row, hence an
    all-zero row in the diffusion operator, so it simply does not aggregate from
    neighbors (no explicit passthrough branch needed). ``eps_delta`` is itself an
    explored hyperparameter: it changes ``b_delta``, hence ``rho``, hence the
    stability bound on ``h`` -- both are always recomputed together from the
    current ``eps_delta``, per model instance (e.g. per hyperparameter-search
    trial), never reused across a different ``eps_delta`` value.

    Args:
        params (dict): Model parameters.
        info (dict): The dictionary containing dataset information.
        interactions (Interactions): The training interactions.
        *args (Any): Variable length argument list.
        seed (int): The seed to use for reproducibility.
        **kwargs (Any): Arbitrary keyword arguments.

    Attributes:
        DATALOADER_TYPE: The type of dataloader used.
        embedding_size (int): The embedding size of user and item.
        n_layers (int): The maximum propagation depth (K).
        diffusion_step (float): The Euler step h. If <= 0, it is automatically set to
            the stability bound 1/(2*rho). If > 0, it must satisfy h <= 1/(2*rho).
        eps_delta (float): The regularization constant of the excess-degree
            normalization, in (0, 1]. Close to 0 (e.g. 1e-6) stays close to the
            pure excess-degree operator (nu - 1, the hub-attenuation novelty); 1
            collapses it to the standard symmetric normalization (LightGCN's
            operator), an ablation setting rather than the default.
        n_probes (int): The number of Rademacher probes (P) used to estimate the
            latent exploitability signal.
        n_anchors (int): The number of degree-stratified anchors (R) per node type.
        temperature (float): The softmax temperature (theta) used in the specificity
            signal.
        gamma (float): The weight of the specificity variation in the controller.
            0 disables specificity's contribution (a valid ablation setting);
            negative values are rejected at the config layer.
        lambda0 (float): The fixed per-layer price in the controller's marginal
            utility.
        lambda_ponder (float): The weight of the ponder loss.
        lambda_unif (float): The weight of the uniformity loss.
        reg_weight (float): The L2 regularization weight on the free embeddings.
        alpha_init (float): The initial value of the pre-softplus alpha parameter.
        batch_size (int): The batch size used for training.
        epochs (int): The number of epochs.
        learning_rate (float): The learning rate value.

    Note:
        The one-time graph precompute (see ``_precompute_graph_signals``) also
        sets, as plain instance attributes/buffers rather than class-level
        annotations (so they are never treated as configurable hyperparameters
        or round-tripped through checkpoint ``params``): ``b_delta`` (the
        excess-degree normalized diffusion operator), ``rho`` (its spectral
        radius), ``h`` (the stable Euler step), ``phi0`` (the frozen Perron
        vector buffer), ``user_anchor_idx``/``item_anchor_idx`` (the frozen
        anchor node indices buffers), ``e_hat`` (the frozen, probe-based
        exploitability signal buffer), and ``degree`` (the frozen raw node
        degree buffer, used only for diagnostics).
    """

    DATALOADER_TYPE = DataLoaderType.POS_NEG_LOADER

    # Internal constants (not exposed as searchable hyperparameters: not annotated,
    # so `Recommender.init_params` will never assign them from the config).
    _EPS = 1e-9
    _PROBE_SEED = 0
    _ANCHOR_SEED = 1000
    _POWER_MAX_ITER = 1000
    _POWER_TOL = 1e-10
    _KBAR_LOG_EVERY_N_EPOCHS = 20

    # Model hyperparameters
    embedding_size: int
    n_layers: int
    diffusion_step: float
    eps_delta: float
    n_probes: int
    n_anchors: int
    temperature: float
    gamma: float
    lambda0: float
    lambda_ponder: float
    lambda_unif: float
    reg_weight: float
    alpha_init: float
    batch_size: int
    epochs: int
    learning_rate: float

    def __init__(
        self,
        params: dict,
        info: dict,
        interactions: Interactions,
        *args: Any,
        seed: int = 42,
        **kwargs: Any,
    ):
        super().__init__(params, info, *args, seed=seed, **kwargs)

        self.user_embedding = nn.Embedding(self.n_users, self.embedding_size)
        self.item_embedding = nn.Embedding(
            self.n_items + 1, self.embedding_size, padding_idx=self.n_items
        )

        # Learned controller parameters: mean-zero bias residual (K values) and the
        # pre-softplus hazard sensitivity alpha_hat (alpha = softplus(alpha_hat) >= 0).
        self.c_bias = nn.Parameter(torch.zeros(self.n_layers))
        self.alpha_hat = nn.Parameter(torch.tensor(float(self.alpha_init)))

        self.apply(self._init_weights)
        self.bpr_loss = BPRLoss()
        self.reg_loss = EmbLoss()

        # Graph-only signals (B_delta, rho, phi0, anchors, E_hat) are precomputed
        # once here and reused as-is at every epoch and at inference.
        self._precompute_calls = 0
        self._precompute_graph_signals(interactions)

        # Bookkeeping for the read-only k_bar diagnostic (see
        # `_log_kbar_diagnostics`): the epoch last logged (to avoid a duplicate
        # entry when the final epoch also happens to be a logging checkpoint)
        # and the resolved CSV path (cached so every row lands in the same file).
        self._last_kbar_log_epoch: int = -1
        self._kbar_csv_path: Optional[str] = None

    # ------------------------------------------------------------------ #
    # One-time precompute (graph-only, does not depend on the embeddings)
    # ------------------------------------------------------------------ #
    def _precompute_graph_signals(self, interactions: Interactions):
        """Precompute the diffusion operator, the Perron pair, the frozen anchor
        pools and the probe-based exploitability signal. Called exactly once,
        from ``__init__``.

        Args:
            interactions (Interactions): The training interactions, used to build
                the bipartite user-item graph.

        Raises:
            ValueError: If ``diffusion_step`` is set to a strictly positive value
                that violates the stability bound ``h <= 1/(2*rho)`` for the
                current ``eps_delta``.
        """
        self._precompute_calls += 1

        n_users, n_items = self.n_users, self.n_items
        n_nodes = n_users + n_items

        coo = interactions.get_sparse().tocoo()
        user_nodes = coo.row
        item_nodes = coo.col + n_users
        row = np.concatenate([user_nodes, item_nodes])
        col = np.concatenate([item_nodes, user_nodes])
        edge_index = torch.tensor(np.vstack([row, col]), dtype=torch.int64)
        adj = SparseTensor(
            row=edge_index[0],
            col=edge_index[1],
            sparse_sizes=(n_nodes, n_nodes),
        )

        degree = adj.sum(dim=1)
        self.degree: Tensor
        self.register_buffer("degree", degree)

        # Regularized excess-degree normalization, well defined for every node
        # degree nu >= 0 (no k-core filtering is required). A degree-0 node has
        # an all-zero row in `adj`, hence an all-zero row in `b_delta` regardless
        # of the denominator used there -- it simply does not aggregate from
        # neighbors (only the self-scaling term in `_apply_diffusion` acts on
        # it), no explicit passthrough branch needed. The clamp only exists to
        # keep the sqrt finite; eps_delta -> 0 stays close to the pure
        # excess-degree operator (nu - 1, the hub-attenuation novelty), while
        # eps_delta = 1 collapses Dtilde to nu exactly, i.e. the standard
        # symmetric normalization used by LightGCN (an ablation, not the
        # default).
        dtilde = torch.clamp(degree - 1.0 + self.eps_delta, min=self.eps_delta)
        dtilde_inv_sqrt = dtilde.pow(-0.5)
        r, c, _ = adj.coo()
        b_values = dtilde_inv_sqrt[r] * dtilde_inv_sqrt[c]
        b_delta = adj.set_value(b_values, layout="coo")
        self.b_delta = b_delta

        # rho (and therefore h's stability bound) depends on eps_delta through
        # b_delta above, so both are always recomputed together from the
        # CURRENT eps_delta -- never reused across a different eps_delta value.
        rho, phi0 = self._power_iteration(b_delta, n_nodes)
        self.rho = rho

        if self.diffusion_step is not None and self.diffusion_step > 0:
            h = float(self.diffusion_step)
            h_max = 1.0 / (2.0 * rho)
            if h > h_max + 1e-9:
                raise ValueError(
                    f"diffusion_step={h} violates the stability bound "
                    f"h <= 1/(2*rho)={h_max:.6f} (rho={rho:.6f})."
                )
        else:
            h = 1.0 / (2.0 * rho)
        self.h = h

        residual = torch.norm(
            sparse_matmul(b_delta, phi0.unsqueeze(1)).squeeze(1) - rho * phi0
        ).item()
        logger.attention(
            f"HALTGRAND: eps_delta={self.eps_delta:.6g}, rho={rho:.6f}, "
            f"h={self.h:.6f}, ||B_delta phi0 - rho phi0||={residual:.3e}"
        )

        # Instance-level (not class-body) annotations: these are only understood
        # by mypy (nn.Module.__getattr__ otherwise types buffer access as
        # `Tensor | Module`) and are never added to `cls.__annotations__`, unlike
        # class-body annotations. That distinction matters here: `Recommender`'s
        # `get_params`/`init_params` iterate `self.__class__.__annotations__` to
        # decide what counts as a configurable hyperparameter, and a class-body
        # annotation for a buffer name would make it round-trip through
        # checkpoint save/load as a fake "hyperparameter", clashing with the
        # actual `register_buffer` call below.
        self.phi0: Tensor
        self.register_buffer("phi0", phi0)

        user_anchor_idx = self._sample_degree_stratified_anchors(
            degree[:n_users], offset=0, seed=self._ANCHOR_SEED
        )
        item_anchor_idx = self._sample_degree_stratified_anchors(
            degree[n_users:], offset=n_users, seed=self._ANCHOR_SEED + 1
        )
        self.user_anchor_idx: Tensor
        self.item_anchor_idx: Tensor
        self.register_buffer("user_anchor_idx", user_anchor_idx)
        self.register_buffer("item_anchor_idx", item_anchor_idx)

        e_hat = self._estimate_exploitability(b_delta, phi0, h, rho, n_nodes)
        self.e_hat: Tensor
        self.register_buffer("e_hat", e_hat)

    def _power_iteration(
        self, b_delta: SparseTensor, n_nodes: int
    ) -> Tuple[float, Tensor]:
        """Estimate the spectral radius ``rho`` and the Perron vector ``phi0`` of
        ``b_delta`` via power iteration on ``I + b_delta``.

        Args:
            b_delta (SparseTensor): The excess-degree normalized diffusion operator.
            n_nodes (int): The total number of graph nodes (users + items).

        Returns:
            Tuple[float, Tensor]: The spectral radius and the unit-norm, positive
                Perron vector.
        """
        v = torch.ones(n_nodes, 1) / math.sqrt(n_nodes)
        for _ in range(self._POWER_MAX_ITER):
            w = v + sparse_matmul(b_delta, v)
            w = w / (w.norm() + self._EPS)
            if torch.norm(w - v) < self._POWER_TOL:
                v = w
                break
            v = w

        v = v.squeeze(1)
        b_v = sparse_matmul(b_delta, v.unsqueeze(1)).squeeze(1)
        rho = float((v @ b_v) / (v @ v))

        if (v < 0).sum() > (v > 0).sum():
            v = -v
        v = v / (v.norm() + self._EPS)
        return rho, v

    def _sample_degree_stratified_anchors(
        self, degrees: Tensor, offset: int, seed: int
    ) -> Tensor:
        """Sample ``n_anchors`` degree-stratified anchor node indices of one type.

        Nodes are ranked by degree and split into ``n_anchors`` contiguous strata;
        one node is drawn uniformly at random from each stratum, so the anchor pool
        spans the whole degree distribution instead of only high-degree hubs.

        Args:
            degrees (Tensor): The degree of every node of this type.
            offset (int): The global node-index offset (0 for users, n_users for
                items).
            seed (int): The fixed seed used to freeze the anchor sample.

        Returns:
            Tensor: The (n_anchors,) tensor of global anchor node indices.
        """
        n_type_nodes = degrees.numel()
        n_anchors = min(self.n_anchors, n_type_nodes)
        order = torch.argsort(degrees)
        bin_edges = torch.linspace(0, n_type_nodes, n_anchors + 1).round().long()
        generator = torch.Generator().manual_seed(seed)

        chosen = []
        for i in range(n_anchors):
            lo, hi = int(bin_edges[i]), int(bin_edges[i + 1])
            hi = max(hi, lo + 1)
            bin_nodes = order[lo:hi]
            pick = bin_nodes[
                torch.randint(0, bin_nodes.numel(), (1,), generator=generator)
            ]
            chosen.append(pick)
        return (torch.cat(chosen) + offset).long()

    def _estimate_exploitability(
        self,
        b_delta: SparseTensor,
        phi0: Tensor,
        h: float,
        rho: float,
        n_nodes: int,
    ) -> Tensor:
        """Estimate the latent exploitability ``E_hat_v^{(k)}`` for every node and
        depth ``k=0..n_layers`` via ``n_probes`` Rademacher probes projected off the
        stationary mode.

        Args:
            b_delta (SparseTensor): The excess-degree normalized diffusion operator.
            phi0 (Tensor): The Perron vector.
            h (float): The Euler step.
            rho (float): The spectral radius.
            n_nodes (int): The total number of graph nodes.

        Returns:
            Tensor: The (n_nodes, n_layers + 1) exploitability estimate.
        """
        generator = torch.Generator().manual_seed(self._PROBE_SEED)
        xi = (
            torch.randint(0, 2, (n_nodes, self.n_probes), generator=generator).float()
            * 2
            - 1
        )
        coeffs = phi0 @ xi
        y_k = xi - torch.outer(phi0, coeffs)

        phi0_sq = phi0.pow(2)
        f0 = (1.0 - phi0_sq).clamp(min=self._EPS)

        e_hat = torch.zeros(n_nodes, self.n_layers + 1)
        for k in range(self.n_layers + 1):
            f_k = (y_k**2).mean(dim=1)
            e_hat[:, k] = 1.0 - f_k / f0
            if k < self.n_layers:
                y_k = (1.0 - h * rho) * y_k + h * sparse_matmul(b_delta, y_k)
        return e_hat

    # ------------------------------------------------------------------ #
    # Diffusion / propagation
    # ------------------------------------------------------------------ #
    def _apply_diffusion(self, x: Tensor) -> Tensor:
        """Apply the Euler-stabilized diffusion operator ``T = I - h*(rho*I - B_delta)``.

        Args:
            x (Tensor): The (n_nodes, d) node representations.

        Returns:
            Tensor: The (n_nodes, d) representations after one diffusion step.
        """
        return (1.0 - self.h * self.rho) * x + self.h * sparse_matmul(self.b_delta, x)

    def _ego_embeddings(self) -> Tensor:
        """Build the initial (layer-0) node representations, users then items,
        excluding the item embedding table's padding row.

        Returns:
            Tensor: The (n_users + n_items, embedding_size) tensor.
        """
        user_emb = self.user_embedding.weight
        item_emb = self.item_embedding.weight[: self.n_items]
        return torch.cat([user_emb, item_emb], dim=0)

    def _propagate(self, x: Tensor) -> List[Tensor]:
        """Propagate ``x`` through ``n_layers`` diffusion steps.

        Args:
            x (Tensor): The (n_nodes, d) layer-0 representations.

        Returns:
            List[Tensor]: ``[z^{(0)}, ..., z^{(K)}]``, each of shape (n_nodes, d).
        """
        if self.b_delta.device() != x.device:
            self.b_delta = self.b_delta.to(x.device)

        z_list = [x]
        for _ in range(self.n_layers):
            z_list.append(self._apply_diffusion(z_list[-1]))
        return z_list

    def _consensus(self, x: Tensor) -> Tensor:
        """Compute the stationary-mode projection ``coeff = phi0^T X``, recomputed
        every forward pass since it depends on the current embeddings ``X``. The
        full consensus is ``mu = phi0 (phi0^T X) = phi0 * coeff`` (an outer
        product), but only ``coeff`` (a single (d,) vector) is returned here:
        materializing the full (n_nodes, d) ``mu`` is wasteful whenever a caller
        only reads it at a handful of node indices (see ``_mu_at``) -- reducing
        over every node's row to get ``coeff`` is unavoidable (mu depends on
        every node's current embedding), but broadcasting it back out to every
        row is not, when only a subset of rows is ever read.

        Args:
            x (Tensor): The (n_nodes, d) layer-0 representations.

        Returns:
            Tensor: The (d,) stationary-mode projection.
        """
        return self.phi0 @ x

    def _mu_at(self, node_idx: Tensor, coeff: Tensor) -> Tensor:
        """Compute the consensus ``mu = phi0 * coeff`` at specific node indices
        only, without materializing the full (n_nodes, d) tensor.

        Args:
            node_idx (Tensor): The global node indices to compute mu for.
            coeff (Tensor): The (d,) stationary-mode projection from
                ``_consensus``.

        Returns:
            Tensor: The (len(node_idx), d) consensus tensor.
        """
        return self.phi0[node_idx].unsqueeze(1) * coeff.unsqueeze(0)

    # ------------------------------------------------------------------ #
    # Specificity S and the survival controller (batch/node-scoped)
    # ------------------------------------------------------------------ #
    def _specificity(
        self, z_list: List[Tensor], coeff: Tensor, node_idx: Tensor
    ) -> Tensor:
        """Compute the latent specificity ``S_v^{(k)}`` for the requested nodes only,
        against the frozen, degree-stratified, same-type anchor pool. A node that is
        itself one of the anchors is masked out of its own softmax.

        Args:
            z_list (List[Tensor]): The ``[z^{(0)}, ..., z^{(K)}]`` propagated
                representations (full graph).
            coeff (Tensor): The (d,) stationary-mode projection from
                ``_consensus``, used to compute ``mu`` only where needed.
            node_idx (Tensor): The global node indices to compute S for.

        Returns:
            Tensor: The (len(node_idx), n_layers + 1) specificity tensor.
        """
        is_user = node_idx < self.n_users
        s = torch.zeros(node_idx.numel(), self.n_layers + 1, device=node_idx.device)
        log_r = math.log(self.n_anchors)

        # Gather only the rows this call actually needs (batch nodes + both anchor
        # pools) before normalizing, instead of normalizing the full (n_nodes, d)
        # tensor every layer: for a real graph, node_idx is a small batch-derived
        # set, so this keeps the cost O(batch + anchors) instead of O(n_nodes).
        n_active = node_idx.numel()
        n_user_anchors = self.user_anchor_idx.numel()
        all_needed = torch.cat([node_idx, self.user_anchor_idx, self.item_anchor_idx])
        unique_needed, inverse = torch.unique(all_needed, return_inverse=True)
        node_pos = inverse[:n_active]
        user_anchor_pos = inverse[n_active : n_active + n_user_anchors]
        item_anchor_pos = inverse[n_active + n_user_anchors :]
        mu_needed = self._mu_at(unique_needed, coeff)

        for k in range(self.n_layers + 1):
            residual = z_list[k][unique_needed] - mu_needed
            r_hat = residual / (residual.norm(dim=1, keepdim=True) + self._EPS)

            for type_mask, anchor_pos, anchor_idx in (
                (is_user, user_anchor_pos, self.user_anchor_idx),
                (~is_user, item_anchor_pos, self.item_anchor_idx),
            ):
                if not torch.any(type_mask):
                    continue
                type_nodes = node_idx[type_mask]
                r_v = r_hat[node_pos[type_mask]]
                r_c = r_hat[anchor_pos]

                logits = (r_v @ r_c.T) / self.temperature
                self_mask = type_nodes.unsqueeze(1) == anchor_idx.unsqueeze(0)
                logits = logits.masked_fill(self_mask, float("-inf"))

                q = torch.softmax(logits, dim=1)
                entropy = -(q * (q + self._EPS).log()).sum(dim=1)
                s[type_mask, k] = 1.0 - entropy / log_r

        return s

    def _bias_schedule(self) -> Tensor:
        """Compute the per-layer hazard bias: a fixed uniform prior plus a
        learned, mean-zero residual.

        ``b_uni[k] = -log(K - k)`` for ``k=0..K-1`` is a fixed (non-learned)
        prior; since ``k`` never reaches ``K``, its argument ``K - k`` never
        reaches 0 (it bottoms out at 1, at ``k=K-1``, where ``b_uni=-log(1)=0``)
        -- ``log(0)`` is structurally unreachable here, and the ``clamp`` is
        only a defensive guard on top of that. ``b_tilde[k] = c[k] - mean(c)``
        is a learned, mean-zero residual (``c`` is ``self.c_bias``), so it can
        only reshape the schedule around ``b_uni``, never shift it uniformly.

        Returns:
            Tensor: The (n_layers,) bias schedule ``b = b_uni + b_tilde``.
        """
        k_arange = torch.arange(
            self.n_layers, dtype=self.c_bias.dtype, device=self.c_bias.device
        )
        b_uni = -torch.log((self.n_layers - k_arange).clamp(min=self._EPS))
        b_tilde = self.c_bias - self.c_bias.mean()
        return b_uni + b_tilde

    def _survival_controller(
        self, e_active: Tensor, s_active: Tensor
    ) -> Tuple[Tensor, Tensor]:
        """Run the survival (hazard) controller for a set of nodes.

        Args:
            e_active (Tensor): The (n_active, n_layers + 1) exploitability signal.
            s_active (Tensor): The (n_active, n_layers + 1) specificity signal.

        Returns:
            Tuple[Tensor, Tensor]: The (n_active, n_layers + 1) survival weights
                (summing to 1 per row) and the (n_active,) expected depth.
        """
        d_e = e_active[:, 1:] - e_active[:, :-1]
        d_s = s_active[:, 1:] - s_active[:, :-1]
        m = d_e + self.gamma * d_s - self.lambda0

        b = self._bias_schedule()

        alpha = F.softplus(self.alpha_hat)
        rho_rate = F.softplus(b.unsqueeze(0) - alpha * m)
        g = 1.0 - torch.exp(-rho_rate)

        n_active = m.size(0)
        w = torch.zeros(n_active, self.n_layers + 1, device=m.device, dtype=m.dtype)
        surv = torch.ones(n_active, device=m.device, dtype=m.dtype)
        for k in range(self.n_layers):
            w[:, k] = g[:, k] * surv
            surv = surv * (1.0 - g[:, k])
        w[:, self.n_layers] = surv

        depth = torch.arange(self.n_layers + 1, dtype=m.dtype, device=m.device)
        k_bar = (w * depth).sum(dim=1)
        return w, k_bar

    def _mix(
        self, z_list: List[Tensor], coeff: Tensor, node_idx: Tensor
    ) -> Tuple[Tensor, Tensor]:
        """Compute the final survival-weighted representation ``z*`` and the
        expected depth ``k_bar`` for the requested nodes.

        Args:
            z_list (List[Tensor]): The ``[z^{(0)}, ..., z^{(K)}]`` propagated
                representations (full graph).
            coeff (Tensor): The (d,) stationary-mode projection from
                ``_consensus``.
            node_idx (Tensor): The global node indices to compute z* for.

        Returns:
            Tuple[Tensor, Tensor]: The (len(node_idx), d) final representations and
                the (len(node_idx),) expected depth.
        """
        s_active = self._specificity(z_list, coeff, node_idx)
        e_active = self.e_hat[node_idx]
        w, k_bar = self._survival_controller(e_active, s_active)

        # Gather node_idx out of each (n_nodes, d) layer BEFORE stacking, instead of
        # stacking the full (K+1, n_nodes, d) tensor and slicing afterwards.
        z_stack = torch.stack([z[node_idx] for z in z_list], dim=0)
        z_star = (w.T.unsqueeze(-1) * z_stack).sum(dim=0)
        return z_star, k_bar

    # ------------------------------------------------------------------ #
    # Public model interface
    # ------------------------------------------------------------------ #
    def forward(self) -> Tuple[Tensor, Tensor, Tensor]:
        """Full-graph forward pass, used for inference (via
        ``GraphRecommenderUtils.propagate_embeddings``).

        Returns:
            Tuple[Tensor, Tensor, Tensor]: User embeddings, item embeddings
                (including the zero padding row), and the expected depth of every
                node.
        """
        x = self._ego_embeddings()
        z_list = self._propagate(x)
        coeff = self._consensus(z_list[0])

        all_idx = torch.arange(self.n_users + self.n_items, device=x.device)
        z_star, k_bar = self._mix(z_list, coeff, all_idx)

        user_all_embeddings = z_star[: self.n_users]
        item_all_embeddings = z_star[self.n_users :]
        item_all_embeddings = torch.cat(
            [item_all_embeddings, item_all_embeddings.new_zeros(1, z_star.size(1))],
            dim=0,
        )
        return user_all_embeddings, item_all_embeddings, k_bar

    def _encode_batch(
        self, user: Tensor, pos_item: Tensor, neg_item: Tensor
    ) -> Tuple[Tensor, Tensor, Tensor, Tensor, Tensor, Tensor]:
        """Training-time forward pass: the diffusion propagation is always run on
        the full graph (message passing requires it), but the specificity signal
        and the survival controller are only evaluated for the nodes touched by
        this batch.

        Args:
            user (Tensor): The batch of user indices.
            pos_item (Tensor): The batch of positive item indices.
            neg_item (Tensor): The batch of negative item indices.

        Returns:
            Tuple[Tensor, Tensor, Tensor, Tensor, Tensor, Tensor]: The z* embeddings
                for user/pos_item/neg_item, the expected depth of the batch's unique
                nodes, and the raw (layer-0) embeddings of the batch's unique users
                and unique items.
        """
        x = self._ego_embeddings()
        z_list = self._propagate(x)
        coeff = self._consensus(z_list[0])

        n_batch = user.numel()
        all_ids = torch.cat([user, pos_item + self.n_users, neg_item + self.n_users])
        active_idx, inverse = torch.unique(all_ids, return_inverse=True)

        z_star_active, k_bar_active = self._mix(z_list, coeff, active_idx)

        z_user = z_star_active[inverse[:n_batch]]
        z_pos = z_star_active[inverse[n_batch : 2 * n_batch]]
        z_neg = z_star_active[inverse[2 * n_batch :]]

        unique_users = torch.unique(user)
        unique_items = torch.unique(torch.cat([pos_item, neg_item]))
        z0_user = z_list[0][unique_users]
        z0_item = z_list[0][unique_items + self.n_users]

        return z_user, z_pos, z_neg, k_bar_active, z0_user, z0_item

    def get_dataloader(
        self,
        interactions: Interactions,
        sessions: Sessions,
        **kwargs: Any,
    ):
        return interactions.get_contrastive_dataloader(
            batch_size=self.batch_size,
            **kwargs,
        )

    def _uniformity_loss(self, z: Tensor) -> Tensor:
        """Wang-Isola uniformity loss on L2-normalized embeddings of a single node
        type.

        Args:
            z (Tensor): The (n, d) raw (layer-0) embeddings of one node type.

        Returns:
            Tensor: The scalar uniformity loss. Returns 0 if fewer than 2 embeddings
                are provided (no off-diagonal pair to compute).
        """
        n = z.size(0)
        if n < 2:
            return z.new_zeros(())

        z_hat = F.normalize(z, p=2, dim=1)
        sq_dist = torch.cdist(z_hat, z_hat, p=2).pow(2)
        off_diag = ~torch.eye(n, dtype=torch.bool, device=z.device)
        return torch.log(torch.exp(-2.0 * sq_dist[off_diag]).mean() + self._EPS)

    def training_step(self, batch: Any, batch_idx: int):
        user, pos_item, neg_item = batch

        z_user, z_pos, z_neg, k_bar_active, z0_user, z0_item = self._encode_batch(
            user, pos_item, neg_item
        )

        pos_scores = torch.mul(z_user, z_pos).sum(dim=1)
        neg_scores = torch.mul(z_user, z_neg).sum(dim=1)
        bpr_loss = self.bpr_loss(pos_scores, neg_scores)

        reg_loss = self.reg_weight * self.reg_loss(
            self.user_embedding(user),
            self.item_embedding(pos_item),
            self.item_embedding(neg_item),
        )

        ponder_loss = k_bar_active.mean()
        unif_loss = 0.5 * (
            self._uniformity_loss(z0_user) + self._uniformity_loss(z0_item)
        )

        loss = (
            bpr_loss
            + reg_loss
            + self.lambda_ponder * ponder_loss
            + self.lambda_unif * unif_loss
        )
        self.log("loss", loss, prog_bar=True, on_step=False, on_epoch=True)
        return loss

    def predict(
        self,
        user_indices: Tensor,
        *args: Any,
        item_indices: Optional[Tensor] = None,
        **kwargs: Any,
    ) -> Tensor:
        """Prediction using the survival-weighted final embeddings.

        Args:
            user_indices (Tensor): The batch of user indices.
            *args (Any): List of arguments.
            item_indices (Optional[Tensor]): The batch of item indices. If None,
                full prediction will be produced.
            **kwargs (Any): The dictionary of keyword arguments.

        Returns:
            Tensor: The score matrix {user x item}.
        """
        user_all_embeddings, item_all_embeddings = self.propagate_embeddings()

        user_embeddings = user_all_embeddings[user_indices]

        if item_indices is None:
            item_embeddings = item_all_embeddings[:-1, :]
            einsum_string = "be,ie->bi"
        else:
            item_embeddings = item_all_embeddings[item_indices]
            einsum_string = "be,bse->bs"

        return torch.einsum(einsum_string, user_embeddings, item_embeddings)

    # ------------------------------------------------------------------ #
    # Read-only k_bar (expected propagation depth) diagnostic, logged every
    # `_KBAR_LOG_EVERY_N_EPOCHS` epochs and once more at the end of training.
    # Never touches gradients or parameters, and has no effect on training.
    # ------------------------------------------------------------------ #
    _KBAR_CSV_FIELDNAMES = [
        "epoch",
        "node_type",
        "mean",
        "std",
        "min",
        "max",
        "p10",
        "p50",
        "p90",
        "spearman_corr_with_degree",
        "spearman_pvalue",
    ]

    def _kbar_stats(self, values: Tensor) -> Dict[str, float]:
        """Compute summary statistics of a 1-D tensor of k_bar values.

        Args:
            values (Tensor): The (n,) tensor of per-node expected depths.

        Returns:
            Dict[str, float]: mean, std, min, max, and the 10th/50th/90th
                percentiles.
        """
        quantiles = torch.quantile(
            values, torch.tensor([0.1, 0.5, 0.9], dtype=values.dtype)
        )
        return {
            "mean": values.mean().item(),
            "std": values.std().item(),
            "min": values.min().item(),
            "max": values.max().item(),
            "p10": quantiles[0].item(),
            "p50": quantiles[1].item(),
            "p90": quantiles[2].item(),
        }

    def _append_kbar_csv(self, rows: List[Dict[str, Any]]) -> None:
        """Best-effort append of k_bar diagnostic rows to a per-run CSV file.
        Never raises: a write failure is logged and otherwise swallowed, since
        this diagnostic must never be able to interrupt training.

        Args:
            rows (List[Dict[str, Any]]): One dict per node type, with keys
                matching ``_KBAR_CSV_FIELDNAMES``.
        """
        if self._kbar_csv_path is None:
            self._kbar_csv_path = f"haltgrand_kbar_diagnostics_{self.name_param}.csv"

        try:
            write_header = not os.path.exists(self._kbar_csv_path)
            with open(
                self._kbar_csv_path, "a", newline="", encoding="utf-8"
            ) as csv_file:
                writer = csv.DictWriter(csv_file, fieldnames=self._KBAR_CSV_FIELDNAMES)
                if write_header:
                    writer.writeheader()
                writer.writerows(rows)
            if write_header:
                logger.attention(
                    "HALTGRAND: writing k_bar diagnostics to "
                    f"{os.path.abspath(self._kbar_csv_path)}"
                )
        except OSError as error:
            logger.attention(
                f"HALTGRAND: could not write k_bar diagnostics CSV: {error}"
            )

    def _log_kbar_diagnostics(self, epoch: int) -> None:
        """Log the distribution of the expected propagation depth ``k_bar_v``
        (Eq. 23, the same quantity the ponder loss already averages per
        minibatch), computed via a full-graph, no-grad forward pass, separately
        for users and items, together with its Spearman correlation against
        raw node degree.

        Args:
            epoch (int): The (1-indexed, human-facing) epoch number to
                log/record. If equal to the last epoch already logged, this
                is a no-op (guards against double-logging when the final
                epoch also happens to be a periodic logging checkpoint).
        """
        if epoch == self._last_kbar_log_epoch:
            return
        self._last_kbar_log_epoch = epoch

        with torch.no_grad():
            _, _, k_bar = self.forward()

        k_bar = k_bar.detach()
        degree = self.degree.detach()

        rows: List[Dict[str, Any]] = []
        for node_type, node_slice in (
            ("user", slice(0, self.n_users)),
            ("item", slice(self.n_users, None)),
        ):
            k_bar_t = k_bar[node_slice].cpu()
            degree_t = degree[node_slice].cpu()

            stats = self._kbar_stats(k_bar_t)
            corr, pvalue = spearmanr(k_bar_t.numpy(), degree_t.numpy())
            stats["spearman_corr_with_degree"] = float(corr)
            stats["spearman_pvalue"] = float(pvalue)

            logger.attention(
                f"HALTGRAND k_bar diagnostics [{node_type}, epoch {epoch}]: "
                f"mean={stats['mean']:.4f} std={stats['std']:.4f} "
                f"min={stats['min']:.4f} max={stats['max']:.4f} "
                f"p10={stats['p10']:.4f} p50={stats['p50']:.4f} "
                f"p90={stats['p90']:.4f} "
                f"spearman(k_bar, degree)={stats['spearman_corr_with_degree']:.4f} "
                f"(p={stats['spearman_pvalue']:.3g})"
            )
            rows.append({"epoch": epoch, "node_type": node_type, **stats})

        self._append_kbar_csv(rows)

    def _is_rank_zero(self) -> bool:
        """Whether this process should run/log the k_bar diagnostic: always
        True outside of an attached Trainer (e.g. in direct unit tests), and
        gated to the global-zero rank under (possibly distributed) training,
        so the diagnostic isn't redundantly computed/logged/written on every
        rank.

        Returns:
            bool: Whether this process is rank zero (or has no attached
                Trainer).
        """
        try:
            return self.trainer.is_global_zero
        except RuntimeError:
            return True

    def on_train_epoch_end(self) -> None:
        """Lightning hook: log the read-only k_bar diagnostic every
        ``_KBAR_LOG_EVERY_N_EPOCHS`` epochs (see ``_log_kbar_diagnostics``).
        """
        if not self._is_rank_zero():
            return
        epoch = self.current_epoch + 1
        if epoch % self._KBAR_LOG_EVERY_N_EPOCHS == 0:
            self._log_kbar_diagnostics(epoch)

    def on_fit_end(self) -> None:
        """Lightning hook: log the read-only k_bar diagnostic once more at the
        end of training (see ``_log_kbar_diagnostics``), unless the last epoch
        was already logged by ``on_train_epoch_end``.
        """
        if not self._is_rank_zero():
            return
        self._log_kbar_diagnostics(self.current_epoch + 1)

    # ------------------------------------------------------------------ #
    # One-time exploitability-increment (dE) diagnostic. E_hat is graph-only
    # and frozen right after precompute, so this never needs to run more than
    # once and is never wired into the training loop -- call it explicitly
    # (e.g. right after construction) whenever needed. Read-only: no_grad,
    # touches only the frozen e_hat/degree buffers, never affects training.
    # This model does not implement the optional isotonic projection
    # mentioned in the method text, so `self.e_hat` is always the raw
    # (unprojected) signal -- there is no separate projected version to
    # report here.
    # ------------------------------------------------------------------ #
    _DELTA_E_CSV_FIELDNAMES = ["node_id", "type", "degree", "k", "E", "dE"]
    _DELTA_E_PERCENTILES = (0, 10, 50, 90, 100)

    def _representative_node_local_indices(self, degree_of_type: Tensor) -> List[int]:
        """Pick one node per requested degree percentile (nearest-rank method),
        deduplicated while preserving order.

        Args:
            degree_of_type (Tensor): The (n_type_nodes,) degree of every node
                of one type.

        Returns:
            List[int]: The local (within-type) indices of the selected nodes.
        """
        order = torch.argsort(degree_of_type)
        n = degree_of_type.numel()
        picked = []
        seen = set()
        for percentile in self._DELTA_E_PERCENTILES:
            pos = min(int(round(percentile / 100.0 * (n - 1))), n - 1)
            local_idx = int(order[pos].item())
            if local_idx not in seen:
                seen.add(local_idx)
                picked.append(local_idx)
        return picked

    def _write_delta_e_csv(self, csv_path: str, rows: List[Dict[str, Any]]) -> None:
        """Best-effort CSV write of the full per-(node, k) dE diagnostic data.
        Never raises: a write failure is logged and otherwise swallowed.

        Args:
            csv_path (str): The destination CSV path (overwritten if it
                already exists, since this is a one-shot diagnostic, not an
                append-per-epoch log like the k_bar CSV).
            rows (List[Dict[str, Any]]): One dict per (node, k) pair, with
                keys matching ``_DELTA_E_CSV_FIELDNAMES``.
        """
        try:
            with open(csv_path, "w", newline="", encoding="utf-8") as csv_file:
                writer = csv.DictWriter(
                    csv_file, fieldnames=self._DELTA_E_CSV_FIELDNAMES
                )
                writer.writeheader()
                writer.writerows(rows)
            logger.attention(
                f"[dE diagnostics] wrote {len(rows)} rows to "
                f"{os.path.abspath(csv_path)}"
            )
        except OSError as error:
            logger.attention(f"[dE diagnostics] could not write CSV: {error}")

    def diagnose_delta_e(
        self, csv_path: Optional[str] = "deltaE_diagnostics.csv"
    ) -> None:
        """One-time diagnostic of the exploitability increments
        ``dE[v,k] = E_hat[v,k+1] - E_hat[v,k]``, which enter the controller's
        marginal utility ``m[v,k] = dE[v,k] + gamma*dS[v,k] - lambda0``. With
        gamma=0 (the specificity-ablation setting), dE is the only thing that
        can differentiate nodes, so this inspects its scale and cross-node
        variability directly: representative nodes at degree percentiles
        {0, 10, 50, 90, 100} per type, aggregate dE statistics per type and
        per k, the scale of dE[:,0] (the value to compare against lambda0),
        and the Spearman correlation between dE[:,0] and degree.

        Read-only: runs under ``torch.no_grad()``, only reads the frozen
        ``e_hat``/``degree`` buffers, and never affects training.

        Args:
            csv_path (Optional[str]): Where to write the full per-(node, k)
                CSV (columns: node_id, type, degree, k, E, dE). If None, the
                CSV is skipped and only the log output is produced.
        """
        with torch.no_grad():
            e_hat = self.e_hat.detach().cpu()
        d_e = e_hat[:, 1:] - e_hat[:, :-1]
        degree = self.degree.detach().cpu()
        n_layers = self.n_layers

        logger.attention(
            f"[dE diagnostics] E_hat and dE for representative nodes (K={n_layers})"
        )
        header_e = "  ".join(f"E{k}" for k in range(n_layers + 1))
        header_de = "  ".join(f"dE{k}" for k in range(n_layers))
        logger.attention(f"type   node     deg    {header_e}    {header_de}")

        csv_rows: List[Dict[str, Any]] = []
        types = (
            ("user", slice(0, self.n_users), 0, self.n_users),
            ("item", slice(self.n_users, None), self.n_users, self.n_items),
        )

        for node_type, node_slice, offset, n_type_nodes in types:
            degree_t = degree[node_slice]
            e_t = e_hat[node_slice]
            d_e_t = d_e[node_slice]

            for local_idx in self._representative_node_local_indices(degree_t):
                global_idx = offset + local_idx
                node_deg = int(degree_t[local_idx].item())
                e_row = e_t[local_idx]
                de_row = d_e_t[local_idx]

                e_str = "  ".join(f"{v:.4e}" for v in e_row.tolist())
                de_str = "  ".join(f"{v:.4e}" for v in de_row.tolist())
                logger.attention(
                    f"{node_type:5s}  {global_idx:6d}  {node_deg:5d}   "
                    f"{e_str}   {de_str}"
                )

            for local_idx in range(n_type_nodes):
                global_idx = offset + local_idx
                node_deg = int(degree_t[local_idx].item())
                for k in range(n_layers + 1):
                    csv_rows.append(
                        {
                            "node_id": global_idx,
                            "type": node_type,
                            "degree": node_deg,
                            "k": k,
                            "E": e_t[local_idx, k].item(),
                            "dE": (d_e_t[local_idx, k].item() if k < n_layers else ""),
                        }
                    )

        for node_type, node_slice, _offset, _n_type_nodes in types:
            degree_t = degree[node_slice]
            d_e_t = d_e[node_slice]

            logger.attention(
                f"[dE diagnostics] dE[:,k] stats over all {node_type.upper()} nodes:"
            )
            for k in range(n_layers):
                stats = self._kbar_stats(d_e_t[:, k])
                logger.attention(
                    f"  k={k}: mean={stats['mean']:.4e} std={stats['std']:.4e} "
                    f"min={stats['min']:.4e} max={stats['max']:.4e} "
                    f"p10={stats['p10']:.4e} p50={stats['p50']:.4e} "
                    f"p90={stats['p90']:.4e}"
                )

            first = d_e_t[:, 0]
            logger.attention(
                f"[dE diagnostics] scale of dE[:,0] (compare with lambda0): "
                f"{node_type} mean={first.mean().item():.4e} "
                f"range=[{first.min().item():.4e}, {first.max().item():.4e}]"
            )

            corr, pvalue = spearmanr(first.numpy(), degree_t.numpy())
            logger.attention(
                f"[dE diagnostics] spearman(dE[:,0], degree) [{node_type}]: "
                f"corr={corr:.4f} (p={pvalue:.3g})"
            )

        if csv_path is not None:
            self._write_delta_e_csv(csv_path, csv_rows)

    # ------------------------------------------------------------------ #
    # One-time specificity (S) and dS diagnostic. Complements
    # `diagnose_delta_e`: unlike E (graph-only, frozen), S depends on the
    # CURRENT embeddings X (via the residuals r_v^{(k)} = z_v^{(k)} - mu_v), so
    # this must be run at a specific, identified point in training (pass an
    # `epoch_label` for the log output) rather than being meaningful at any
    # arbitrary time. Reuses `_specificity` as-is (same frozen anchors, same
    # self-anchor masking, same mu = phi0 (phi0^T X) computed from the current
    # X) -- this diagnostic only adds aggregation/reporting on top, and calls
    # it with every node (not just a training batch) for complete statistics.
    # Read-only: no_grad, no effect on training.
    # ------------------------------------------------------------------ #
    _DELTA_S_CSV_FIELDNAMES = ["node_id", "type", "degree", "k", "S", "dS"]

    def _write_delta_s_csv(self, csv_path: str, rows: List[Dict[str, Any]]) -> None:
        """Best-effort CSV write of the full per-(node, k) dS diagnostic data.
        Never raises: a write failure is logged and otherwise swallowed.

        Args:
            csv_path (str): The destination CSV path (overwritten if it
                already exists).
            rows (List[Dict[str, Any]]): One dict per (node, k) pair, with
                keys matching ``_DELTA_S_CSV_FIELDNAMES``.
        """
        try:
            with open(csv_path, "w", newline="", encoding="utf-8") as csv_file:
                writer = csv.DictWriter(
                    csv_file, fieldnames=self._DELTA_S_CSV_FIELDNAMES
                )
                writer.writeheader()
                writer.writerows(rows)
            logger.attention(
                f"[dS diagnostics] wrote {len(rows)} rows to "
                f"{os.path.abspath(csv_path)}"
            )
        except OSError as error:
            logger.attention(f"[dS diagnostics] could not write CSV: {error}")

    def diagnose_delta_s(
        self,
        epoch_label: str = "current",
        csv_path: Optional[str] = "deltaS_diagnostics.csv",
    ) -> None:
        """One-time diagnostic of the specificity signal ``S_v^{(k)}`` and its
        SIGNED increments ``dS[v,k] = S[v,k+1] - S[v,k]``, which enter the
        controller's marginal utility
        ``m[v,k] = dE[v,k] + gamma*dS[v,k] - lambda0``. Reports, per type: the
        same representative nodes as ``diagnose_delta_e`` (so the two tables
        are directly comparable row-by-row), aggregate dS statistics over
        every node, the scale of ``mean|dS[:,0]|`` against ``mean|dE[:,0]|``
        (and the gamma value that would balance them), the coefficient of
        variation of dE0 vs dS0 (higher CV = more node-specific), and the
        Spearman correlation of S/dS against node degree.

        Since S depends on the CURRENT embeddings (unlike the graph-only,
        frozen E), this diagnostic is only meaningful at whatever point in
        training it's called -- ``epoch_label`` identifies that point in the
        log output (e.g. an epoch number or a checkpoint path); it has no
        effect on the computation itself.

        Read-only: runs under ``torch.no_grad()``, and never affects
        training. Reuses ``_specificity`` unchanged (same frozen anchors,
        the same self-anchor masking that already excludes a node from its
        own softmax, and ``mu`` recomputed from the current X), called here
        over every node rather than a training batch.

        Args:
            epoch_label (str): A label identifying the training state this
                diagnostic reflects, printed in the log output.
            csv_path (Optional[str]): Where to write the full per-(node, k)
                CSV (columns: node_id, type, degree, k, S, dS). If None, the
                CSV is skipped and only the log output is produced.
        """
        with torch.no_grad():
            x = self._ego_embeddings()
            z_list = self._propagate(x)
            coeff = self._consensus(z_list[0])
            n_nodes = self.n_users + self.n_items
            all_idx = torch.arange(n_nodes, device=x.device)
            s_all = self._specificity(z_list, coeff, all_idx).cpu()

        d_s = s_all[:, 1:] - s_all[:, :-1]
        e_hat = self.e_hat.detach().cpu()
        d_e = e_hat[:, 1:] - e_hat[:, :-1]
        degree = self.degree.detach().cpu()
        n_layers = self.n_layers

        logger.attention(
            f"[dS diagnostics @ {epoch_label}] S and dS for representative "
            f"nodes (K={n_layers})"
        )
        header_s = "  ".join(f"S{k}" for k in range(n_layers + 1))
        header_ds = "  ".join(f"dS{k}" for k in range(n_layers))
        logger.attention(f"type   node     deg    {header_s}    {header_ds}")

        csv_rows: List[Dict[str, Any]] = []
        types = (
            ("user", slice(0, self.n_users), 0, self.n_users),
            ("item", slice(self.n_users, None), self.n_users, self.n_items),
        )

        for node_type, node_slice, offset, n_type_nodes in types:
            degree_t = degree[node_slice]
            s_t = s_all[node_slice]
            d_s_t = d_s[node_slice]

            for local_idx in self._representative_node_local_indices(degree_t):
                global_idx = offset + local_idx
                node_deg = int(degree_t[local_idx].item())
                s_str = "  ".join(f"{v:.4e}" for v in s_t[local_idx].tolist())
                ds_str = "  ".join(f"{v:.4e}" for v in d_s_t[local_idx].tolist())
                logger.attention(
                    f"{node_type:5s}  {global_idx:6d}  {node_deg:5d}   "
                    f"{s_str}   {ds_str}"
                )

            for local_idx in range(n_type_nodes):
                global_idx = offset + local_idx
                node_deg = int(degree_t[local_idx].item())
                for k in range(n_layers + 1):
                    csv_rows.append(
                        {
                            "node_id": global_idx,
                            "type": node_type,
                            "degree": node_deg,
                            "k": k,
                            "S": s_t[local_idx, k].item(),
                            "dS": (d_s_t[local_idx, k].item() if k < n_layers else ""),
                        }
                    )

        for node_type, node_slice, _offset, _n_type_nodes in types:
            degree_t = degree[node_slice]
            s_t = s_all[node_slice]
            d_s_t = d_s[node_slice]
            d_e_t = d_e[node_slice]

            logger.attention(
                f"[dS diagnostics] dS[:,k] stats over all {node_type.upper()} nodes:"
            )
            for k in range(n_layers):
                stats = self._kbar_stats(d_s_t[:, k])
                logger.attention(
                    f"  k={k}: mean={stats['mean']:.4e} std={stats['std']:.4e} "
                    f"min={stats['min']:.4e} max={stats['max']:.4e} "
                    f"p10={stats['p10']:.4e} p50={stats['p50']:.4e} "
                    f"p90={stats['p90']:.4e}"
                )

            de0 = d_e_t[:, 0]
            ds0 = d_s_t[:, 0]
            mean_abs_de0 = de0.abs().mean().item()
            mean_abs_ds0 = ds0.abs().mean().item()
            ratio = mean_abs_de0 / (mean_abs_ds0 + self._EPS)
            logger.attention(
                f"[scale comparison @ k=0, {node_type}] mean|dE0|="
                f"{mean_abs_de0:.4e} mean|dS0|={mean_abs_ds0:.4e} "
                f"ratio={ratio:.4g} -> suggested gamma ~ {ratio:.4g} "
                "(to balance gamma*dS against dE at layer 0)"
            )

            cv_de0 = de0.std().item() / (abs(de0.mean().item()) + self._EPS)
            cv_ds0 = ds0.std().item() / (abs(ds0.mean().item()) + self._EPS)
            logger.attention(
                f"[node-specificity, {node_type}] CV(dE0)={cv_de0:.4g} "
                f"CV(dS0)={cv_ds0:.4g} (higher CV = more node-specific)"
            )

            corr_s, pvalue_s = spearmanr(s_t[:, 0].numpy(), degree_t.numpy())
            corr_ds, pvalue_ds = spearmanr(ds0.numpy(), degree_t.numpy())
            logger.attention(
                f"[spearman @ k=0, {node_type}] S vs degree: corr={corr_s:.4f} "
                f"(p={pvalue_s:.3g}); dS vs degree: corr={corr_ds:.4f} "
                f"(p={pvalue_ds:.3g})"
            )

        if csv_path is not None:
            self._write_delta_s_csv(csv_path, csv_rows)
