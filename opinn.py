"""
OPINN: Physics-Informed Neural Opinion Dynamics
================================================
Implementation of the architecture from:
  "Advancing Opinion Dynamics Modeling with Neural
   Diffusion-Convection-Reaction Equation" (arXiv 2602.05403)

Section layout (reading top-to-bottom in dependency order):
  1. ODE solvers          – Euler, RK-4 (no external deps)
  2. Graph preprocessing  – normalized adjacency matrix
  3. DCR sub-modules      – Diffusion, Convection, Reaction
  4. Neural dynamics      – DCR combination + gating weights
  5. Encoder / Decoder    – GRU + MLP
  6. Full OPINN model
  7. Training & evaluation utilities
  8. Weight-matrix CSV export
  9. run_experiment helper
 10. __main__ demo

Tensor shape conventions used throughout:
  N  = number of users/nodes
  D  = hidden (latent) dimension
  F  = input feature dimension per node (1 for scalar opinion)
  T  = total time steps in a dataset
  c  = encoder context length
  h  = forecast horizon

The model's input format is always [N, F, T] (or windows thereof).
Setting F=1 matches the paper; increasing F later only requires
changing the `input_dim` argument – no other modifications.

CSV weight-matrix convention:
  Weight matrices are stored as W such that the forward computation
  is  X @ W.T  (PyTorch nn.Linear convention: weight shape [out, in]).
  In the CSV each *row* corresponds to one output dimension.
  Pass normalize=True to save Frobenius-normalized versions for
  clean cross-dataset comparison.
"""

import copy
import math
from pathlib import Path
from typing import Dict, List, Literal, Optional

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F

# ─── 1. ODE Solvers ───────────────────────────────────────────────────────────

def _euler_step(
    f,
    z: torch.Tensor,
    t: float,
    dt: float,
) -> torch.Tensor:
    """First-order Euler step: z(t+dt) = z(t) + dt·f(z, t)."""
    return z + dt * f(z, t)


def _rk4_step(
    f,
    z: torch.Tensor,
    t: float,
    dt: float,
) -> torch.Tensor:
    """Classical 4th-order Runge-Kutta step."""
    k1 = f(z,                  t)
    k2 = f(z + 0.5 * dt * k1, t + 0.5 * dt)
    k3 = f(z + 0.5 * dt * k2, t + 0.5 * dt)
    k4 = f(z + dt * k3,        t + dt)
    return z + (dt / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4)


def ode_solve(
    f,
    z0: torch.Tensor,
    n_steps: int,
    dt: float = 1.0,
    method: Literal["rk4", "euler"] = "rk4",
) -> List[torch.Tensor]:
    """
    Integrate an ODE from z0 for n_steps steps of size dt.

    Args:
        f:       right-hand side, signature f(z, t) -> dz/dt  [N, D]
        z0:      initial state                                  [N, D]
        n_steps: number of integration steps (= forecast horizon)
        dt:      step size (paper default: 1.0)
        method:  'rk4' (recommended) or 'euler'

    Returns:
        List of n_steps tensors each [N, D], corresponding to
        z(dt), z(2·dt), …, z(n_steps·dt).  Initial state z0 is
        NOT included in the returned list.
    """
    step_fn = _rk4_step if method == "rk4" else _euler_step
    states: List[torch.Tensor] = []
    z = z0
    for i in range(n_steps):
        z = step_fn(f, z, i * dt, dt)
        states.append(z)
    return states


# ─── 2. Graph Preprocessing ──────────────────────────────────────────────────

def compute_normalized_adjacency(adj: torch.Tensor) -> torch.Tensor:
    """
    Symmetric normalized adjacency: A_norm = D^{-1/2} A D^{-1/2}.

    This equals (I - L̃) where L̃ is the symmetric normalized Laplacian,
    so the diffusion term σ((I - L̃) Z W_D) is equivalent to
    σ(A_norm @ Z @ W_D).

    Isolated nodes (degree 0) get D^{-1/2} = 0, leaving their rows/cols
    as all-zeros in A_norm (no information flow, which is correct).

    Args:
        adj: [N, N] symmetric adjacency matrix (binary or weighted)

    Returns:
        A_norm: [N, N]
    """
    deg = adj.sum(dim=1)                     # [N]
    d_inv_sqrt = deg.pow(-0.5)
    d_inv_sqrt[~torch.isfinite(d_inv_sqrt)] = 0.0   # isolated nodes → 0
    D_inv_sqrt = torch.diag(d_inv_sqrt)             # [N, N]
    return D_inv_sqrt @ adj @ D_inv_sqrt


# ─── 3. DCR Sub-Modules ──────────────────────────────────────────────────────

class DiffusionModule(nn.Module):
    """
    Local opinion consensus via graph convolution (Eq. 11).

        dZ/dt_Dif = ReLU( A_norm @ Z @ W_D )

    where A_norm = D^{-1/2} A D^{-1/2} is the symmetric normalized
    adjacency and W_D ∈ R^{D×D} is the learnable weight matrix.

    The paper uses σ generically; the implementation details specify
    ReLU for both diffusion and convection activations.
    """

    def __init__(self, hidden_dim: int) -> None:
        super().__init__()
        # weight shape: [D, D]  (stored as [out, in] = [D, D] in nn.Linear)
        self.W_D = nn.Linear(hidden_dim, hidden_dim, bias=False)

    def forward(self, Z: torch.Tensor, A_norm: torch.Tensor) -> torch.Tensor:
        """
        Args:
            Z:      system state   [N, D]
            A_norm: precomputed normalized adjacency [N, N]
        Returns:
            dZ/dt_Dif              [N, D]
        """
        return F.relu(A_norm @ self.W_D(Z))

    def get_weight(self) -> Dict[str, torch.Tensor]:
        """Return W_D as a [D, D] cpu tensor."""
        return {"W_D": self.W_D.weight.detach().cpu()}


class ConvectionModule(nn.Module):
    """
    Global directional opinion drift via attention (Eqs. 12-13).

    Standard attention (quadratic, O(N²·D)):
        V̄_ij  = sigmoid( (z_i - z_j) @ W_V.T )   scalar velocity j→i
        A      = softmax( V̄, dim=1 )               row-stochastic [N, N]
        out    = A @ W_C(Z)

        ⚠ For N > ~500 the intermediate [N, N, D] tensor becomes large.
          Switch to attention='linear' for big graphs.

    Linear attention (O(N·D²), from Appendix E.3 / SGFormer [47]):
        φ(x) = ELU(x) + 1     (positive feature map)
        Q = φ( W_Q(Z) ),  K = φ( W_K(Z) )        both [N, D]
        V = W_C(Z)                                  [N, D]
        num   = Q @ (K.T @ V)                       [N, D]
        denom = ( Q * K.sum(dim=0) ).sum(dim=1, keepdim=True)   [N, 1]
        out   = num / (denom + ε)

    W_C is shared between both paths (it transforms the value vectors).
    """

    def __init__(
        self,
        hidden_dim: int,
        attention: Literal["standard", "linear"] = "standard",
    ) -> None:
        super().__init__()
        self.attention = attention

        # Shared: value / output projection   [D, D]
        self.W_C = nn.Linear(hidden_dim, hidden_dim, bias=False)

        if attention == "standard":
            # Maps D-dim difference vector to scalar velocity   weight: [1, D]
            self.W_V = nn.Linear(hidden_dim, 1, bias=False)
        else:  # linear
            self.W_Q = nn.Linear(hidden_dim, hidden_dim, bias=False)
            self.W_K = nn.Linear(hidden_dim, hidden_dim, bias=False)

    def forward(self, Z: torch.Tensor) -> torch.Tensor:
        """
        Args:
            Z: system state [N, D]
        Returns:
            dZ/dt_Con [N, D]
        """
        if self.attention == "standard":
            return self._standard(Z)
        return self._linear(Z)

    def _standard(self, Z: torch.Tensor) -> torch.Tensor:
        N = Z.shape[0]
        # Pairwise differences  diff[i, j] = z_i - z_j
        zi = Z.unsqueeze(1).expand(N, N, -1)   # [N, N, D]
        zj = Z.unsqueeze(0).expand(N, N, -1)   # [N, N, D]
        diff = zi - zj                          # [N, N, D]
        # Scalar velocity: W_V applied to last dim → [N, N, 1] → [N, N]
        V_bar = torch.sigmoid(self.W_V(diff).squeeze(-1))  # [N, N]
        A = F.softmax(V_bar, dim=1)            # row-stochastic [N, N]
        return A @ self.W_C(Z)                 # [N, D]

    def _linear(self, Z: torch.Tensor) -> torch.Tensor:
        phi = lambda x: F.elu(x) + 1.0        # positive feature map
        Q = phi(self.W_Q(Z))                   # [N, D]
        K = phi(self.W_K(Z))                   # [N, D]
        V = self.W_C(Z)                        # [N, D]
        # Kernel trick: avoid explicit N×N attention matrix
        KtV = K.T @ V                          # [D, D]   O(N·D²)
        numerator = Q @ KtV                    # [N, D]
        K_sum = K.sum(dim=0)                   # [D]
        # denom_i = Q_i · (Σ_j K_j)  — one scalar per node
        denominator = (Q * K_sum).sum(dim=1, keepdim=True)  # [N, 1]
        return numerator / (denominator + 1e-6)

    def get_weights(self) -> Dict[str, torch.Tensor]:
        """
        Return weight matrices on CPU.
        W_V is stored as [1, D] in nn.Linear; returned as [D] (1-D vector)
        to keep CSV rows meaningful.
        """
        out: Dict[str, torch.Tensor] = {
            "W_C": self.W_C.weight.detach().cpu(),
        }
        if self.attention == "standard":
            out["W_V"] = self.W_V.weight.detach().cpu().squeeze(0)  # [D]
        else:
            out["W_Q"] = self.W_Q.weight.detach().cpu()
            out["W_K"] = self.W_K.weight.detach().cpu()
        return out


class ReactionModule(nn.Module):
    """
    Endogenous opinion dynamics / individual-level shifts (Eq. 14).

        source:     dZ/dt_Rea = Z                    (identity, no params)
        linear:     dZ/dt_Rea = Z @ W_Rea.T          (single linear layer)
        nonlinear:  dZ/dt_Rea = MLP(Z)               (two linear layers)

    The source term parallels the FJ model's stubbornness; the linear
    term captures anchoring / relaxation; the nonlinear term can model
    Allen-Cahn-type polarization dynamics.
    """

    def __init__(
        self,
        hidden_dim: int,
        reaction_type: Literal["source", "linear", "nonlinear"] = "nonlinear",
    ) -> None:
        super().__init__()
        self.reaction_type = reaction_type

        if reaction_type == "linear":
            self.W_Rea = nn.Linear(hidden_dim, hidden_dim, bias=False)
        elif reaction_type == "nonlinear":
            self.mlp = nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim),
                nn.Tanh(),
                nn.Linear(hidden_dim, hidden_dim),
            )

    def forward(self, Z: torch.Tensor) -> torch.Tensor:
        """
        Args:
            Z: system state [N, D]
        Returns:
            dZ/dt_Rea [N, D]
        """
        if self.reaction_type == "source":
            return Z
        if self.reaction_type == "linear":
            return self.W_Rea(Z)
        return self.mlp(Z)  # nonlinear

    def get_weights(self) -> Dict[str, torch.Tensor]:
        """Return learnable weight tensors on CPU (empty dict for 'source')."""
        if self.reaction_type == "source":
            return {}
        if self.reaction_type == "linear":
            return {"W_Rea": self.W_Rea.weight.detach().cpu()}
        # nonlinear: two weight matrices + two bias vectors
        return {
            "Rea_W1": self.mlp[0].weight.detach().cpu(),   # [D, D]
            "Rea_b1": self.mlp[0].bias.detach().cpu(),     # [D]
            "Rea_W2": self.mlp[2].weight.detach().cpu(),   # [D, D]
            "Rea_b2": self.mlp[2].bias.detach().cpu(),     # [D]
        }


# ─── 4. Neural Dynamics Module ───────────────────────────────────────────────

class NeuralOpinionDynamics(nn.Module):
    """
    DCR vector field (Eq. 15):

        dZ/dt = ω · dZ/dt_Dif + (1-ω) · dZ/dt_Con + δ · dZ/dt_Rea

    Gating weights ω and δ are always learned as nn.Parameters
    (stored in logit space so sigmoid keeps them in (0, 1)).

    Post-training analysis:
        model.dynamics.set_gating_weights(omega=0.9, delta=0.1)
            → overrides used in forward() only; learned parameters
              are untouched and still returned by get_weight_matrices()
        model.dynamics.clear_gating_overrides()
            → restores learned behavior
    """

    def __init__(
        self,
        hidden_dim: int,
        adj: torch.Tensor,
        omega_init: float = 0.5,
        delta_init: float = 0.5,
        attention: Literal["standard", "linear"] = "standard",
        reaction_type: Literal["source", "linear", "nonlinear"] = "nonlinear",
    ) -> None:
        super().__init__()

        # Precompute normalized adjacency once; register as buffer so
        # it moves with model.to(device) automatically.
        A_norm = compute_normalized_adjacency(adj)
        self.register_buffer("A_norm", A_norm)

        # DCR components
        self.diffusion  = DiffusionModule(hidden_dim)
        self.convection = ConvectionModule(hidden_dim, attention)
        self.reaction   = ReactionModule(hidden_dim, reaction_type)

        # Gating weights in logit space → sigmoid maps them to (0, 1).
        # logit(0.5) = 0 so initializing at 0.5 means starting at 0.
        self._omega_logit = nn.Parameter(
            torch.tensor(math.log(omega_init / (1.0 - omega_init)))
        )
        self._delta_logit = nn.Parameter(
            torch.tensor(math.log(delta_init / (1.0 - delta_init)))
        )

        # Override slots for post-training sensitivity analysis.
        # These are plain Python floats, NOT parameters – they are
        # never seen by the optimizer.
        self._omega_override: Optional[float] = None
        self._delta_override: Optional[float] = None

    # ── Learned gating values (always based on parameters) ──────────────────

    @property
    def omega(self) -> torch.Tensor:
        """Learned ω ∈ (0, 1) – diffusion vs. convection balance."""
        return torch.sigmoid(self._omega_logit)

    @property
    def delta(self) -> torch.Tensor:
        """Learned δ ∈ (0, 1) – reaction strength."""
        return torch.sigmoid(self._delta_logit)

    # ── Effective values used inside forward() ──────────────────────────────

    def _eff_omega(self) -> torch.Tensor:
        if self._omega_override is not None:
            return torch.tensor(
                self._omega_override, dtype=torch.float32, device=self.A_norm.device
            )
        return self.omega

    def _eff_delta(self) -> torch.Tensor:
        if self._delta_override is not None:
            return torch.tensor(
                self._delta_override, dtype=torch.float32, device=self.A_norm.device
            )
        return self.delta

    # ── Public API for post-training analysis ───────────────────────────────

    def set_gating_weights(
        self,
        omega: Optional[float] = None,
        delta: Optional[float] = None,
    ) -> None:
        """
        Override gating weights for inference / sensitivity analysis.
        Learned parameters are NOT modified; call clear_gating_overrides()
        to restore them.  Pass None to keep a dimension's current value.
        """
        if omega is not None:
            self._omega_override = float(omega)
        if delta is not None:
            self._delta_override = float(delta)

    def clear_gating_overrides(self) -> None:
        """Restore learned gating weights in forward()."""
        self._omega_override = None
        self._delta_override = None

    def get_gating_info(self) -> Dict[str, float]:
        """
        Return a dict with both learned and effective (post-override) values.
        Useful for displaying what the model actually uses.
        """
        return {
            "omega_learned":   self.omega.item(),
            "delta_learned":   self.delta.item(),
            "omega_effective": self._eff_omega().item(),
            "delta_effective": self._eff_delta().item(),
        }

    # ── Weight-matrix extraction (always uses learned parameters) ───────────

    def get_weight_matrices(self) -> Dict[str, torch.Tensor]:
        """
        Collect all learnable weight matrices from the DCR components.
        Gating scalars are returned as shape-[1,1] tensors so the CSV
        export logic can treat everything uniformly.

        NOTE: always reflects learned parameter values, never overrides.
        """
        matrices: Dict[str, torch.Tensor] = {}
        matrices.update(self.diffusion.get_weight())
        matrices.update(self.convection.get_weights())
        matrices.update(self.reaction.get_weights())
        # Scalars as [1, 1] so Frobenius-norm code works uniformly
        matrices["omega"] = self.omega.detach().cpu().view(1, 1)
        matrices["delta"] = self.delta.detach().cpu().view(1, 1)
        return matrices

    # ── ODE right-hand side ─────────────────────────────────────────────────

    def forward(self, Z: torch.Tensor, t: float = 0.0) -> torch.Tensor:  # noqa: ARG002
        """
        Args:
            Z: system state [N, D]
            t: current time (kept for ODE solver interface; not used here)
        Returns:
            dZ/dt [N, D]
        """
        dZ_dif = self.diffusion(Z, self.A_norm)
        dZ_con = self.convection(Z)
        dZ_rea = self.reaction(Z)
        w = self._eff_omega()
        d = self._eff_delta()
        return w * dZ_dif + (1.0 - w) * dZ_con + d * dZ_rea


# ─── 5. Encoder & Decoder ────────────────────────────────────────────────────

class Encoder(nn.Module):
    """
    GRU-based encoder (Eq. 16).

    Maps per-user opinion history → latent system state Z(t).

        Z(t) = Φ( X[t-c : t] )

    Input shape:  [N, F, c]   (N nodes, F features, c context steps)
    Output shape: [N, D]

    The GRU treats each node independently as a separate batch item.
    Extending to F > 1 only requires passing a different input_dim –
    no structural changes are needed anywhere else.
    """

    def __init__(self, input_dim: int, hidden_dim: int) -> None:
        super().__init__()
        self.gru = nn.GRU(
            input_size=input_dim,
            hidden_size=hidden_dim,
            batch_first=True,   # input: [batch=N, seq=c, features=F]
        )

    def forward(self, X_window: torch.Tensor) -> torch.Tensor:
        """
        Args:
            X_window: [N, F, c]
        Returns:
            Z: [N, D]
        """
        # GRU with batch_first=True wants [N, seq_len, features]
        x = X_window.permute(0, 2, 1)      # [N, c, F]
        _, h_n = self.gru(x)               # h_n: [1, N, D]
        return h_n.squeeze(0)              # [N, D]


class Decoder(nn.Module):
    """
    Two-layer MLP decoder (Eq. 19).

    Maps future latent states → opinion predictions.

        X̂[T+1:T+h] = Ψ( Ẑ[T+1:T+h] )

    Input shape:  [N, D, h]
    Output shape: [N, h]  (for output_dim=1)
    """

    def __init__(self, hidden_dim: int, output_dim: int = 1) -> None:
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, output_dim),
        )
        self.output_dim = output_dim

    def forward(self, Z_seq: torch.Tensor) -> torch.Tensor:
        """
        Args:
            Z_seq: [N, D, h]
        Returns:
            X_hat: [N, h] if output_dim=1, else [N, h, output_dim]
        """
        N, D, h = Z_seq.shape
        # Flatten nodes & horizon for batch MLP application
        Z_flat = Z_seq.permute(0, 2, 1).reshape(N * h, D)  # [N*h, D]
        out = self.mlp(Z_flat)                              # [N*h, output_dim]
        out = out.reshape(N, h, self.output_dim)            # [N, h, output_dim]
        if self.output_dim == 1:
            out = out.squeeze(-1)                           # [N, h]
        return out


# ─── 6. Full OPINN Model ─────────────────────────────────────────────────────

class OPINN(nn.Module):
    """
    Physics-Informed Neural Network for Opinion Dynamics (Figure 3).

    Architecture:
        Encoder (GRU)  →  Neural ODE (DCR dynamics)  →  Decoder (MLP)

    Args:
        adj:           [N, N] adjacency matrix (CPU tensor)
        input_dim:     opinion feature size per node (1 for scalar opinion)
        hidden_dim:    latent space dimension D
        omega_init:    initial ω ∈ (0,1) – diffusion / convection balance
        delta_init:    initial δ ∈ (0,1) – reaction strength
        attention:     'standard' (quadratic) or 'linear' (O(N·D²))
        reaction_type: 'source', 'linear', or 'nonlinear'
        ode_method:    'rk4' (default) or 'euler'
        ode_dt:        integration step size (paper default: 1.0)
    """

    def __init__(
        self,
        adj: torch.Tensor,
        input_dim: int = 1,
        hidden_dim: int = 32,
        omega_init: float = 0.5,
        delta_init: float = 0.5,
        attention: Literal["standard", "linear"] = "standard",
        reaction_type: Literal["source", "linear", "nonlinear"] = "nonlinear",
        ode_method: Literal["rk4", "euler"] = "rk4",
        ode_dt: float = 1.0,
    ) -> None:
        super().__init__()
        self.ode_method = ode_method
        self.ode_dt = ode_dt

        self.encoder = Encoder(input_dim, hidden_dim)
        self.dynamics = NeuralOpinionDynamics(
            hidden_dim=hidden_dim,
            adj=adj,
            omega_init=omega_init,
            delta_init=delta_init,
            attention=attention,
            reaction_type=reaction_type,
        )
        self.decoder = Decoder(hidden_dim, output_dim=input_dim)

    def forward(
        self,
        X_window: torch.Tensor,
        horizon: int,
    ) -> torch.Tensor:
        """
        Args:
            X_window: historical opinion window  [N, input_dim, context_len]
            horizon:  number of future steps to predict
        Returns:
            X_hat: opinion predictions  [N, horizon]  (input_dim=1)
                   or [N, horizon, input_dim] for input_dim > 1
        """
        Z0 = self.encoder(X_window)                         # [N, D]
        Z_steps = ode_solve(
            self.dynamics, Z0,
            n_steps=horizon,
            dt=self.ode_dt,
            method=self.ode_method,
        )                                                   # list of horizon [N,D] tensors
        Z_seq = torch.stack(Z_steps, dim=2)                 # [N, D, horizon]
        return self.decoder(Z_seq)                          # [N, horizon]

    # ── Convenience delegates ────────────────────────────────────────────────

    def set_gating_weights(
        self,
        omega: Optional[float] = None,
        delta: Optional[float] = None,
    ) -> None:
        """Override gating weights for post-training analysis. See NeuralOpinionDynamics."""
        self.dynamics.set_gating_weights(omega, delta)

    def clear_gating_overrides(self) -> None:
        """Restore learned gating weights."""
        self.dynamics.clear_gating_overrides()

    def get_gating_info(self) -> Dict[str, float]:
        """Learned and effective (post-override) gating values."""
        return self.dynamics.get_gating_info()

    def get_weight_matrices(self) -> Dict[str, torch.Tensor]:
        """All learnable DCR weight matrices (CPU tensors). Never uses overrides."""
        return self.dynamics.get_weight_matrices()


# ─── 7. Training & Evaluation ────────────────────────────────────────────────

def train_epoch(
    model: OPINN,
    optimizer: torch.optim.Optimizer,
    X_seq: torch.Tensor,
    context_len: int,
    horizon: int,
    device: torch.device,
    max_grad_norm: float = 1.0,
) -> float:
    """
    One training epoch with sliding-window batches.

    Args:
        X_seq:        full opinion time series  [N, input_dim, T]
        context_len:  encoder look-back window
        horizon:      forecast horizon
        max_grad_norm: gradient clipping threshold (helps RK-4 stability)

    Returns:
        Average MSE loss over all windows.
    """
    model.train()
    T = X_seq.shape[2]
    total_loss = 0.0
    n_windows = 0

    for start in range(0, T - context_len - horizon + 1, horizon):
        end_ctx = start + context_len
        end_tgt = end_ctx + horizon

        X_ctx = X_seq[:, :, start:end_ctx].to(device)        # [N, F, c]
        # Target: first feature channel only (scalar opinion)
        X_tgt = X_seq[:, 0, end_ctx:end_tgt].to(device)      # [N, h]

        X_hat = model(X_ctx, horizon)                          # [N, h]
        loss = F.mse_loss(X_hat, X_tgt)

        optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
        optimizer.step()

        total_loss += loss.item()
        n_windows += 1

    return total_loss / max(n_windows, 1)


@torch.no_grad()
def evaluate(
    model: OPINN,
    X_seq: torch.Tensor,
    context_len: int,
    horizon: int,
    device: torch.device,
) -> Dict[str, float]:
    """
    Compute MAE and RMSE on the last window of X_seq.

    Uses the `horizon` steps immediately preceding the last `horizon`
    steps as context, and the final `horizon` steps as targets.

    Args:
        X_seq: [N, input_dim, T]

    Returns:
        {'MAE': float, 'RMSE': float}
    """
    model.eval()
    T = X_seq.shape[2]
    start = T - context_len - horizon

    X_ctx = X_seq[:, :, start:start + context_len].to(device)   # [N, F, c]
    X_tgt = X_seq[:, 0, start + context_len:].cpu()             # [N, h]

    X_hat = model(X_ctx, horizon).cpu()                          # [N, h]

    mae  = (X_hat - X_tgt).abs().mean().item()
    rmse = ((X_hat - X_tgt).pow(2).mean()).sqrt().item()
    return {"MAE": mae, "RMSE": rmse}


# ─── 8. Weight-Matrix CSV Export ─────────────────────────────────────────────

def save_weight_matrices(
    model: OPINN,
    experiment_name: str,
    output_dir: str = "weights",
    normalize: bool = False,
) -> Dict[str, torch.Tensor]:
    """
    Save all DCR weight matrices to CSV files for cross-dataset comparison.

    Per-matrix files:
        {output_dir}/{experiment_name}_{matrix_name}.csv
        Rows = output dimensions, columns labelled dim_0, dim_1, ...
        1-D tensors (bias vectors, W_V) are stored as a single data row.
        Scalars (omega, delta) stored as a single-cell CSV.

    Master summary (appended, not overwritten):
        {output_dir}/summary.csv
        Columns: experiment, matrix_name, shape, frobenius_norm, mean, std, min, max

    Args:
        normalize: if True, divide each matrix by its Frobenius norm before
                   saving (the summary always records the *raw* Frobenius norm
                   for comparison, regardless of this flag).
    """
    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    raw_matrices = model.get_weight_matrices()   # always learned values
    summary_rows: List[Dict] = []

    for name, W_cpu in raw_matrices.items():
        W_np = W_cpu.numpy()
        frob = float(np.linalg.norm(W_np))      # Frobenius norm (works for any shape)

        # Optionally normalize
        W_save = W_np / (frob + 1e-9) if normalize and frob > 1e-9 else W_np

        # ── Determine CSV shape ──────────────────────────────────────────────
        if W_save.ndim == 0:
            # Scalar (shouldn't happen with [1,1] but be safe)
            df = pd.DataFrame({"value": [float(W_save)]})
        elif W_save.ndim == 1:
            # Bias vector or W_V [D] – store as single row
            cols = {f"dim_{i}": [float(v)] for i, v in enumerate(W_save)}
            df = pd.DataFrame(cols)
        else:
            # 2-D matrix [R, C]
            cols = [f"dim_{i}" for i in range(W_save.shape[1])]
            df = pd.DataFrame(W_save, columns=cols)
            df.index.name = "row"

        fname = out_path / f"{experiment_name}_{name}.csv"
        df.to_csv(fname)

        # ── Summary row ─────────────────────────────────────────────────────
        summary_rows.append({
            "experiment":     experiment_name,
            "matrix_name":    name,
            "shape":          str(W_np.shape),
            "frobenius_norm": frob,
        })

    # Append to (or create) the master summary CSV
    summary_path = out_path / "summary.csv"
    df_new = pd.DataFrame(summary_rows)
    df_new.to_csv(
        summary_path,
        mode="a",
        header=not summary_path.exists(),
        index=False,
    )

    n = len(raw_matrices)
    print(f"  [weights] Saved {n} matrices → {out_path}/  "
          f"(normalize={normalize})")

    # Return raw matrices so callers can feed them into compare_weight_matrices
    return raw_matrices


# ─── 8b. Weight-Matrix Distance Analysis ─────────────────────────────────────

def compare_weight_matrices(
    matrices_a: Dict[str, torch.Tensor],
    matrices_b: Dict[str, torch.Tensor],
    name_a: str = "A",
    name_b: str = "B",
) -> pd.DataFrame:
    """
    Compute distance metrics between corresponding weight matrices from two
    experiments.  For each shared matrix key the following are computed:

      frob_dist       ||W_A − W_B||_F                   (absolute distance)
      frob_dist_norm  ||Ŵ_A − Ŵ_B||_F                  (after Frobenius-normalising each)
      cosine_dist     1 − cos(vec(W_A), vec(W_B))        (0 = same direction, 2 = opposite)

    frob_dist_norm and cosine_dist are scale-invariant so they measure
    *structural* similarity independent of weight magnitude.

    Args:
        matrices_a / matrices_b: dicts from model.get_weight_matrices()
        name_a / name_b:         labels used for the frobenius_norm columns

    Returns:
        DataFrame with columns:
          matrix_name, frob_{name_a}, frob_{name_b},
          frob_dist, frob_dist_norm, cosine_dist
    """
    rows = []
    for key in sorted(set(matrices_a) & set(matrices_b)):
        wa = matrices_a[key].numpy().ravel().astype(np.float64)
        wb = matrices_b[key].numpy().ravel().astype(np.float64)

        frob_a = float(np.linalg.norm(wa))
        frob_b = float(np.linalg.norm(wb))

        frob_dist = float(np.linalg.norm(wa - wb))

        wa_hat = wa / (frob_a + 1e-9)
        wb_hat = wb / (frob_b + 1e-9)
        frob_dist_norm = float(np.linalg.norm(wa_hat - wb_hat))

        cosine_dist = 1.0 - float(np.dot(wa_hat, wb_hat))

        rows.append({
            "matrix_name":      key,
            f"frob_{name_a}":   frob_a,
            f"frob_{name_b}":   frob_b,
            "frob_dist":        frob_dist,
            "frob_dist_norm":   frob_dist_norm,
            "cosine_dist":      cosine_dist,
        })

    return pd.DataFrame(rows)


def save_weight_distances(
    experiment_matrices: Dict[str, Dict[str, torch.Tensor]],
    output_dir: str = "weights",
) -> pd.DataFrame:
    """
    Compute and save pairwise weight-matrix distances for all experiment pairs.

    Adds exp_a / exp_b columns and concatenates compare_weight_matrices()
    results for every unique pair.

    Saves to {output_dir}/distances.csv and returns the DataFrame.

    Args:
        experiment_matrices: {experiment_name: weight_matrices_dict}
    """
    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    names = list(experiment_matrices.keys())
    all_rows: List[pd.DataFrame] = []

    for i, name_a in enumerate(names):
        for name_b in names[i + 1:]:
            df_pair = compare_weight_matrices(
                experiment_matrices[name_a],
                experiment_matrices[name_b],
                name_a=name_a,
                name_b=name_b,
            )
            df_pair.insert(0, "exp_b", name_b)
            df_pair.insert(0, "exp_a", name_a)
            all_rows.append(df_pair)

    if not all_rows:
        return pd.DataFrame()

    df_all = pd.concat(all_rows, ignore_index=True)
    dist_path = out_path / "distances.csv"
    df_all.to_csv(dist_path, index=False)
    print(f"  [distances] Saved pairwise matrix distances → {dist_path}")
    return df_all


# ─── 8c. Cross-Prediction Analysis ───────────────────────────────────────────

def transplant_dynamics_weights(
    source_model: "OPINN",
    target_model: "OPINN",
) -> "OPINN":
    """
    Return a deep copy of *target_model* whose DCR dynamics weights have been
    replaced by those of *source_model*.

    Only `model.dynamics` (DiffusionModule, ConvectionModule, ReactionModule,
    and gating parameters) is swapped.  The encoder and decoder are kept from
    *target_model*, so the returned model uses target's learned representation
    but source's learned physics.

    Neither input model is modified.
    """
    transplanted = copy.deepcopy(target_model)
    transplanted.dynamics.load_state_dict(
        copy.deepcopy(source_model.dynamics.state_dict())
    )
    return transplanted


@torch.no_grad()
def cross_predict_analysis(
    results: Dict[str, Dict],
    datasets: Dict[str, torch.Tensor],
    context_len: int,
    horizon: int,
    output_dir: str = "weights",
) -> pd.DataFrame:
    """
    Swap DCR dynamics weights between all pairs of trained models and measure
    the prediction quality degradation on each dataset.

    For every combination (dynamics_from, data_from):
      - If dynamics_from == data_from  → baseline (no transplant)
      - Otherwise                       → transplant source dynamics into a
                                          copy of the target model; evaluate
                                          on the target dataset

    rel_RMSE = cross_RMSE / baseline_RMSE
      ≈ 1 : dynamics are interchangeable (high exchangeability)
      > 1 : dynamics are dataset-specific (low exchangeability)

    Saves results to {output_dir}/cross_prediction.csv and returns the
    DataFrame.

    Args:
        results:      {dataset_name: run_experiment() return dict}
        datasets:     {dataset_name: X_seq [N, F, T]}
        context_len:  encoder look-back (must match training)
        horizon:      forecast horizon   (must match training)
    """
    rows = []
    names = list(results.keys())

    for data_name in names:
        own_model = results[data_name]["model"]
        X_seq = datasets[data_name]
        device = next(own_model.parameters()).device

        # Baseline: own model on own data (no transplant)
        base_m = evaluate(own_model, X_seq, context_len, horizon, device)
        baseline_rmse = base_m["RMSE"]
        rows.append({
            "dynamics_from": data_name,
            "data_from":     data_name,
            "baseline":      True,
            "MAE":           base_m["MAE"],
            "RMSE":          baseline_rmse,
            "rel_RMSE":      1.0,
        })

        for dyn_name in names:
            if dyn_name == data_name:
                continue
            src_model = results[dyn_name]["model"]
            # Keep target's encoder/decoder; swap in source's DCR physics
            transplanted = transplant_dynamics_weights(src_model, own_model)
            transplanted.eval()
            cross_m = evaluate(transplanted, X_seq, context_len, horizon, device)
            rows.append({
                "dynamics_from": dyn_name,
                "data_from":     data_name,
                "baseline":      False,
                "MAE":           cross_m["MAE"],
                "RMSE":          cross_m["RMSE"],
                "rel_RMSE":      cross_m["RMSE"] / (baseline_rmse + 1e-9),
            })

    df = pd.DataFrame(rows)
    out_path = Path(output_dir) / "cross_prediction.csv"
    df.to_csv(out_path, index=False)
    print(f"  [cross-pred] Saved cross-prediction results → {out_path}")
    return df


# ─── 9. run_experiment Helper ────────────────────────────────────────────────

def run_experiment(
    dataset_name: str,
    X_seq: torch.Tensor,            # [N, input_dim, T]
    adj: torch.Tensor,              # [N, N]
    hidden_dim: int = 32,
    context_len: int = 10,
    horizon: int = 5,
    n_epochs: int = 50,
    lr: float = 1e-3,
    omega_init: float = 0.5,
    delta_init: float = 0.5,
    attention: Literal["standard", "linear"] = "standard",
    reaction_type: Literal["source", "linear", "nonlinear"] = "nonlinear",
    ode_method: Literal["rk4", "euler"] = "rk4",
    output_dir: str = "weights",
    normalize_weights: bool = False,
    device: Optional[torch.device] = None,
    print_every: int = 10,
) -> Dict:
    """
    Train OPINN on one dataset, save weight matrices, return results.

    Args:
        X_seq:             [N, input_dim, T]  – full time series
        adj:               [N, N]             – adjacency matrix (CPU)
        normalize_weights: passed to save_weight_matrices

    Returns:
        dict with keys: 'dataset', 'model', 'MAE', 'RMSE',
                        'omega_learned', 'delta_learned'
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    N, F, T = X_seq.shape
    print(f"\n{'='*62}")
    print(f"  Experiment : {dataset_name}")
    print(f"  N={N} nodes  F={F} features  T={T} steps")
    print(f"  hidden_dim={hidden_dim}  attention={attention}  "
          f"reaction={reaction_type}")
    print(f"  ω_init={omega_init}  δ_init={delta_init}  "
          f"ode={ode_method}  device={device}")
    print(f"{'='*62}")

    model = OPINN(
        adj=adj,
        input_dim=F,
        hidden_dim=hidden_dim,
        omega_init=omega_init,
        delta_init=delta_init,
        attention=attention,
        reaction_type=reaction_type,
        ode_method=ode_method,
    ).to(device)

    optimizer = torch.optim.Adam(
        model.parameters(), lr=lr, weight_decay=5e-5
    )

    for epoch in range(1, n_epochs + 1):
        loss = train_epoch(
            model, optimizer, X_seq, context_len, horizon, device
        )
        if epoch % print_every == 0 or epoch == 1:
            metrics = evaluate(model, X_seq, context_len, horizon, device)
            g = model.get_gating_info()
            print(
                f"  epoch {epoch:3d}  loss={loss:.5f}  "
                f"MAE={metrics['MAE']:.4f}  RMSE={metrics['RMSE']:.4f}  "
                f"ω={g['omega_learned']:.3f}  δ={g['delta_learned']:.3f}"
            )

    final_metrics = evaluate(model, X_seq, context_len, horizon, device)
    g = model.get_gating_info()
    print(f"\n  Final  MAE={final_metrics['MAE']:.4f}  "
          f"RMSE={final_metrics['RMSE']:.4f}  "
          f"ω={g['omega_learned']:.3f}  δ={g['delta_learned']:.3f}")

    weight_matrices = save_weight_matrices(
        model, dataset_name, output_dir, normalize=normalize_weights
    )

    # ── Save full model checkpoint for inference (no retraining needed) ───────
    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    model_ckpt_path = out_path / f"{dataset_name}_model.pt"
    torch.save(model.state_dict(), model_ckpt_path)
    print(f"  [checkpoint] model state dict → {model_ckpt_path}")

    # Save adj so the model can be reconstructed for inference
    adj_ckpt_path = out_path / f"{dataset_name}_adj.pt"
    torch.save(adj.cpu(), adj_ckpt_path)

    # Save architecture config as JSON
    import json as _json
    config = {
        "input_dim":     F,
        "hidden_dim":    hidden_dim,
        "omega_init":    omega_init,
        "delta_init":    delta_init,
        "attention":     attention,
        "reaction_type": reaction_type,
        "ode_method":    ode_method,
        "context_len":   context_len,
        "horizon":       horizon,
    }
    config_path = out_path / f"{dataset_name}_config.json"
    with open(config_path, "w") as fh:
        _json.dump(config, fh, indent=2)
    print(f"  [checkpoint] model config     → {config_path}")

    return {
        "dataset":         dataset_name,
        "model":           model,
        "MAE":             final_metrics["MAE"],
        "RMSE":            final_metrics["RMSE"],
        "omega_learned":   g["omega_learned"],
        "delta_learned":   g["delta_learned"],
        "weight_matrices": weight_matrices,   # raw (un-normalised) tensors
    }


# ─── 10. __main__ Demo ───────────────────────────────────────────────────────

if __name__ == "__main__":
    torch.manual_seed(42)
    np.random.seed(42)

    # ── Shared graph: sparse Erdős-Rényi, N=20 ──────────────────────────────
    N = 20
    T = 120
    p_edge = 0.25
    A_np = (np.random.rand(N, N) < p_edge).astype(float)
    np.fill_diagonal(A_np, 0.0)
    A_np = np.maximum(A_np, A_np.T)        # symmetric
    adj = torch.tensor(A_np, dtype=torch.float32)

    # Row-normalized adjacency used for simulation (NOT the same as A_norm
    # inside the model, which uses symmetric normalization).
    deg = A_np.sum(1, keepdims=True).clip(1)
    A_rn = A_np / deg

    # ── Dataset 1: Consensus dynamics ───────────────────────────────────────
    # Opinions converge toward a shared mean over time (DeGroot-like).
    X_con_np = np.zeros((N, T))
    X_con_np[:, 0] = np.random.uniform(-1, 1, N)
    for t in range(1, T):
        X_con_np[:, t] = (
            0.7 * A_rn @ X_con_np[:, t - 1]
            + 0.3 * X_con_np[:, t - 1]
            + 0.02 * np.random.randn(N)
        )
    X_consensus = torch.tensor(X_con_np, dtype=torch.float32).unsqueeze(1)  # [N,1,T]

    # ── Dataset 2: Polarization dynamics ────────────────────────────────────
    # Two communities drift to opposite poles (+1 / -1).
    X_pol_np = np.zeros((N, T))
    X_pol_np[:N // 2, 0] = np.random.uniform(0.0, 0.5, N // 2)   # group A
    X_pol_np[N // 2:, 0] = np.random.uniform(-0.5, 0.0, N // 2)  # group B
    bias = np.concatenate([np.full(N // 2, 0.8), np.full(N // 2, -0.8)])
    for t in range(1, T):
        prev = X_pol_np[:, t - 1]
        X_pol_np[:, t] = np.clip(
            0.6 * A_rn @ prev + 0.3 * bias + 0.1 * prev + 0.02 * np.random.randn(N),
            -1.0, 1.0,
        )
    X_polarization = torch.tensor(X_pol_np, dtype=torch.float32).unsqueeze(1)  # [N,1,T]

    # ── Run experiments ──────────────────────────────────────────────────────
    CONTEXT = 12
    HORIZON = 6
    HIDDEN  = 16
    EPOCHS  = 40
    OUT_DIR = "weights"

    # Remove stale summary from previous runs so we get a clean table
    summary_file = Path(OUT_DIR) / "summary.csv"
    if summary_file.exists():
        summary_file.unlink()

    datasets = {
        "consensus":    X_consensus,
        "polarization": X_polarization,
    }

    results = {}
    for name, X_seq in datasets.items():
        results[name] = run_experiment(
            dataset_name=name,
            X_seq=X_seq,
            adj=adj,
            hidden_dim=HIDDEN,
            context_len=CONTEXT,
            horizon=HORIZON,
            n_epochs=EPOCHS,
            lr=5e-3,
            omega_init=0.5,
            delta_init=0.5,
            attention="standard",
            reaction_type="nonlinear",
            output_dir=OUT_DIR,
            normalize_weights=True,   # Frobenius-normalize for comparison
            print_every=10,
        )

    # ── Cross-dataset metrics summary ────────────────────────────────────────
    print(f"\n{'='*62}")
    print("  Cross-dataset comparison")
    print(f"  {'Dataset':<16} {'MAE':>8} {'RMSE':>8} "
          f"{'ω (learned)':>13} {'δ (learned)':>13}")
    print(f"  {'-'*60}")
    for name, r in results.items():
        print(f"  {name:<16} {r['MAE']:>8.4f} {r['RMSE']:>8.4f} "
              f"{r['omega_learned']:>13.4f} {r['delta_learned']:>13.4f}")

    # ── Post-training gating override demo ──────────────────────────────────
    print(f"\n{'='*62}")
    print("  Gating override sensitivity analysis (consensus model)")
    print(f"  {'Config':<28} {'MAE':>8} {'RMSE':>8}")
    print(f"  {'-'*46}")

    model_con = results["consensus"]["model"]
    device_con = next(model_con.parameters()).device

    scenarios = {
        "full model (learned)":       (None, None),
        "diffusion only (ω=1, δ=0)": (1.0,  0.0),
        "convection only (ω=0, δ=0)": (0.0,  0.0),
        "reaction only (δ=1)":        (0.5,  1.0),
    }
    for label, (w, d) in scenarios.items():
        model_con.set_gating_weights(omega=w, delta=d)
        m = evaluate(model_con, X_consensus, CONTEXT, HORIZON, device_con)
        print(f"  {label:<28} {m['MAE']:>8.4f} {m['RMSE']:>8.4f}")

    model_con.clear_gating_overrides()

    # ── Weight matrix Frobenius norms (per experiment) ───────────────────────
    print(f"\n{'='*62}")
    print("  Weight matrix Frobenius norms  (from weights/summary.csv)")
    print(f"{'='*62}")
    df_summary = pd.read_csv(summary_file)
    pivot = df_summary.pivot_table(
        index="matrix_name",
        columns="experiment",
        values="frobenius_norm",
        aggfunc="first",
    )
    print(pivot.to_string())

    # ── Pairwise weight-matrix distances ─────────────────────────────────────
    print(f"\n{'='*62}")
    print("  Pairwise weight-matrix distances")
    print("  (frob_dist_norm and cosine_dist are scale-invariant)")
    print(f"{'='*62}")
    experiment_matrices = {
        name: r["weight_matrices"] for name, r in results.items()
    }
    df_dist = save_weight_distances(experiment_matrices, output_dir=OUT_DIR)
    if not df_dist.empty:
        print(df_dist.to_string(index=False))

    # ── Cross-prediction exchangeability test ────────────────────────────────
    print(f"\n{'='*62}")
    print("  Cross-prediction: swap DCR dynamics between models")
    print("  rel_RMSE = cross_RMSE / baseline_RMSE")
    print("  (≈1 → exchangeable dynamics; >1 → dataset-specific physics)")
    print(f"{'='*62}")
    df_cross = cross_predict_analysis(
        results, datasets,
        context_len=CONTEXT,
        horizon=HORIZON,
        output_dir=OUT_DIR,
    )
    print(df_cross.to_string(index=False))
    print(f"\nFull per-matrix CSVs, distances, and cross-prediction saved in '{OUT_DIR}/'")
