"""
circle_packing_jax.py
---------------------
Maximize the sum of radii of n circles with fixed centers inside a unit square,
penalizing circle–circle overlap and circle–boundary overlap.

CLI
------------
python3 ./circle_packing_jax.py \
  --n=26 \
  --seed=42 \
  --png_filename="__DELETE_ME__.png" \
  --initial_radius=0.01 \
  --max_margin_param=2.0 \
  --optimizer=adam \
  --lr=1e-4 \
  --newton_alpha=1.0 \
  --newton_damping=1.0 \
  --newton_max_delta=1.0 \
  --penalty_weight=100.0 \
  --n_steps=10000 \
  --log_every=500 \
  --learn_centers=true \
  --sgd_momentum=0.9 \
  --adamw_weight_decay=1e-4 \
  --candidates_multiplier=5 \
  --hex_init=false

Supported optimizers: adam | adamw | sgd | newton
"""

from absl import flags
from absl import app
from absl import logging
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import math
import numpy as np
import random
import jax
import jax.numpy as jnp
import optax
import sys
from typing import Optional


# # Comment out after debugging:
# jax.config.update('jax_disable_jit', True)


flags.DEFINE_integer("n", 26, "Number of circles.")
flags.DEFINE_integer("seed", 42, "RNG seed.")
flags.DEFINE_string("png_filename", "", "PNG filename to save plot to.")
flags.DEFINE_float("initial_radius", 0.01, "Initial value for radii of the circles.")
flags.DEFINE_float(
    "max_margin_param",
    2.0,
    "A parameter that controls the distribution of centers margins from the boundaries.",
)
flags.DEFINE_enum(
    "optimizer",
    "adam",
    ["adam", "adamw", "sgd", "newton"],
    "Optimizer to use: adam | adamw | sgd | newton.",
)
flags.DEFINE_float("lr", 1e-4, "Learning rate (step size) for first-order optimizers.")
flags.DEFINE_float(
    "newton_alpha",
    1.0,
    "Step-size multiplier alpha for the Newton optimizer: "
    "params -= alpha * H^{-1} g.",
)
flags.DEFINE_float(
    "newton_damping",
    1.0,
    "Tikhonov damping added to the diagonal of H before inversion: "
    "H_reg = H + damping * I.  Improves stability when H is near-singular. "
    "Recommended range: 0.1–10.0 when penalty_weight=100.",
)
flags.DEFINE_float(
    "newton_max_delta",
    1.0,
    "Per-parameter clamp on the Newton step: delta = clip(H^{-1} g, "
    "[-max_delta, max_delta]).  Prevents large jumps when the Hessian is "
    "poorly conditioned early in training.",
)
flags.DEFINE_float(
    "penalty_weight",
    100.0,
    "Weight applied to circle–circle and circle–boundary overlap penalties "
    "in the loss: loss = -sum(radii) + penalty_weight * overlap.",
)
flags.DEFINE_integer(
    "n_steps",
    10000,
    "Total number of optimisation steps.",
)
flags.DEFINE_integer(
    "log_every",
    500,
    "Print a progress line every this many steps. 0 = silent.",
)
flags.DEFINE_bool(
    "learn_centers",
    True,
    "If False, circle centers are frozen and only radii are optimised "
    "(first-order optimizers only; Newton always learns centers).",
)
flags.DEFINE_float(
    "sgd_momentum",
    0.9,
    "Momentum coefficient for the SGD optimizer.",
)
flags.DEFINE_float(
    "adamw_weight_decay",
    1e-4,
    "Weight-decay (L2 regularisation) coefficient for the AdamW optimizer.",
)
flags.DEFINE_integer(
    "candidates_multiplier",
    5,
    "Controls the number of candidate points sampled per circle during "
    "Mitchell's best-candidate (Poisson disk) initialisation: "
    "candidates = max(10, n * candidates_multiplier).",
)
flags.DEFINE_bool(
    "hex_init",
    False,
    "If True, initialise circle centers on a regular hexagonal grid fitted "
    "inside the unit square, instead of the default Mitchell best-candidate "
    "(Poisson disk) sampling.  Exactly n centers are selected by filling hex "
    "rows from the bottom-left and discarding excess points.",
)
FLAGS = flags.FLAGS


# ---------------------------------------------------------------------------
# Center generator and plotting helper
# ---------------------------------------------------------------------------

def generate_points(n, max_margin_param=2.25, candidates_multiplier=5) -> dict:
    """
    Generate n well-spaced points using Mitchell's best-candidate algorithm
    (Poisson disk sampling).
    """
    candidates = max(10, n * candidates_multiplier)
    points = {}

    max_marg = max_margin_param * math.sqrt(
        1.0 / ((math.sqrt(3) + math.pi) * n)
    )
    rand_marg = random.uniform(0, max_marg)

    def min_dist_to_existing(x, y):
        if not points:
            return float("inf")
        return min(
            math.sqrt((x - px) ** 2 + (y - py) ** 2)
            for px, py in points.values()
        )

    for i in range(n):
        best_pt, best_dist = None, -1.0
        for _ in range(candidates):
            cx = random.uniform(rand_marg, 1.0 - rand_marg)
            cy = random.uniform(rand_marg, 1.0 - rand_marg)
            d = min_dist_to_existing(cx, cy)
            if d > best_dist:
                best_dist = d
                best_pt = (cx, cy)
        points[i] = best_pt

    return points


def generate_hex_points(n) -> dict:
    """
    Generate n points on a regular hexagonal grid fitted inside the unit square.

    Layout
    ------
    A hex grid uses two interleaved rectangular sub-grids, offset by half a
    column spacing horizontally and half a row spacing vertically:

        Row 0 (even):  x = margin + 0·dx,  margin + 1·dx,  margin + 2·dx, ...
        Row 1 (odd):   x = margin + 0.5·dx, margin + 1.5·dx, ...
        Row 2 (even):  same as row 0, ...

    The vertical spacing between adjacent rows in a regular hex grid is
    dy = dx * sqrt(3) / 2, which keeps all nearest-neighbour distances equal
    to dx.

    Grid sizing
    -----------
    We want to pack at least n points inside [0, 1]² with a uniform margin on
    all four sides.  A good approximation for the spacing is derived from the
    area of the unit square:

        n ≈ cols * rows,  with rows ≈ cols * sqrt(3)/2
        → cols ≈ sqrt(n * 2/sqrt(3))

    We round cols up, recompute rows so that cols*rows >= n, then fit the grid
    symmetrically with equal margins on opposite sides:

        margin_x = (1 - (cols-1)*dx) / 2
        margin_y = (1 - (rows-1)*dy) / 2

    Points are enumerated left-to-right, bottom-to-top; the first n are kept.
    This gives a centred hex lattice with consistent spacing regardless of n.

    Parameters
    ----------
    n : int
        Number of center points to return.

    Returns
    -------
    dict mapping index -> (x, y) for indices 0 … n-1.
    """
    if n <= 0:
        return {}
    if n == 1:
        return {0: (0.5, 0.5)}

    # Estimate grid dimensions so that cols * rows >= n.
    cols = math.ceil(math.sqrt(n * 2.0 / math.sqrt(3)))
    rows = math.ceil(n / cols)
    # Expand until the grid has at least n cells.
    while cols * rows < n:
        cols += 1

    # Spacing between adjacent columns; rows are dy = dx*sqrt(3)/2 apart.
    dx = 1.0 / max(cols - 1, 1)
    dy = dx * math.sqrt(3) / 2.0

    # If the rows would overflow the unit square, shrink dx until they fit.
    grid_height = (rows - 1) * dy
    if grid_height > 1.0:
        dy = 1.0 / max(rows - 1, 1)
        dx = dy * 2.0 / math.sqrt(3)

    # Centre the grid inside [0, 1]².
    grid_width = (cols - 1) * dx
    margin_x = (1.0 - grid_width) / 2.0
    margin_y = (1.0 - (rows - 1) * dy) / 2.0

    # Enumerate all grid points, keeping the first n.
    all_pts = []
    for row in range(rows):
        x_offset = 0.5 * dx if (row % 2 == 1) else 0.0
        for col in range(cols):
            x = margin_x + col * dx + x_offset
            y = margin_y + row * dy
            # Odd rows have one fewer column to stay inside the unit square.
            if x <= 1.0 + 1e-9:
                all_pts.append((x, y))

    # Sort bottom-to-top, left-to-right for a deterministic ordering, then
    # take the first n.  If the grid produced fewer than n valid points
    # (can happen for very small n with the odd-row trim), raise clearly.
    all_pts.sort(key=lambda p: (round(p[1] / dy), p[0]))
    if len(all_pts) < n:
        raise ValueError(
            f"Hex grid produced only {len(all_pts)} points but n={n} requested. "
            "This should not happen; please file a bug."
        )

    return {i: all_pts[i] for i in range(n)}


def plot(n: int, points: list, max_radius: list, filename: str = "result.png"):
    fig, ax = plt.subplots(figsize=(7, 7))
    box = patches.Rectangle(
        (0, 0), 1, 1, linewidth=2, edgecolor="black", facecolor="lightyellow", zorder=0
    )
    ax.add_patch(box)

    cmap = plt.cm.tab20
    for i in range(n):
        x, y = points[i]
        r = max_radius[i]
        color = cmap(i % 20)
        circle = plt.Circle(
            (x, y), r, facecolor=color, alpha=0.4, linewidth=1.2, edgecolor=color, zorder=2
        )
        ax.add_patch(circle)
        ax.plot(x, y, "o", color=color, markersize=4, zorder=3)
        ax.annotate(
            str(i), (x, y),
            textcoords="offset points", xytext=(4, 4),
            fontsize=7, color=color, zorder=4,
        )

    total = round(float(sum(max_radius)), 5)
    ax.set_xlim(-0.05, 1.05)
    ax.set_ylim(-0.05, 1.05)
    ax.set_aspect("equal")
    ax.set_title(f"Circle Packing in Unit Square  |  Sum of radii = {total}", fontsize=12)
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    plt.tight_layout()
    plt.axis("off")
    plt.savefig(filename, bbox_inches="tight", dpi=150)
    plt.close()
    print(f"\nPlot saved to '{filename}'")


# ---------------------------------------------------------------------------
# Parameterisation helpers
# ---------------------------------------------------------------------------

def centers_to_raw(centers: jnp.ndarray) -> jnp.ndarray:
    """Inverse of sigmoid, mapping (0, 1) -> R."""
    c = jnp.clip(centers, 1e-6, 1.0 - 1e-6)
    return jnp.log(c / (1.0 - c))


def radii_to_raw(radii: jnp.ndarray) -> jnp.ndarray:
    """Inverse of softplus: raw = log(exp(r) - 1).

    Fix: clip the expm1 result away from zero before taking log to prevent
    log(0) = -inf when radii are very small.
    """
    return jnp.log(jnp.clip(jnp.expm1(jnp.clip(radii, 1e-4, None)), 1e-10, None))


def raw_to_centers(raw_centers: jnp.ndarray) -> jnp.ndarray:
    """Center coordinates in (0, 1), shape (n, 2)."""
    return jax.nn.sigmoid(raw_centers)


def raw_to_radii(raw_radii: jnp.ndarray) -> jnp.ndarray:
    """Positive radii via softplus, shape (n,)."""
    return jax.nn.softplus(raw_radii)


# ---------------------------------------------------------------------------
# Overlap area helpers  (pure JAX, NaN-safe gradients)
# ---------------------------------------------------------------------------

def _circle_circle_overlap(r1, r2, d_raw):
    """Intersection area (lens area) of two circles with radii r1, r2.

    NaN-safety strategy
    -------------------
    JAX traces *both* branches of every jnp.where before applying the mask,
    so a NaN or Inf produced in a "logically dead" branch (e.g. d == 0 on
    the diagonal, or dist == r at the wall) still poisons the gradient.

    There are three dangerous sites in this function:

    1. Division by d (or d*r1, d*r2) when d == 0 (coincident centers).
       Fix: clamp d to d_safe = max(d, eps) before any division.
       The diagonal / same-circle entries are excluded by the upper-triangular
       mask in circle_circle_penalty, so the safe value never enters the result.

    2. arccos(x) when |x| >= 1: gradient is -1/sqrt(1-x^2) -> ±Inf.
       Fix: clip arccos arguments to [-1+eps, 1-eps].

    3. sqrt(radicand) when radicand == 0: gradient is 1/(2*sqrt) -> Inf.
       Fix: clamp radicand to a small positive floor eps^2 so the gradient
       of sqrt stays bounded.  The forward value at the floor is tiny
       (O(eps)), which is negligible compared to the other terms.
    """
    eps = 1e-6
    # (1) Safe distance: avoids 0/0 in arccos arguments and denominators.
    d = jnp.maximum(d_raw, eps)

    arg1 = jnp.clip((d*d + r1*r1 - r2*r2) / (2.0*d*r1), -1.0 + eps, 1.0 - eps)
    arg2 = jnp.clip((d*d + r2*r2 - r1*r1) / (2.0*d*r2), -1.0 + eps, 1.0 - eps)

    term1 = r1*r1 * jnp.arccos(arg1)
    term2 = r2*r2 * jnp.arccos(arg2)

    # (3) Clamp radicand away from 0 so sqrt gradient is finite.
    radicand = (-d + r1 + r2) * (d - r1 + r2) * (d + r1 - r2) * (d + r1 + r2)
    term3 = 0.5 * jnp.sqrt(jnp.maximum(radicand, eps * eps))

    return term1 + term2 - term3


def _circle_wall_overlap(r, dist_raw):
    """Area of circular segment cut off by a wall at distance dist from center.

    NaN-safety strategy
    -------------------
    Two dangerous sites:

    4. arccos(dist/r) when dist >= r: argument >= 1, gradient -> -Inf.
       Fix: clamp dist to dist_safe = min(dist, r - eps) before division.
       The outer jnp.where in circle_boundary_penalty zeros the result
       whenever dist >= r, so the safe substitution only affects the dead
       branch and never changes the real answer.

    5. sqrt(r^2 - dist^2) when dist -> r: value -> 0, gradient -> Inf.
       Fix: clamp the radicand to at least eps^2.
    """
    eps = 1e-6
    # (4) Safe distance: keeps arccos argument strictly inside (-1, 1).
    dist = jnp.minimum(dist_raw, r - eps)

    arg        = jnp.clip(dist / r, -1.0 + eps, 1.0 - eps)
    # (5) Clamp radicand away from 0 so sqrt gradient is finite.
    under_sqrt = jnp.maximum(r*r - dist*dist, eps * eps)

    return r*r * jnp.arccos(arg) - dist * jnp.sqrt(under_sqrt)


# ---------------------------------------------------------------------------
# Penalty functions
# ---------------------------------------------------------------------------

def circle_circle_penalty(centers: jnp.ndarray, radii: jnp.ndarray) -> jnp.ndarray:
    """Total pairwise circle-circle overlap area (vectorised, NaN-safe).

    The pairwise distance d is computed as sqrt(||diff||^2 + eps) rather than
    sqrt(||diff||^2).  This keeps the gradient of d w.r.t. centers finite
    even when two centers coincide (d == 0 -> gradient would be 0/0 = NaN
    without the eps).  The error introduced is O(sqrt(eps)) ~ 1e-6, which
    is negligible relative to the circle geometry.
    """
    diff = centers[:, None, :] - centers[None, :, :]           # (n, n, 2)
    d    = jnp.sqrt(jnp.sum(diff**2, axis=-1) + 1e-12)        # (n, n) — eps inside sqrt

    ri, rj = radii[:, None], radii[None, :]
    overlap     = _circle_circle_overlap(ri, rj, d)
    # nan_to_num is the last-resort guard: even with all the clamping above,
    # jnp.where evaluates *both* branches and backpropagates through both.
    # Any residual NaN/Inf in the masked-off branch (e.g. non-overlapping
    # pairs) would still corrupt the gradient.  Replacing them with 0 before
    # the where-mask is applied makes the dead branch genuinely inert.
    overlap     = jnp.nan_to_num(overlap, nan=0.0, posinf=0.0, neginf=0.0)
    overlapping = d < (ri + rj)
    n_circles   = radii.shape[0]
    upper       = jnp.triu(jnp.ones((n_circles, n_circles), dtype=bool), k=1)

    return jnp.sum(jnp.where(overlapping & upper, overlap, 0.0))


def circle_boundary_penalty(centers: jnp.ndarray, radii: jnp.ndarray) -> jnp.ndarray:
    """Total circle-boundary overlap area for all four walls (NaN-safe).

    Safe because _circle_wall_overlap clamps dist to min(dist, r-eps) and
    the radicand to eps^2, so the formula is finite even in the masked
    (no-overlap) branch that jnp.where still eagerly evaluates.
    """
    cx, cy = centers[:, 0], centers[:, 1]

    def wall_contrib(dist):
        overlap = _circle_wall_overlap(radii, dist)
        # Same defence as in circle_circle_penalty: sanitise before where so
        # the dead branch (circle does not reach the wall) cannot contribute
        # a NaN or Inf to the gradient.
        overlap = jnp.nan_to_num(overlap, nan=0.0, posinf=0.0, neginf=0.0)
        return jnp.where(dist < radii, overlap, 0.0)

    return (
        wall_contrib(cx)       +   # x = 0 wall
        wall_contrib(1.0 - cx) +   # x = 1 wall
        wall_contrib(cy)       +   # y = 0 wall
        wall_contrib(1.0 - cy)     # y = 1 wall
    ).sum()


# ---------------------------------------------------------------------------
# Loss function
# ---------------------------------------------------------------------------

def loss_fn(params: dict, penalty_weight: float = 1.0):
    """
    Loss = -sum(radii) + penalty_weight * (circle-circle + boundary penalties)

    Returns (loss, penalty) where penalty is an auxiliary scalar.
    """
    centers = raw_to_centers(params["raw_centers"])
    radii   = raw_to_radii(params["raw_radii"])
    sum_r   = radii.sum()
    penalty = (circle_circle_penalty(centers, radii) +
               circle_boundary_penalty(centers, radii))
    return -sum_r + penalty_weight * penalty, penalty


# ---------------------------------------------------------------------------
# Optimizer factory
# ---------------------------------------------------------------------------

def make_optax_optimizer(
    name: str,
    lr: float,
    sgd_momentum: float = 0.9,
    adamw_weight_decay: float = 1e-4,
) -> optax.GradientTransformation:
    """Return an optax GradientTransformation for adam / adamw / sgd."""
    name = name.lower()
    if name == "adam":
        return optax.adam(lr)
    elif name == "adamw":
        return optax.adamw(lr, weight_decay=adamw_weight_decay)
    elif name == "sgd":
        return optax.sgd(lr, momentum=sgd_momentum)
    else:
        raise ValueError(
            f"Unknown first-order optimizer '{name}'. "
            "Choose from: adam | adamw | sgd | newton"
        )


# ---------------------------------------------------------------------------
# Newton optimizer step
# ---------------------------------------------------------------------------

def _flatten_params(params: dict) -> jnp.ndarray:
    """Flatten the parameter dict into a single 1-D vector."""
    return jnp.concatenate([params["raw_centers"].ravel(), params["raw_radii"].ravel()])


def _unflatten_params(flat: jnp.ndarray, ref: dict) -> dict:
    """Reconstruct a parameter dict from a flat vector using ref for shapes."""
    nc = ref["raw_centers"].size
    return {
        "raw_centers": flat[:nc].reshape(ref["raw_centers"].shape),
        "raw_radii":   flat[nc:].reshape(ref["raw_radii"].shape),
    }


def newton_step(
    params: dict,
    penalty_weight: float,
    alpha: float,
    damping: float,
    max_delta: float,
) -> tuple:
    """
    One damped Newton step using the exact Hessian.

    Update rule:
        p  ←  p  -  alpha * (H + damping * I)^{-1}  g

    where g = ∇loss  and  H = ∇²loss  are computed via JAX autodiff.

    Stability notes
    ---------------
    * The loss is NaN-free (see overlap helpers above), so g and H are also
      NaN-free.
    * `damping` regularises H against near-singularity.  Because the penalty
      weight is 100, the Hessian entries are O(100); damping should be in the
      same ballpark (default 1.0 rather than the previous 1e-4).
    * `jnp.linalg.solve` is used instead of explicit inversion.
    * The solved delta is clipped to [-max_delta, max_delta] per parameter to
      prevent a single Newton step from jumping across the entire parameter
      space when the Hessian is still poorly conditioned early in training.
    """
    flat = _flatten_params(params)
    p    = flat.shape[0]

    def scalar_loss(flat_p: jnp.ndarray) -> jnp.ndarray:
        return loss_fn(_unflatten_params(flat_p, params), penalty_weight)[0]

    # Gradient and Hessian via JAX autodiff.
    # jax.hessian uses forward-over-reverse, giving an exact p×p matrix.
    g = jax.grad(scalar_loss)(flat)             # (p,)
    H = jax.hessian(scalar_loss)(flat)          # (p, p)

    H_reg = H + damping * jnp.eye(p)                   # Tikhonov damping
    delta = jnp.linalg.solve(H_reg, g)                 # (p,)

    # Safeguard: cap each component so one step cannot move more than max_delta
    # in raw-parameter space (sigmoid/softplus map this to a bounded real change).
    delta = jnp.clip(delta, -max_delta, max_delta)

    new_flat   = flat - alpha * delta
    new_params = _unflatten_params(new_flat, params)

    loss, penalty = loss_fn(params, penalty_weight)
    return new_params, loss, penalty


# ---------------------------------------------------------------------------
# Optimisation loop
# ---------------------------------------------------------------------------

def optimize(
    n: Optional[int] = None,
    init_centers: Optional[jnp.ndarray] = None,
    init_radii: Optional[jnp.ndarray] = None,
    default_init_radius: float = 0.01,
    penalty_weight: float = 100.0,
    learn_centers: bool = True,
    optimizer_name: str = "adam",
    n_steps: int = 10000,
    lr: float = 1e-2,
    newton_alpha: float = 1.0,
    newton_damping: float = 1.0,
    newton_max_delta: float = 1.0,
    sgd_momentum: float = 0.9,
    adamw_weight_decay: float = 1e-4,
    log_every: int = 500,
) -> dict:
    """
    Run the chosen optimiser and return a result dictionary.

    Parameters
    ----------
    n                   : number of circles (inferred from init_centers if None)
    init_centers        : initial center coordinates in [0, 1], shape (n, 2)
    init_radii          : initial radii, shape (n,)
    default_init_radius : fallback radius when init_radii is None
    penalty_weight      : weight for the overlap penalty terms
    learn_centers       : if False, centers are frozen (first-order only)
    optimizer_name      : "adam" | "adamw" | "sgd" | "newton"
    n_steps             : number of optimisation steps
    lr                  : learning rate for adam / adamw / sgd
    newton_alpha        : step-size multiplier for Newton
    newton_damping      : Tikhonov damping for Newton Hessian (H + damping*I)
    newton_max_delta    : per-parameter Newton step clamp
    sgd_momentum        : momentum coefficient for SGD
    adamw_weight_decay  : weight-decay coefficient for AdamW
    log_every           : print progress every this many steps (0 = silent)

    Returns
    -------
    dict with keys: radii, centers, sum_radii, loss_curve
    """
    # ------------------------------------------------------------------
    # Initialise raw parameters
    # ------------------------------------------------------------------
    if init_centers is not None:
        if n is None:
            n = init_centers.shape[0]
        raw_c = centers_to_raw(init_centers.astype(jnp.float32))
    else:
        if n is None:
            raise ValueError("Provide either `n` or `init_centers`.")
        key  = jax.random.PRNGKey(0)
        unif = jax.random.uniform(key, shape=(n, 2), minval=0.1, maxval=0.9)
        raw_c = centers_to_raw(unif)

    raw_r = (
        radii_to_raw(init_radii.astype(jnp.float32))
        if init_radii is not None
        else radii_to_raw(jnp.full((n,), default_init_radius, dtype=jnp.float32))
    )

    initial_centers = raw_to_centers(raw_c)
    params = {"raw_centers": raw_c, "raw_radii": raw_r}

    use_newton = optimizer_name.lower() == "newton"

    # ------------------------------------------------------------------
    # First-order optimiser setup (not used for Newton)
    # ------------------------------------------------------------------
    if not use_newton:
        optimizer = make_optax_optimizer(
            optimizer_name, lr,
            sgd_momentum=sgd_momentum,
            adamw_weight_decay=adamw_weight_decay,
        )

        if not learn_centers:
            optimizer = optax.masked(
                optimizer,
                mask={"raw_centers": False, "raw_radii": True},
            )

        opt_state = optimizer.init(params)
        # jax.debug.print("[JAX DEBUG] Initial Optimizer State: {state}", state=opt_state)

        @jax.jit
        def first_order_step(params, opt_state):
            (loss, penalty), grads = jax.value_and_grad(loss_fn, has_aux=True)(
                params, penalty_weight
            )

            # # For debuging only!
            # jax.debug.print("\n[JAX DEBUG] Loss: {loss}", loss=loss)
            # jax.debug.print("[JAX DEBUG] Penalty: {penalty}", penalty=penalty)
            # jax.debug.print("[JAX DEBUG] Gradients: {grads}\n", grads=grads)

            updates, new_opt_state = optimizer.update(grads, opt_state, params)
            new_params = optax.apply_updates(params, updates)

            # # For debuging only!
            # jax.debug.print("[JAX DEBUG] Optimizer State: {state}", state=opt_state)
            # jax.debug.print("[JAX DEBUG] Optimizer Updates: {updates}", updates=updates)

            return new_params, new_opt_state, loss, penalty

    # ------------------------------------------------------------------
    # Newton setup — JIT the step for speed.
    # Note: jax.hessian traces through the full computation graph, so the
    # compiled XLA program contains the Hessian computation.  For large n
    # this is O(p^2) memory and O(p^3) compute per step (p = 3n params).
    # ------------------------------------------------------------------
    else:
        if not learn_centers:
            print(
                "[Newton] learn_centers=False is not supported; "
                "proceeding with learn_centers=True."
            )

        @jax.jit
        def newton_step_jit(params):
            return newton_step(
                params, penalty_weight, newton_alpha, newton_damping, newton_max_delta
            )

    # ------------------------------------------------------------------
    # Main loop
    # ------------------------------------------------------------------
    loss_curve = []

    for i in range(1, n_steps + 1):
        prev_params = params

        if use_newton:
            params, loss, penalty = newton_step_jit(params)
        else:
            params, opt_state, loss, penalty = first_order_step(params, opt_state)

        loss_val = float(loss)
        loss_curve.append(loss_val)

        # NaN guard — report which tensor went bad to aid debugging.
        radii_now   = raw_to_radii(params["raw_radii"])
        centers_now = raw_to_centers(params["raw_centers"])
        if (
            math.isnan(loss_val)
            or bool(jnp.any(jnp.isnan(radii_now)))
            or bool(jnp.any(jnp.isnan(centers_now)))
        ):
            bad = []
            if math.isnan(loss_val):                        bad.append("loss")
            if bool(jnp.any(jnp.isnan(radii_now))):         bad.append("radii")
            if bool(jnp.any(jnp.isnan(centers_now))):       bad.append("centers")
            print(f"[NaN] step {i}, bad tensors: {bad}")

            # Plot last good state.
            prev_centers = raw_to_centers(prev_params["raw_centers"])
            prev_radii   = raw_to_radii(prev_params["raw_radii"])
            plot(
                int(prev_radii.shape[0]),
                prev_centers.tolist(),
                max_radius=prev_radii.tolist(),
                filename=FLAGS.png_filename.replace(".png", "_nan_checkpoint.png"),
            )
            raise ValueError(
                f"NaN detected at step {i} in: {bad}. "
                "Last-good-state plot saved to *_nan_checkpoint.png."
            )

        if log_every and i % log_every == 0:
            sum_r = float(radii_now.sum())
            pen   = float(penalty)
            delta = float(jnp.linalg.norm(initial_centers - centers_now))
            print(
                f"\tStep {i:6d} | loss={loss_val:+.6f} | "
                f"sum_r={sum_r:.6f} | penalty={pen:.6f} | "
                f"δ centers={delta:.6f}"
            )

    final_radii   = raw_to_radii(params["raw_radii"])
    final_centers = raw_to_centers(params["raw_centers"])

    return {
        "radii":      final_radii,
        "centers":    final_centers,
        "sum_radii":  float(final_radii.sum()),
        "loss_curve": loss_curve,
    }


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main(argv):
    """Main function."""
    n                    = int(FLAGS.n)
    seed                 = int(FLAGS.seed)
    png_filename         = FLAGS.png_filename
    initial_radius       = FLAGS.initial_radius
    max_margin_param     = FLAGS.max_margin_param
    optimizer_name       = FLAGS.optimizer
    lr                   = FLAGS.lr
    newton_alpha         = FLAGS.newton_alpha
    newton_damping       = FLAGS.newton_damping
    newton_max_delta     = FLAGS.newton_max_delta
    penalty_weight       = FLAGS.penalty_weight
    n_steps              = FLAGS.n_steps
    log_every            = FLAGS.log_every
    learn_centers        = FLAGS.learn_centers
    sgd_momentum         = FLAGS.sgd_momentum
    adamw_weight_decay   = FLAGS.adamw_weight_decay
    candidates_multiplier = FLAGS.candidates_multiplier
    hex_init              = FLAGS.hex_init

    if seed is not None:
        random.seed(seed)

    if hex_init:
        point_dict = generate_hex_points(n)
        init_method = "hex grid"
    else:
        point_dict = generate_points(n, max_margin_param, candidates_multiplier)
        init_method = "Mitchell best-candidate (Poisson disk)"

    centers = jnp.array(list(point_dict.values()), dtype=jnp.float32)
    print(f"Initialisation: {init_method}")
    print(f"{centers=}")

    plot(
        n,
        centers.tolist(),
        max_radius=[initial_radius] * n,
        filename=png_filename.replace(".png", "_initial.png"),
    )

    print("=" * 60)
    print(f"Circle packing optimisation – unit square, n={n}")
    print(f"Initialisation: {init_method}")
    print(f"Optimizer     : {optimizer_name.upper()}")
    print(f"penalty_weight: {penalty_weight}")
    print(f"n_steps       : {n_steps}")
    print(f"learn_centers : {learn_centers}")
    if optimizer_name.lower() == "newton":
        print(f"  alpha       : {newton_alpha}")
        print(f"  damping     : {newton_damping}")
        print(f"  max_delta   : {newton_max_delta}")
    elif optimizer_name.lower() == "sgd":
        print(f"  lr          : {lr}")
        print(f"  momentum    : {sgd_momentum}")
    elif optimizer_name.lower() == "adamw":
        print(f"  lr          : {lr}")
        print(f"  weight_decay: {adamw_weight_decay}")
    else:
        print(f"  lr          : {lr}")
    print("=" * 60)

    result = optimize(
        init_centers=centers,
        default_init_radius=initial_radius,
        penalty_weight=penalty_weight,
        learn_centers=learn_centers,
        optimizer_name=optimizer_name,
        n_steps=n_steps,
        lr=lr,
        newton_alpha=newton_alpha,
        newton_damping=newton_damping,
        newton_max_delta=newton_max_delta,
        sgd_momentum=sgd_momentum,
        adamw_weight_decay=adamw_weight_decay,
        log_every=log_every,
    )

    print("\nFinal radii  :", np.array(result["radii"]).round(6))
    print("Final Centers:", np.array(result["centers"]).round(6))
    print("\nSum of radii :", round(result["sum_radii"], 6))

    plot(
        n,
        result["centers"].tolist(),
        max_radius=result["radii"].tolist(),
        filename=png_filename.replace(".png", "_final.png"),
    )


if __name__ == "__main__":
    try:
        app.run(main)
    except Exception:
        # exc_info=True forces the full traceback to print to stderr
        logging.exception("Program terminated with an error:", exc_info=True)
        sys.exit(1)