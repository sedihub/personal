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
  --lr=1e-3 \
  --newton_alpha=1.0 \
  --newton_damping=1e-4

Supported optimizers: adam | adamw | sgd | newton
"""

from absl import flags
from absl import app
from absl import logging
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import math
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
    1e-4,
    "Tikhonov damping added to the diagonal of H before inversion: "
    "H_reg = H + damping * I.  Improves stability when H is near-singular.",
)
FLAGS = flags.FLAGS


# ---------------------------------------------------------------------------
# Center generator and plotting helper
# ---------------------------------------------------------------------------

def generate_points(n, max_margin_param=2.25) -> dict:
    """
    Generate n well-spaced points using Mitchell's best-candidate algorithm
    (Poisson disk sampling).
    """
    candidates = max(10, n * 5)
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
    """Inverse of softplus: raw = log(exp(r) - 1)."""
    return jnp.log(jnp.expm1(jnp.clip(radii, 1e-4, None)))


def raw_to_centers(raw_centers: jnp.ndarray) -> jnp.ndarray:
    """Center coordinates in (0, 1), shape (n, 2)."""
    return jax.nn.sigmoid(raw_centers)


def raw_to_radii(raw_radii: jnp.ndarray) -> jnp.ndarray:
    """Positive radii via softplus, shape (n,)."""
    return jax.nn.softplus(raw_radii)


# ---------------------------------------------------------------------------
# Overlap area helpers  (pure JAX, fully differentiable)
# ---------------------------------------------------------------------------

def _circle_circle_overlap(r1, r2, d):
    """Intersection area (lens) of two circles with radii r1, r2, separation d."""
    eps = 1e-6
    arg1 = jnp.clip((d*d + r1*r1 - r2*r2) / (2.0*d*r1 + eps), -1.0+eps, 1.0-eps)
    arg2 = jnp.clip((d*d + r2*r2 - r1*r1) / (2.0*d*r2 + eps), -1.0+eps, 1.0-eps)
    term1 = r1*r1 * jnp.arccos(arg1)
    term2 = r2*r2 * jnp.arccos(arg2)
    radicand = jnp.clip((-d+r1+r2)*(d-r1+r2)*(d+r1-r2)*(d+r1+r2), 0.0)
    term3 = 0.5 * jnp.sqrt(radicand)
    return term1 + term2 - term3


def _circle_wall_overlap(r, dist):
    """Area of circular segment cut off by a wall at distance `dist` from center."""
    eps = 1e-6
    arg = jnp.clip(dist / r, -1.0+eps, 1.0-eps)
    return r*r * jnp.arccos(arg) - dist * jnp.sqrt(jnp.clip(r*r - dist*dist, 0.0))


# ---------------------------------------------------------------------------
# Penalty functions
# ---------------------------------------------------------------------------

def circle_circle_penalty(centers: jnp.ndarray, radii: jnp.ndarray) -> jnp.ndarray:
    """Total pairwise circle-circle overlap area (vectorised over all pairs)."""
    diff = centers[:, None, :] - centers[None, :, :]       # (n, n, 2)
    d    = jnp.sqrt(jnp.sum(diff**2, axis=-1) + 1e-16)    # (n, n)
    ri, rj = radii[:, None], radii[None, :]
    overlap    = _circle_circle_overlap(ri, rj, d)
    overlapping = d < (ri + rj)
    n = radii.shape[0]
    upper = jnp.triu(jnp.ones((n, n), dtype=bool), k=1)
    return jnp.sum(jnp.where(overlapping & upper, overlap, 0.0))


def circle_boundary_penalty(centers: jnp.ndarray, radii: jnp.ndarray) -> jnp.ndarray:
    """Total circle-boundary overlap area for all four walls."""
    cx, cy = centers[:, 0], centers[:, 1]
    zero = jnp.zeros_like(radii)

    def wall_contrib(dist):
        return jnp.where(dist < radii, _circle_wall_overlap(radii, dist), zero)

    return (
        wall_contrib(cx) + wall_contrib(1.0 - cx) +
        wall_contrib(cy) + wall_contrib(1.0 - cy)
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
    penalty = circle_circle_penalty(centers, radii) + circle_boundary_penalty(centers, radii)
    return -sum_r + penalty_weight * penalty, penalty


# ---------------------------------------------------------------------------
# Optimizer factory
# ---------------------------------------------------------------------------

def make_optax_optimizer(name: str, lr: float) -> optax.GradientTransformation:
    """
    Return an optax GradientTransformation for the given name.

    Supported names: "adam", "adamw", "sgd".
    The Newton optimizer is handled separately and does not go through this
    factory.
    """
    name = name.lower()
    if name == "adam":
        return optax.adam(lr)
    elif name == "adamw":
        # weight_decay=1e-4 is a sensible default; can be exposed as a flag.
        return optax.adamw(lr, weight_decay=1e-4)
    elif name == "sgd":
        # Momentum=0.9 follows common practice; pure SGD would use 0.0.
        return optax.sgd(lr, momentum=0.9)
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
    """Reconstruct a parameter dict from a flat 1-D vector, using ref for shapes."""
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
) -> tuple:
    """
    One damped Newton step using the exact Hessian.

    The update rule is:

        p  ←  p  -  alpha * (H + damping * I)^{-1}  g

    where g is the gradient vector and H is the full Hessian matrix, both
    computed via JAX's forward-over-reverse autodiff (jax.hessian).

    Parameters
    ----------
    params        : dict with 'raw_centers' and 'raw_radii'
    penalty_weight: passed through to loss_fn
    alpha         : step-size multiplier (analogous to learning rate)
    damping       : Tikhonov regularisation on H for numerical stability

    Returns
    -------
    new_params : updated parameter dict
    loss       : scalar loss before the step
    penalty    : auxiliary penalty scalar before the step
    """
    # Work in a flat parameter space so the Hessian is a 2-D matrix.
    flat = _flatten_params(params)
    p    = flat.shape[0]

    def scalar_loss(flat_p: jnp.ndarray) -> jnp.ndarray:
        """Loss as a function of the flat parameter vector (aux dropped)."""
        p_dict = _unflatten_params(flat_p, params)
        return loss_fn(p_dict, penalty_weight)[0]

    # Gradient and Hessian via JAX autodiff.
    # jax.hessian uses forward-over-reverse, giving an exact p×p matrix.
    g = jax.grad(scalar_loss)(flat)             # (p,)
    H = jax.hessian(scalar_loss)(flat)          # (p, p)

    # Tikhonov damping: H_reg = H + damping * I
    H_reg = H + damping * jnp.eye(p)

    # Solve H_reg @ delta = g  (more numerically stable than explicit inversion)
    delta = jnp.linalg.solve(H_reg, g)          # (p,)

    new_flat   = flat - alpha * delta
    new_params = _unflatten_params(new_flat, params)

    # Recompute loss/penalty at the *original* params for logging consistency.
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
    penalty_weight: float = 1.0,
    learn_centers: bool = True,
    optimizer_name: str = "adam",
    n_steps: int = 2000,
    lr: float = 1e-2,
    newton_alpha: float = 1.0,
    newton_damping: float = 1e-4,
    log_every: int = 200,
) -> dict:
    """
    Run the chosen optimiser and return a result dictionary.

    Parameters
    ----------
    n                : number of circles (inferred from init_centers if None)
    init_centers     : initial center coordinates in [0, 1], shape (n, 2)
    init_radii       : initial radii, shape (n,)
    default_init_radius : fallback radius when init_radii is None
    penalty_weight   : weight for the overlap penalty terms
    learn_centers    : if False, centers are frozen and only radii are updated
    optimizer_name   : one of "adam" | "adamw" | "sgd" | "newton"
    n_steps          : number of optimisation steps
    lr               : learning rate for first-order optimisers (adam/adamw/sgd)
    newton_alpha     : step-size multiplier for Newton: p -= alpha * H^{-1} g
    newton_damping   : Tikhonov damping for Newton's Hessian: H + damping * I
    log_every        : print progress every this many steps (0 = silent)

    Returns
    -------
    dict with keys:
        radii       - final radii (jnp.ndarray, shape (n,))
        centers     - final centers (jnp.ndarray, shape (n, 2))
        sum_radii   - scalar sum of radii (float)
        loss_curve  - list of per-step loss values
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
        key   = jax.random.PRNGKey(0)
        unif  = jax.random.uniform(key, shape=(n, 2), minval=0.1, maxval=0.9)
        raw_c = centers_to_raw(unif)

    raw_r = (
        radii_to_raw(init_radii.astype(jnp.float32))
        if init_radii is not None
        else radii_to_raw(jnp.full((n,), default_init_radius, dtype=jnp.float32))
    )

    initial_centers = raw_to_centers(raw_c)
    params = {"raw_centers": raw_c, "raw_radii": raw_r}

    # For debug only!
    jax.debug.print("[JAX DEBUG] Initial Parameters: {params}", params=params)

    use_newton = optimizer_name.lower() == "newton"

    # ------------------------------------------------------------------
    # First-order optimiser setup (not used for Newton)
    # ------------------------------------------------------------------
    if not use_newton:
        optimizer = make_optax_optimizer(optimizer_name, lr)

        if not learn_centers:
            optimizer = optax.masked(
                optimizer,
                mask={"raw_centers": False, "raw_radii": True},
            )

        opt_state = optimizer.init(params)
        jax.debug.print("[JAX DEBUG] Initial Optimizer State: {state}", state=opt_state)

        @jax.jit
        def first_order_step(params, opt_state):
            (loss, penalty), grads = jax.value_and_grad(loss_fn, has_aux=True)(
                params, penalty_weight
            )

            # For debuging only!
            jax.debug.print("\n[JAX DEBUG] Loss: {loss}", loss=loss)
            jax.debug.print("[JAX DEBUG] Penalty: {penalty}", penalty=penalty)
            jax.debug.print("[JAX DEBUG] Gradients: {grads}\n", grads=grads)

            updates, new_opt_state = optimizer.update(grads, opt_state, params)
            new_params = optax.apply_updates(params, updates)

            # For debuging only!
            jax.debug.print("[JAX DEBUG] Optimizer State: {state}", state=opt_state)
            jax.debug.print("[JAX DEBUG] Optimizer Updates: {updates}", updates=updates)

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
                "[Newton] learn_centers=False is not supported for the Newton "
                "optimizer (centers would need a separate masked step). "
                "Proceeding with learn_centers=True."
            )

        @jax.jit
        def newton_step_jit(params):
            return newton_step(params, penalty_weight, newton_alpha, newton_damping)

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

        loss_curve.append(float(loss))

        # NaN guard
        if (
            jnp.isnan(loss)
            or jnp.any(jnp.isnan(raw_to_radii(params["raw_radii"])))
            or jnp.any(jnp.isnan(raw_to_centers(params["raw_centers"])))
        ):
            centers_np = raw_to_centers(params["raw_centers"])
            radii_np   = raw_to_radii(params["raw_radii"])
            print(f"{loss=}")
            print(f"{radii_np=}")
            print(f"{centers_np=}\n")
            prev_centers_np = raw_to_centers(prev_params["raw_centers"])
            prev_radii_np   = raw_to_radii(prev_params["raw_radii"])
            print(f"{prev_radii_np=}")
            print(f"{prev_centers_np=}")
            plot(
                int(radii_np.shape[0]),
                centers_np.tolist(),
                max_radius=radii_np.tolist(),
                filename=FLAGS.png_filename.replace(".png", "_final.png"),
            )
            raise ValueError("NaN detected, terminating training.")

        if log_every and i % log_every == 0:
            radii   = raw_to_radii(params["raw_radii"])
            centers = raw_to_centers(params["raw_centers"])
            sum_r   = float(radii.sum())
            pen     = float(penalty)
            delta   = float(jnp.linalg.norm(initial_centers - centers))
            print(
                f"\tStep {i:6d} | loss={float(loss):+.6f} | "
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
    n                = int(FLAGS.n)
    seed             = int(FLAGS.seed)
    png_filename     = FLAGS.png_filename
    initial_radius   = FLAGS.initial_radius
    max_margin_param = FLAGS.max_margin_param
    optimizer_name   = FLAGS.optimizer
    lr               = FLAGS.lr
    newton_alpha     = FLAGS.newton_alpha
    newton_damping   = FLAGS.newton_damping

    if seed is not None:
        random.seed(seed)

    centers = jnp.array(
        list(generate_points(n, max_margin_param).values()),
        dtype=jnp.float32,
    )
    print(f"{centers=}")

    plot(
        n,
        centers.tolist(),
        max_radius=[initial_radius] * n,
        filename=png_filename.replace(".png", "_initial.png"),
    )

    print("=" * 60)
    print(f"Circle packing optimisation – unit square, n={n}")
    print(f"Optimizer : {optimizer_name.upper()}")
    if optimizer_name.lower() == "newton":
        print(f"\talpha   : {newton_alpha}")
        print(f"\tdamping : {newton_damping}")
    else:
        print(f"\tlr      : {lr}")
    print("=" * 60)

    result = optimize(
        init_centers=centers,
        default_init_radius=initial_radius,
        penalty_weight=100.0,
        learn_centers=True,
        optimizer_name=optimizer_name,
        n_steps=20000,
        lr=lr,
        newton_alpha=newton_alpha,
        newton_damping=newton_damping,
        log_every=500,
    )

    import numpy as np
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