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
  --max_margin_param=2.0
"""

from absl import flags
from absl import app
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import math
import random
import jax
import jax.numpy as jnp
import optax
from typing import Optional, NamedTuple


flags.DEFINE_integer(
    "n",
    26,
    "Number of circles.",
)
flags.DEFINE_integer(
    "seed",
    42,
    "RNG seed.",
)
flags.DEFINE_string(
    "png_filename",
    "",
    "PNG filename to save plot to.",
)
flags.DEFINE_float(
    "initial_radius",
    0.01,
    "Initial value for radii of the circles.",
)
flags.DEFINE_float(
    "max_margin_param",
    2.0,
    "A parameter that controls the distribution "
    "of centers margins from the boundaries.",
)
FLAGS = flags.FLAGS


# ---------------------------------------------------------------------------
# Center generator and plotting helper
# ---------------------------------------------------------------------------

def generate_points(n, max_margin_param=2.25) -> dict:
    """
    Generate n well-spaced points using Mitchell's best-candidate algorithm
    (Poisson disk sampling). For each new point, draw `candidates` random
    proposals and keep the one furthest from all existing points.
    """
    candidates = max(10, n * 5)
    points = {}

    # Margin: rand_marg drawn from Uniform[0, max_marg]
    # max_marg derived from expected packing radius for n points
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
        best_pt   = None
        best_dist = -1.0
        for _ in range(candidates):
            cx = random.uniform(rand_marg, 1.0 - rand_marg)
            cy = random.uniform(rand_marg, 1.0 - rand_marg)
            d = min_dist_to_existing(cx, cy)
            if d > best_dist:
                best_dist = d
                best_pt   = (cx, cy)
        points[i] = best_pt

    return points


def plot(
    n: int,
    points: list,
    max_radius: list,
    filename: str = "result.png",
):
    fig, ax = plt.subplots(figsize=(7, 7))

    # Draw bounding box
    box = patches.Rectangle(
        (0, 0), 1, 1,
        linewidth=2, edgecolor="black", facecolor="lightyellow", zorder=0
    )
    ax.add_patch(box)

    # Draw circles and points
    cmap = plt.cm.tab20
    for i in range(n):
        x, y = points[i]
        r = max_radius[i]
        color = cmap(i % 20)

        circle = plt.Circle(
            (x, y), r,
            facecolor=color, alpha=0.4, linewidth=1.2,
            edgecolor=color, zorder=2
        )
        ax.add_patch(circle)
        ax.plot(x, y, "o", color=color, markersize=4, zorder=3)
        ax.annotate(
            str(i), (x, y),
            textcoords="offset points", xytext=(4, 4),
            fontsize=7, color=color, zorder=4
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
    return jnp.log(jnp.expm1(jnp.clip(radii, min=1e-4)))


def raw_to_centers(raw_centers: jnp.ndarray) -> jnp.ndarray:
    """Center coordinates in (0, 1), shape (n, 2)."""
    return jax.nn.sigmoid(raw_centers)


def raw_to_radii(raw_radii: jnp.ndarray) -> jnp.ndarray:
    """Positive radii via softplus, shape (n,)."""
    return jax.nn.softplus(raw_radii)


# ---------------------------------------------------------------------------
# Overlap area helpers  (pure JAX, fully differentiable)
# ---------------------------------------------------------------------------

def _circle_circle_overlap(r1: jnp.ndarray, r2: jnp.ndarray,
                            d: jnp.ndarray) -> jnp.ndarray:
    """
    Intersection area of two circles with radii r1, r2 and center distance d.

    Formula (lens area):
        A = r1^2 * arccos((d^2 + r1^2 - r2^2) / (2*d*r1))
          + r2^2 * arccos((d^2 + r2^2 - r1^2) / (2*d*r2))
          - 0.5 * sqrt((-d+r1+r2)(d-r1+r2)(d+r1-r2)(d+r1+r2))
    """
    eps = 1e-8

    arg1 = (d * d + r1 * r1 - r2 * r2) / (2.0 * d * r1 + eps)
    arg2 = (d * d + r2 * r2 - r1 * r1) / (2.0 * d * r2 + eps)

    arg1 = jnp.clip(arg1, -1.0 + eps, 1.0 - eps)
    arg2 = jnp.clip(arg2, -1.0 + eps, 1.0 - eps)

    term1 = r1 * r1 * jnp.arccos(arg1)
    term2 = r2 * r2 * jnp.arccos(arg2)

    radicand = ((-d + r1 + r2) * (d - r1 + r2) *
                (d + r1 - r2) * (d + r1 + r2))
    radicand = jnp.clip(radicand, 0.0)
    term3 = 0.5 * jnp.sqrt(radicand)

    return term1 + term2 - term3


def _circle_wall_overlap(r: jnp.ndarray, dist: jnp.ndarray) -> jnp.ndarray:
    """
    Area of a circle of radius r whose center is at distance `dist` from a
    straight wall (dist < r for there to be overlap).

    Formula (circular segment):
        A = r^2 * arccos(dist / r) - dist * sqrt(r^2 - dist^2)
    """
    eps = 1e-8
    arg = jnp.clip(dist / r, -1.0 + eps, 1.0 - eps)
    return r * r * jnp.arccos(arg) - dist * jnp.sqrt(
        jnp.clip(r * r - dist * dist, 0.0)
    )


# ---------------------------------------------------------------------------
# Penalty functions operating on raw parameter arrays
# ---------------------------------------------------------------------------

def circle_circle_penalty(centers: jnp.ndarray, radii: jnp.ndarray) -> jnp.ndarray:
    """
    Total pairwise circle-circle overlap area.

    Vectorised over all pairs (i, j) with i < j using broadcasting.
    jnp.where masks are used to zero-out non-overlapping pairs while
    keeping gradients flowing through d and the radii.
    """
    # centers: (n, 2),  radii: (n,)
    # Pairwise distances via broadcasting
    diff = centers[:, None, :] - centers[None, :, :]  # (n, n, 2)
    d = jnp.sqrt(jnp.sum(diff ** 2, axis=-1) + 1e-16)  # (n, n)

    ri = radii[:, None]  # (n, 1)
    rj = radii[None, :]  # (1, n)

    overlap = _circle_circle_overlap(ri, rj, d)           # (n, n)
    overlapping = d < (ri + rj)                            # (n, n) bool mask

    # Upper-triangular mask (i < j) to count each pair once
    n = radii.shape[0]
    upper = jnp.triu(jnp.ones((n, n), dtype=bool), k=1)

    total = jnp.sum(
        jnp.where(overlapping & upper, overlap, 0.0)
    )
    return total


def circle_boundary_penalty(centers: jnp.ndarray, radii: jnp.ndarray) -> jnp.ndarray:
    """
    Total circle-boundary overlap area for all four walls.

    Vectorised over all circles and all four walls.
    """
    cx = centers[:, 0]  # (n,)
    cy = centers[:, 1]  # (n,)

    zero = jnp.zeros_like(radii)

    # Distances to each wall
    dist_x0 = cx                # x = 0
    dist_x1 = 1.0 - cx          # x = 1
    dist_y0 = cy                # y = 0
    dist_y1 = 1.0 - cy          # y = 1

    def wall_contrib(dist):
        return jnp.where(
            dist < radii,
            _circle_wall_overlap(radii, dist),
            zero,
        )

    return (wall_contrib(dist_x0) + wall_contrib(dist_x1) +
            wall_contrib(dist_y0) + wall_contrib(dist_y1)).sum()


# ---------------------------------------------------------------------------
# Loss function (operates on raw params dict)
# ---------------------------------------------------------------------------

def loss_fn(params: dict, penalty_weight: float = 1.0):
    """
    Loss = -sum(radii)  +  penalty_weight * (circle-circle + boundary)

    `params` is a dict with keys 'raw_centers' and 'raw_radii'.
    """
    centers = raw_to_centers(params["raw_centers"])
    radii   = raw_to_radii(params["raw_radii"])

    sum_r   = radii.sum()
    penalty = (circle_circle_penalty(centers, radii) +
               circle_boundary_penalty(centers, radii))

    loss = -sum_r + penalty_weight * penalty
    return loss, penalty


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
    n_steps: int = 2000,
    lr: float = 1e-2,
    log_every: int = 200,
) -> dict:
    """
    Run Adam optimisation and return a result dictionary.

    Parameters
    ----------
    n : int or None
        Number of circles.  Inferred from `init_centers` when not given.
    init_centers : jnp.ndarray or None, shape (n, 2)
        Initial center coordinates in [0, 1].  Random if None.
    init_radii : jnp.ndarray or None, shape (n,)
    default_init_radius : float
    penalty_weight : float
    learn_centers : bool
        Set False to keep centers fixed and only optimise radii.
    n_steps : int
    lr : float
    log_every : int
        Print progress every this many steps (0 = silent).

    Returns
    -------
    dict with keys:
        radii       - final optimised radii (jnp.ndarray, shape (n,))
        centers     - final optimised centers (jnp.ndarray, shape (n, 2))
        sum_radii   - scalar sum of radii (float)
        loss_curve  - list of loss values recorded at each step
    """
    if init_centers is not None:
        if n is None:
            n = init_centers.shape[0]
        raw_c = centers_to_raw(init_centers.astype(jnp.float32))
    else:
        if n is None:
            raise ValueError("Provide either `n` or `init_centers`.")
        key = jax.random.PRNGKey(0)
        uniform = jax.random.uniform(key, shape=(n, 2), minval=0.1, maxval=0.9)
        raw_c = centers_to_raw(uniform)

    if init_radii is not None:
        raw_r = radii_to_raw(init_radii.astype(jnp.float32))
    else:
        raw_r = radii_to_raw(
            jnp.full((n,), default_init_radius, dtype=jnp.float32)
        )

    initial_centers = raw_to_centers(raw_c)

    # Build parameter dict; centers may be frozen
    params = {
        "raw_centers": raw_c,
        "raw_radii":   raw_r,
    }

    # Optimiser – use optax Adam
    optimizer = optax.adam(lr)

    # When learn_centers=False we zero-out the gradient for raw_centers using
    # a masked optimiser that applies no update to that leaf.
    if not learn_centers:
        optimizer = optax.masked(
            optimizer,
            mask={"raw_centers": False, "raw_radii": True},
        )

    opt_state = optimizer.init(params)

    # JIT-compile the value-and-grad step for speed
    @jax.jit
    def step(params, opt_state):
        (loss, penalty), grads = jax.value_and_grad(loss_fn, has_aux=True)(
            params, penalty_weight
        )
        updates, new_opt_state = optimizer.update(grads, opt_state, params)
        new_params = optax.apply_updates(params, updates)
        return new_params, new_opt_state, loss, penalty

    loss_curve = []

    for i in range(1, n_steps + 1):
        prev_params = params
        params, opt_state, loss, penalty = step(params, opt_state)
        loss_curve.append(float(loss))

        # NaN guard
        if (jnp.isnan(loss) or
                jnp.any(jnp.isnan(raw_to_radii(params["raw_radii"]))) or
                jnp.any(jnp.isnan(raw_to_centers(params["raw_centers"])))):
            centers_np = raw_to_centers(prev_params["raw_centers"])
            radii_np   = raw_to_radii(prev_params["raw_radii"])
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
# Quick demo
# ---------------------------------------------------------------------------

def main(argv):
    """Main function."""

    # Get parameters:
    n                = int(FLAGS.n)
    seed             = int(FLAGS.seed)
    png_filename     = FLAGS.png_filename
    initial_radius   = FLAGS.initial_radius
    max_margin_param = FLAGS.max_margin_param

    # Set Python random seed
    if seed is not None:
        random.seed(seed)

    # Set JAX seed (used only if init_centers is None in optimize())
    # The center generation uses Python's random module directly via
    # generate_points(), so the Python seed above is sufficient.

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
    print("=" * 60)

    result = optimize(
        init_centers=centers,
        default_init_radius=initial_radius,
        penalty_weight=100.0,       # TO-DO: Expose these as flags
        learn_centers=True,         # Set to False to freeze the centers
        n_steps=20000,
        lr=1.0e-4,
        log_every=500,
    )

    # Display results
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
    app.run(main)