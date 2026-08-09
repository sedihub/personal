"""
Optimize a set of 2D points so that pairwise distances cluster around 1.0.

loss = -(1/n^2) * sum_{i} sum_{j>i} exp(-(d_ij - 1.0)^2)

Supports SGD, Adam, AdamW (via optax) and a Newton's-method optimizer
(via jax.hessian + linear solve).
"""

import numpy as np
import numpy.typing as npt
import jax
import jax.numpy as jnp
import optax
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from scipy.spatial.distance import pdist

jax.config.update("jax_enable_x64", True)  # helps Newton's method numerically


# --------------------------------------------------------------------------
# Loss function
# --------------------------------------------------------------------------
def pairwise_loss(coords):
    """coords: (n, 2) array. Returns scalar loss."""
    n = coords.shape[0]
    diff = coords[:, None, :] - coords[None, :, :]           # (n, n, 2)
    d2 = jnp.sum(diff ** 2, axis=-1)                         # (n, n)
    d = jnp.sqrt(d2 + 1e-12)                                 # avoid nan grad at d=0

    mask = jnp.triu(jnp.ones((n, n), dtype=coords.dtype), k=1)  # i < j only
    energy = jnp.exp(-(d - 1.0) ** 2)

    coef = -1.0 / (n ** 2 * jnp.exp(-1.0))
    loss = 0.5 * coef * jnp.sum(energy * mask)       # Extra 0.5 because of the mask 
    soft_constraint = -coef * jnp.sum(jnp.exp(-d2))  # Prevent overlap
    soft_constraint += 0.6 * jnp.sum(d) / n**2       # Confine
    return loss + 0.2 * soft_constraint


# --------------------------------------------------------------------------
# Gradient-based optimizers (SGD / Adam / AdamW) via optax
# --------------------------------------------------------------------------
def run_optax(coords0, optimizer, n_steps=1000, log_every=100):
    coords = jnp.array(coords0)
    opt_state = optimizer.init(coords)

    @jax.jit
    def step(coords, opt_state):
        loss, grads = jax.value_and_grad(pairwise_loss)(coords)
        updates, opt_state = optimizer.update(grads, opt_state, coords)
        coords = optax.apply_updates(coords, updates)
        return coords, opt_state, loss

    for i in range(n_steps):
        coords, opt_state, loss = step(coords, opt_state)
        if i % log_every == 0:
            print(f"[optax] step {i:5d}  loss = {loss: .6f}")

    print(f"[optax] final   loss = {pairwise_loss(coords): .6f}")
    return coords


# --------------------------------------------------------------------------
# Newton's method (full Hessian, dense linear solve)
# --------------------------------------------------------------------------
def run_newton(coords0, n_steps=50, damping=1e-4, log_every=10):
    coords0 = jnp.array(coords0)
    n = coords0.shape[0]

    # Work with a flattened (2n,) vector since jax.hessian needs a flat input
    # to produce a clean (2n, 2n) matrix.
    def flat_loss(flat):
        return pairwise_loss(flat.reshape(n, 2))

    grad_fn = jax.jit(jax.grad(flat_loss))
    hess_fn = jax.jit(jax.hessian(flat_loss))

    flat = coords0.reshape(-1)
    eye = jnp.eye(flat.shape[0], dtype=flat.dtype)

    for i in range(n_steps):
        g = grad_fn(flat)
        H = hess_fn(flat)
        # Damping (Levenberg-Marquardt style) keeps the solve well-posed,
        # since the loss is non-convex and H need not be positive definite.
        H_damped = H + damping * eye
        step_dir = jnp.linalg.solve(H_damped, g)
        flat = flat - step_dir

        if i % log_every == 0:
            print(f"[newton] step {i:5d}  loss = {flat_loss(flat): .6f}")

    print(f"[newton] final   loss = {flat_loss(flat): .6f}")
    return flat.reshape(n, 2)


def plot(
    coords_initial: npt.NDArray, 
    coords_final: npt.NDArray, 
    filename: str = "result.png",
    num_bins: int = 200
):
    """Plots the initial and final points (top) and their distance distributions (bottom).
    """
    assert coords_initial.shape == coords_final.shape

    # Create a figure with 2 subplots (stacked vertically)
    fig, (ax1, ax2) = plt.subplots(
        nrows=2, ncols=1, figsize=(12, 18),
        gridspec_kw={'height_ratios': [3, 1]}
    )
    
    # ==========================================
    # Top Subplot: Scatter Plot (Original logic)
    # ==========================================
    box = patches.Rectangle(
        (-0.5, -0.5), 1, 1, linewidth=1, linestyle="--", 
        edgecolor="green", facecolor="None", zorder=0,
    )
    ax1.add_patch(box)

    ax1.scatter(
        coords_initial[:, 0], coords_initial[:, 1], 
        s=5.0, color="darkgrey", alpha=0.5, label="Initial",
    )
    ax1.scatter(
        coords_final[:, 0], coords_final[:, 1], 
        s=5.0, color="navy", alpha=0.50, label="Final",
    )
    ax1.legend()
    # ax1.set_xlim(-0.05, 1.05)
    # ax1.set_ylim(-0.05, 1.05)
    ax1.set_aspect("equal")
    ax1.set_title("Initial and Final Vertices", fontsize=12)
    ax1.set_xlabel("x")
    ax1.set_ylabel("y")
    ax1.axis("off")

    # ==========================================
    # Bottom Subplot: Distance Histograms
    # ==========================================
    # Calculate Euclidean distances from the origin
    dist_initial = pdist(coords_initial, metric="euclidean")
    dist_final = pdist(coords_final, metric="euclidean")
    print(f"\n\t{dist_initial.shape=}")
    print(f"\t{dist_final.shape=}")

    # Plot both histograms matching the scatter plot colors
    dist_initial, bins, _ = ax2.hist(dist_initial, bins=num_bins, align="mid", color="darkgrey", alpha=0.5, label="Initial")
    dist_final, _, _ = ax2.hist(dist_final, bins=bins, align="mid", color="navy", alpha=0.5, label="Final")
    print(f"\n\t{dist_initial.shape=}, {np.sum(dist_initial)=}")
    print(f"\t{dist_final.shape=}, {np.sum(dist_final)=}")
    
    ax2.legend()
    ax2.set_title("Distribution of Distances from Origin", fontsize=12)
    ax2.set_xlabel("Distance")
    ax2.set_ylabel("Count")

    # Save and cleanup
    plt.tight_layout()
    plt.savefig(filename, bbox_inches="tight", dpi=300)
    plt.close()
    print(f"\nPlot saved to '{filename}'")


# --------------------------------------------------------------------------
# Main
# --------------------------------------------------------------------------
if __name__ == "__main__":


    n = 4000
    n_steps = 10000
    lr = 0.001
    # np.random.seed(42)
    # coords0 = np.random.randn(n, 2)
    rng = np.random.default_rng(seed=42)
    coords0 = rng.uniform(low=-1.5, high=1.5, size=(n, 2))

    # ---- choose optimizer: "sgd", "adam", "adamw", or "newton" ----
    method = "adamw"

    if method == "sgd":
        optimizer = optax.sgd(learning_rate=lr, momentum=0.0)
        coords_final = run_optax(coords0, optimizer, n_steps=n_steps, log_every=1000)

    elif method == "adam":
        optimizer = optax.adam(learning_rate=lr)
        coords_final = run_optax(coords0, optimizer, n_steps=n_steps, log_every=1000)

    elif method == "adamw":
        optimizer = optax.adamw(learning_rate=lr, weight_decay=1e-5)
        coords_final = run_optax(coords0, optimizer, n_steps=n_steps, log_every=1000)

    elif method == "newton":
        coords_final = run_newton(coords0, n_steps=50, damping=1e-3)

    else:
        raise ValueError(f"Unknown method: {method}")

    # print("\nFinal coordinates:")
    # print(np.asarray(coords_final))
    plot(coords0, coords_final, f"result_III_{n}.png")
    print("Done!\n\n")