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
    soft_constraint += 0.1 * jnp.sum(d) / n**2       # Confine
    return loss + soft_constraint


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


def plot(coords_initial: npt.NDArray, coords_final: npt.NDArray, filename: str = "result.png"):
    """Plots the initial and final points.
    """
    assert coords_initial.shape == coords_final.shape

    fig, ax = plt.subplots(figsize=(12, 12))
    box = patches.Rectangle(
        (-0.5, -0.5), 1, 1, linewidth=1, linestyle="--", 
        edgecolor="green", facecolor="None", zorder=0,
    )
    ax.add_patch(box)

    # cmap = plt.cm.tab20
    # for i in range(n):
    #     x, y = points[i]
    #     r = max_radius[i]
    #     color = cmap(i % 20)
    #     circle = plt.Circle(
    #         (x, y), r, facecolor=color, alpha=0.4, linewidth=1.2, edgecolor=color, zorder=2
    #     )
    #     ax.add_patch(circle)
    #     ax.plot(x, y, "o", color=color, markersize=4, zorder=3)
    #     ax.annotate(
    #         str(i), (x, y),
    #         textcoords="offset points", xytext=(4, 4),
    #         fontsize=7, color=color, zorder=4,
    #     )

    ax.scatter(
        coords_initial[:, 0], coords_initial[:, 1], 
        s=5.0, color="darkgrey", alpha=0.5, label="Initial",
    )
    ax.scatter(
        coords_final[:, 0], coords_final[:, 1], 
        s=5.0, color="navy", alpha=0.50, label="Final",
    )
    ax.legend()
    # ax.set_xlim(-0.05, 1.05)
    # ax.set_ylim(-0.05, 1.05)
    ax.set_aspect("equal")
    ax.set_title(f"Initial and Final Vertices", fontsize=12)
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    plt.tight_layout()
    plt.axis("off")
    plt.savefig(filename, bbox_inches="tight", dpi=150)
    plt.close()
    print(f"\nPlot saved to '{filename}'")


# --------------------------------------------------------------------------
# Main
# --------------------------------------------------------------------------
if __name__ == "__main__":


    n = 1000
    # np.random.seed(42)
    # coords0 = np.random.randn(n, 2)
    rng = np.random.default_rng(seed=42)
    coords0 = rng.uniform(low=-1.5, high=1.5, size=(n, 2))

    # ---- choose optimizer: "sgd", "adam", "adamw", or "newton" ----
    method = "adamw"

    if method == "sgd":
        optimizer = optax.sgd(learning_rate=0.05, momentum=0.0)
        coords_final = run_optax(coords0, optimizer, n_steps=2000, log_every=1000)

    elif method == "adam":
        optimizer = optax.adam(learning_rate=0.05)
        coords_final = run_optax(coords0, optimizer, n_steps=10000, log_every=1000)

    elif method == "adamw":
        optimizer = optax.adamw(learning_rate=0.05, weight_decay=1e-5)
        coords_final = run_optax(coords0, optimizer, n_steps=20000, log_every=1000)

    elif method == "newton":
        coords_final = run_newton(coords0, n_steps=50, damping=1e-3)

    else:
        raise ValueError(f"Unknown method: {method}")

    # print("\nFinal coordinates:")
    # print(np.asarray(coords_final))
    plot(coords0, coords_final, f"result_III_{n}.png")
    print("Done!\n\n")