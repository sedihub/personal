"""
Pack N disks into a unit square [0,1]x[0,1] such that:
  - No two disks overlap
  - No disk bleeds outside the square
  - The sum of radii is maximized

Strategy:
  1. Start with a known good grid/hex layout as initial guess
  2. Use scipy.optimize.minimize with SLSQP (supports constraints)
  3. Run multiple random restarts and keep the best result
  4. Visualize the final packing

Usage:
  python pack_26_disks.py
  python pack_26_disks.py --n_disks=10 --n_restarts=20 --seed=7 --output=out.png
"""

import numpy as np
from scipy.optimize import minimize
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from itertools import combinations

from absl import app, flags

FLAGS = flags.FLAGS

flags.DEFINE_integer("n_disks",     26,          "Number of disks to pack.")
flags.DEFINE_integer("n_restarts",  30,          "Number of random restarts.")
flags.DEFINE_integer("seed",        0,           "Random seed for reproducibility.")
flags.DEFINE_string( "output",      "result.png","Output filename for the plot.")


# ---------------------------------------------------------------------------
# Helpers that depend on n (passed explicitly, not a global)
# ---------------------------------------------------------------------------

def unpack(x, n):
    """Split flat vector into (cx, cy, r) arrays."""
    return x[0:n], x[n:2*n], x[2*n:3*n]

# ---------------------------------------------------------------------------
# Objective and constraints
# ---------------------------------------------------------------------------

def neg_sum_radii(x, n):
    """Objective: negative sum of radii (we minimise)."""
    return -np.sum(x[2*n:3*n])


def neg_sum_radii_grad(x, n):
    g = np.zeros_like(x)
    g[2*n:3*n] = -1.0
    return g


def build_constraints(n):
    """
    Return a list of scipy constraint dicts.
    - Each disk stays inside the unit square: r_i <= cx_i, cx_i <= 1-r_i, etc.
    - Each pair of disks does not overlap: dist(ci,cj) >= r_i + r_j
    """
    cons = []

    # ---- boundary constraints (4 per disk) ----
    for i in range(n):
        # cx_i - r_i >= 0
        def left(x, i=i):
            return x[i] - x[2*n+i]
        # 1 - cx_i - r_i >= 0
        def right(x, i=i):
            return 1.0 - x[i] - x[2*n+i]
        # cy_i - r_i >= 0
        def bottom(x, i=i):
            return x[n+i] - x[2*n+i]
        # 1 - cy_i - r_i >= 0
        def top(x, i=i):
            return 1.0 - x[n+i] - x[2*n+i]

        cons += [
            {'type': 'ineq', 'fun': left},
            {'type': 'ineq', 'fun': right},
            {'type': 'ineq', 'fun': bottom},
            {'type': 'ineq', 'fun': top},
        ]

    # ---- non-overlap constraints (one per pair) ----
    for i, j in combinations(range(n), 2):
        def no_overlap(x, i=i, j=j):
            dx = x[i] - x[j]
            dy = x[n+i] - x[n+j]
            dist = np.sqrt(dx*dx + dy*dy)
            return dist - (x[2*n+i] + x[2*n+j])
        cons.append({'type': 'ineq', 'fun': no_overlap})

    return cons


# ---------------------------------------------------------------------------
# Initialisation helpers
# ---------------------------------------------------------------------------

def hex_grid_init(n, r0=0.07):
    """
    Place disks on a hexagonal grid inside [0,1]x[0,1] with equal radii r0.
    Returns flat x vector.
    """
    positions = []
    row = 0
    while len(positions) < n:
        cols = int(1.0 / (2*r0))
        offset = r0 if row % 2 == 1 else 0.0
        x_start = r0
        for col in range(cols):
            cx = x_start + col * 2*r0 + offset
            cy = r0 + row * r0 * np.sqrt(3)
            if cx + r0 <= 1.0 and cy + r0 <= 1.0:
                positions.append((cx, cy))
            if len(positions) >= n:
                break
        row += 1
        if row > 50:
            break

    # fall back: place remaining on a simple grid
    while len(positions) < n:
        positions.append((0.5, 0.5))

    positions = positions[:n]
    cx = np.array([p[0] for p in positions])
    cy = np.array([p[1] for p in positions])
    r  = np.full(n, r0)
    return np.concatenate([cx, cy, r])


def random_init(n, rng=None):
    """Random feasible initialisation with small equal radii."""
    if rng is None:
        rng = np.random.default_rng()
    r0 = 0.04
    cx = rng.uniform(r0, 1-r0, n)
    cy = rng.uniform(r0, 1-r0, n)
    r  = np.full(n, r0)
    return np.concatenate([cx, cy, r])


# ---------------------------------------------------------------------------
# Penalty-based optimisation (faster, used for warm starts)
# ---------------------------------------------------------------------------

def penalty_objective(x, n, penalty=1e4):
    """Soft-penalty version: maximise sum(r) - penalty * violations."""
    cx, cy, r = unpack(x, n)
    obj = -np.sum(r)

    # boundary violations
    viol = np.maximum(0, r - cx)
    viol += np.maximum(0, r - (1 - cx))
    viol += np.maximum(0, r - cy)
    viol += np.maximum(0, r - (1 - cy))

    # overlap violations
    for i, j in combinations(range(n), 2):
        dist = np.sqrt((cx[i]-cx[j])**2 + (cy[i]-cy[j])**2)
        viol_ij = max(0.0, r[i]+r[j] - dist)
        viol += viol_ij

    return obj + penalty * np.sum(viol**2)


def run_penalty_phase(x0, n):
    """Increasing-penalty phase to drive x0 toward feasibility."""
    x = x0.copy()
    for p in [1e2, 1e3, 1e4]:
        res = minimize(penalty_objective, x, args=(n, p),
                       method='L-BFGS-B',
                       bounds=[(1e-6, 1-1e-6)]*n +
                               [(1e-6, 1-1e-6)]*n +
                               [(1e-6, 0.5)]*n,
                       options={'maxiter': 2000, 'ftol': 1e-12})
        x = res.x
    return x


def run_slsqp(x0, n, cons):
    """Final constrained polish with SLSQP."""
    bounds = [(1e-8, 1-1e-8)]*n + [(1e-8, 1-1e-8)]*n + [(1e-8, 0.5)]*n
    res = minimize(neg_sum_radii, x0,
                   args=(n,),
                   method='SLSQP',
                   jac=neg_sum_radii_grad,
                   bounds=bounds,
                   constraints=cons,
                   options={'maxiter': 5000, 'ftol': 1e-12, 'disp': False})
    return res


# ---------------------------------------------------------------------------
# Feasibility check
# ---------------------------------------------------------------------------

def is_feasible(x, n, tol=1e-5):
    cx, cy, r = unpack(x, n)
    if np.any(r < tol): return False
    if np.any(cx - r < -tol): return False
    if np.any(1 - cx - r < -tol): return False
    if np.any(cy - r < -tol): return False
    if np.any(1 - cy - r < -tol): return False
    for i, j in combinations(range(n), 2):
        dist = np.sqrt((cx[i]-cx[j])**2 + (cy[i]-cy[j])**2)
        if dist < r[i]+r[j] - tol:
            return False
    return True


# ---------------------------------------------------------------------------
# Main optimisation loop
# ---------------------------------------------------------------------------

def optimise(n, n_restarts=25, seed=42):
    rng = np.random.default_rng(seed)
    cons = build_constraints(n)
    best_val = -np.inf
    best_x   = None
    x_cand   = None

    print(f"Packing {n} disks — running {n_restarts} restarts …")

    for trial in range(n_restarts):
        x0 = hex_grid_init(n) if trial == 0 else random_init(n, rng=rng)

        x1     = run_penalty_phase(x0, n)
        res    = run_slsqp(x1, n, cons)
        x_cand = res.x
        val    = np.sum(x_cand[2*n:3*n])

        feasible = is_feasible(x_cand, n)
        status   = "✓" if feasible else "✗"
        print(f"  trial {trial+1:2d}: sum_r = {val:.6f}  feasible={status}")

        if feasible and val > best_val:
            best_val = val
            best_x   = x_cand.copy()

    if best_x is None:
        print("No feasible solution found; returning best infeasible result.")
        best_x = x_cand

    return best_x, best_val


# ---------------------------------------------------------------------------
# Visualisation
# ---------------------------------------------------------------------------

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
# Entry point
# ---------------------------------------------------------------------------

def main(_argv):
    n           = FLAGS.n_disks
    n_restarts  = FLAGS.n_restarts
    seed        = FLAGS.seed
    output      = FLAGS.output

    best_x, best_val = optimise(n, n_restarts=n_restarts, seed=seed)

    cx, cy, r = unpack(best_x, n)
    print("\n=== Best solution ===")
    print(f"Sum of radii : {best_val:.8f}")
    print(f"Min radius   : {r.min():.8f}")
    print(f"Max radius   : {r.max():.8f}")
    print(f"Feasible     : {is_feasible(best_x, n)}")
    print("\nDisk positions and radii:")
    print(f"{'#':>3}  {'cx':>10}  {'cy':>10}  {'r':>10}")
    for i in range(n):
        print(f"{i+1:3d}  {cx[i]:10.6f}  {cy[i]:10.6f}  {r[i]:10.6f}")

    points     = [(cx[i], cy[i]) for i in range(n)]
    max_radius = list(r)
    plot(n, points, max_radius, filename=output)


if __name__ == "__main__":
    app.run(main)