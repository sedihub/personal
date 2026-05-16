"""
Learn radii
"""

"""
circle_packing.py
-----------------
Maximize the sum of radii of n circles with fixed centers inside a unit square,
penalizing circle–circle overlap and circle–boundary overlap.

Usage example
-------------
    import torch
    from circle_packing import CirclePacker, optimize

    centers = torch.tensor([[0.25, 0.25],
                             [0.75, 0.25],
                             [0.50, 0.75]])

    result = optimize(centers, n_steps=2000, lr=1e-2)
    print("Optimal radii:", result["radii"])
    print("Sum of radii: ", result["sum_radii"])


CLI
------------
python3 ./circle_packing_pytorch.py \
  --n=26 \
  --seed=42 \
  --png_filename="__DELETE_ME__.png" \
  --initial_radius=0.01 \
  --max_margin_param=2.0
"""

import math
import torch
import torch.nn as nn
from typing import Optional
import random
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from absl import flags
from absl import app

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
# Overlap area helpers
# ---------------------------------------------------------------------------

def _circle_circle_overlap(r1: torch.Tensor, r2: torch.Tensor,
                            d: torch.Tensor) -> torch.Tensor:
    """
    Intersection area of two circles with radii r1, r2 and center distance d.

    Formula (lens area):
        A = r1² · arccos((d² + r1² - r2²) / (2·d·r1))
          + r2² · arccos((d² + r2² - r1²) / (2·d·r2))
          - ½ · sqrt((−d+r1+r2)(d−r1+r2)(d+r1−r2)(d+r1+r2))

    Only called when d < r1 + r2  (circles actually overlap).
    """
    eps = 1e-8

    arg1 = (d * d + r1 * r1 - r2 * r2) / (2.0 * d * r1 + eps)
    arg2 = (d * d + r2 * r2 - r1 * r1) / (2.0 * d * r2 + eps)

    # Clamp to [-1, 1] to keep arccos numerically safe
    arg1 = arg1.clamp(-1.0 + eps, 1.0 - eps)
    arg2 = arg2.clamp(-1.0 + eps, 1.0 - eps)

    term1 = r1 * r1 * torch.acos(arg1)
    term2 = r2 * r2 * torch.acos(arg2)

    # Radicand: (−d+r1+r2)(d−r1+r2)(d+r1−r2)(d+r1+r2)
    radicand = ((-d + r1 + r2) * (d - r1 + r2) *
                (d + r1 - r2) * (d + r1 + r2))
    radicand = radicand.clamp(min=0.0)          # numerical guard
    term3 = 0.5 * torch.sqrt(radicand)

    return term1 + term2 - term3


def _circle_wall_overlap(r: torch.Tensor, dist: torch.Tensor) -> torch.Tensor:
    """
    Area of a circle of radius r that crosses a straight wall at distance
    `dist` from the center (dist < r required for there to be overlap).

    Formula (circular segment):
        A = r² · arccos(dist / r) − dist · sqrt(r² − dist²)
    """
    eps = 1e-8
    arg = (dist / r).clamp(-1.0 + eps, 1.0 - eps)
    return r * r * torch.acos(arg) - dist * torch.sqrt((r * r - dist * dist).clamp(min=0.0))


# ---------------------------------------------------------------------------
# nn.Module
# ---------------------------------------------------------------------------

class CirclePacker(nn.Module):
    """
    Learnable radii for n circles with fixed centers inside a unit square.

    Parameters
    ----------
    centers : Tensor, shape (n, 2)
        Fixed (x, y) coordinates of circle centers.  Values must lie in [0, 1].
    init_radii : Tensor or None, shape (n,)
        Initial radii.  If None, all radii start at `default_init`.
    default_init : float
        Fallback initial radius when `init_radii` is None.
    penalty_weight : float
        Scalar multiplier applied to all penalty terms.
    """

    def __init__(self,
                 centers: torch.Tensor,
                 init_radii: Optional[torch.Tensor] = None,
                 default_init: float = 0.01,
                 penalty_weight: float = 1.0):
        super().__init__()

        n = centers.shape[0]
        self.register_buffer("centers", centers.float())
        self.penalty_weight = penalty_weight

        if init_radii is not None:
            raw_init = self._to_raw(init_radii.float())
        else:
            raw_init = self._to_raw(
                torch.full((n,), default_init, dtype=torch.float32))

        # Radii are parameterised as softplus(raw) so they stay positive.
        self.raw_radii = nn.Parameter(raw_init)

    # ------------------------------------------------------------------
    # Parameterisation helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _to_raw(radii: torch.Tensor) -> torch.Tensor:
        """Inverse of softplus: raw = log(exp(r) - 1)."""
        return torch.log(torch.expm1(radii.clamp(min=1e-6)))

    @property
    def radii(self) -> torch.Tensor:
        """Positive radii obtained via softplus."""
        return nn.functional.softplus(self.raw_radii)

    # ------------------------------------------------------------------
    # Penalty terms
    # ------------------------------------------------------------------

    def circle_circle_penalty(self) -> torch.Tensor:
        """Total pairwise circle–circle overlap area."""
        r = self.radii                          # (n,)
        cx = self.centers                       # (n, 2)
        n = r.shape[0]

        total = torch.zeros(1, dtype=r.dtype, device=r.device)

        for i in range(n):
            for j in range(i + 1, n):
                d = torch.norm(cx[i] - cx[j])
                sum_r = r[i] + r[j]
                if d < sum_r:                   # circles overlap
                    total = total + _circle_circle_overlap(r[i], r[j], d)

        return total

    def circle_boundary_penalty(self) -> torch.Tensor:
        """Total circle–boundary overlap area for all four walls."""
        r = self.radii
        cx = self.centers[:, 0]                 # x-coordinates
        cy = self.centers[:, 1]                 # y-coordinates
        n = r.shape[0]

        total = torch.zeros(1, dtype=r.dtype, device=r.device)

        for i in range(n):
            ri = r[i]

            # x = 0  wall: overlap when cx[i] < ri
            dist_x0 = cx[i]
            if dist_x0 < ri:
                total = total + _circle_wall_overlap(ri, dist_x0)

            # x = 1  wall: overlap when (1 - cx[i]) < ri
            dist_x1 = 1.0 - cx[i]
            if dist_x1 < ri:
                total = total + _circle_wall_overlap(ri, dist_x1)

            # y = 0  wall: overlap when cy[i] < ri
            dist_y0 = cy[i]
            if dist_y0 < ri:
                total = total + _circle_wall_overlap(ri, dist_y0)

            # y = 1  wall: overlap when (1 - cy[i]) < ri
            dist_y1 = 1.0 - cy[i]
            if dist_y1 < ri:
                total = total + _circle_wall_overlap(ri, dist_y1)

        return total

    # ------------------------------------------------------------------
    # Forward: returns the scalar loss to be *minimised*
    # ------------------------------------------------------------------

    def forward(self) -> torch.Tensor:
        """
        Loss = −sum(radii)  +  penalty_weight · (circle–circle + boundary penalties)

        Minimising this loss maximises the sum of radii while suppressing overlaps.
        """
        sum_r = self.radii.sum()
        penalty = self.circle_circle_penalty() + self.circle_boundary_penalty()
        return -sum_r + self.penalty_weight * penalty


# ---------------------------------------------------------------------------
# Convenience optimisation loop
# ---------------------------------------------------------------------------

def optimize(centers: torch.Tensor,
             init_radii: Optional[torch.Tensor] = None,
             default_init: float = 0.01,
             penalty_weight: float = 1.0,
             n_steps: int = 2000,
             lr: float = 1e-2,
             log_every: int = 200,
             device: Optional[torch.device] = None) -> dict:
    """
    Run Adam optimisation and return a result dictionary.

    Parameters
    ----------
    centers : Tensor, shape (n, 2)
    init_radii : Tensor or None, shape (n,)
    default_init : float
    penalty_weight : float
    n_steps : int
    lr : float
    log_every : int
        Print progress every this many steps (0 = silent).
    device : torch.device or None

    Returns
    -------
    dict with keys:
        radii       – final optimised radii (Tensor, shape (n,))
        sum_radii   – scalar sum of radii (float)
        loss_curve  – list of loss values recorded at each step
        model       – the CirclePacker instance
    """
    if device is None:
        device = torch.device("cpu")

    centers = centers.to(device)
    if init_radii is not None:
        init_radii = init_radii.to(device)

    model = CirclePacker(centers,
                         init_radii=init_radii,
                         default_init=default_init,
                         penalty_weight=penalty_weight).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    loss_curve = []

    for step in range(1, n_steps + 1):
        optimizer.zero_grad()
        loss = model()
        loss.backward()
        optimizer.step()
        loss_curve.append(loss.item())

        if log_every and step % log_every == 0:
            radii = model.radii.detach()
            print(f"Step {step:5d} | loss={loss.item():+.6f} | "
                  f"sum_r={radii.sum().item():.6f} | "
                  f"radii={radii.cpu().numpy().round(4)}")

    final_radii = model.radii.detach()
    return {
        "radii": final_radii,
        "sum_radii": final_radii.sum().item(),
        "loss_curve": loss_curve,
        "model": model,
    }


# ---------------------------------------------------------------------------
# Quick demo
# ---------------------------------------------------------------------------
def main(argv):
    """Main function.
    """

    # Get parameters:
    n = int(FLAGS.n)
    seed = int(FLAGS.seed)
    png_filename = FLAGS.png_filename
    initial_radius = FLAGS.initial_radius
    max_margin_param = FLAGS.max_margin_param

	# Set Python random seed:    
    if seed is not None:
        random.seed(seed)

    # Set pytorch seed
    torch.manual_seed(seed)

    # Three circles arranged in a triangle
    centers = torch.tensor(
        list(generate_points(n, max_margin_param).values())
    )
    print(centers)

    plot(
        n, 
        centers.tolist(),
        max_radius=[initial_radius] * n,
        filename=png_filename.replace(".png", "_initial.png"),
    )

    print("=" * 60)
    print("Circle packing optimisation – unit square, n=3")
    print("=" * 60)

    result = optimize(
        centers,
        default_init=initial_radius,
        penalty_weight=100.0,  # TO-DO: Expose these as flags
        n_steps=10000,
        lr=1.0e-3,
        log_every=500,
    )

    # Display results:
    print("\nFinal radii :", result["radii"].numpy().round(6))
    print("Sum of radii:", round(result["sum_radii"], 6))
    plot(
        n, 
        centers.tolist(),
        max_radius=result["radii"].tolist(),
        filename=png_filename.replace(".png", "_final.png"),
    )


if __name__ == "__main__":
    app.run(main)