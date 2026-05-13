import random
import math
import matplotlib.pyplot as plt
import matplotlib.patches as patches


class CirclePacking:
    def __init__(self, n: int, seed: int = None):
        self.n = n
        if seed is not None:
            random.seed(seed)
        self.points = self._generate_points()
        self.max_radius = self._init_max_radius()
        self.distances = self._compute_distances()
        self._run_triangle_updates()

    def _generate_points(self) -> dict:
        """Generate n random points in the unit square."""
        return {i: (random.random(), random.random()) for i in range(self.n)}

    def _min_dist_to_perimeter(self, x: float, y: float) -> float:
        """Minimum distance from point (x, y) to the perimeter of the unit square."""
        return min(x, 1 - x, y, 1 - y)

    def _init_max_radius(self) -> dict:
        """Initialize max_radius with each point's minimum distance to the perimeter."""
        return {
            i: self._min_dist_to_perimeter(x, y)
            for i, (x, y) in self.points.items()
        }

    def _compute_distances(self) -> dict:
        """Compute pairwise distances for all pairs (i, j) where i < j."""
        distances = {}
        for i in range(self.n):
            for j in range(i + 1, self.n):
                xi, yi = self.points[i]
                xj, yj = self.points[j]
                distances[(i, j)] = math.sqrt((xi - xj) ** 2 + (yi - yj) ** 2)
        return distances

    def _get_distance(self, i: int, j: int) -> float:
        """Retrieve distance between points i and j (handles both orderings)."""
        if i < j:
            return self.distances[(i, j)]
        return self.distances[(j, i)]

    def _run_triangle_updates(self):
        """
        Iterate over all pairs (i, j) with i < j, sorted by distance ascending.
        For each pair, update r[i] = min(r[i], max(d/2, d - r[j])) and
        symmetrically r[j] = min(r[j], max(d/2, d - r[i])).
        """
        n = self.n
        r = self.max_radius  # shorthand reference

        # Sort pairs by distance so closest neighbours are processed first
        pairs = sorted(
            ((i, j) for i in range(n) for j in range(i + 1, n)),
            key=lambda p: self._get_distance(p[0], p[1])
        )

        for i, j in pairs:
            d_ij = self._get_distance(i, j)
            r[i] = min(r[i], max(d_ij / 2, d_ij - r[j]))
            r[j] = min(r[j], max(d_ij / 2, d_ij - r[i]))

    def sum_of_radii(self) -> float:
        return sum(self.max_radius.values())

    def print_results(self):
        print(f"Sum of radii: {self.sum_of_radii():.6f}")
        print("\nPoint coordinates and radii:")
        print(f"{'Index':>6}  {'x':>10}  {'y':>10}  {'radius':>12}")
        print("-" * 44)
        for i in range(self.n):
            x, y = self.points[i]
            r = self.max_radius[i]
            print(f"{i:>6}  {x:>10.6f}  {y:>10.6f}  {r:>12.6f}")

    def plot(self, filename: str = "result.png"):
        fig, ax = plt.subplots(figsize=(7, 7))

        # Draw bounding box
        box = patches.Rectangle(
            (0, 0), 1, 1,
            linewidth=2, edgecolor="black", facecolor="lightyellow", zorder=0
        )
        ax.add_patch(box)

        # Draw circles and points
        cmap = plt.cm.tab20
        for i in range(self.n):
            x, y = self.points[i]
            r = self.max_radius[i]
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

        total = round(self.sum_of_radii(), 3)
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


# ── Entry point ──────────────────────────────────────────────────────────────
if __name__ == "__main__":
    import sys

    n = int(sys.argv[1]) if len(sys.argv) > 1 else 10
    seed = int(sys.argv[2]) if len(sys.argv) > 2 else 42

    packing = CirclePacking(n=n, seed=seed)
    packing.print_results()
    packing.plot("result.png")
