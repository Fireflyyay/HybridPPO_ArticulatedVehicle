import argparse
import os
import sys
from typing import Iterable, List, Sequence, Tuple

import numpy as np

THIS_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.dirname(THIS_DIR)
SRC_DIR = os.path.join(REPO_ROOT, "src")
if SRC_DIR not in sys.path:
    sys.path.append(SRC_DIR)

import matplotlib.pyplot as plt
from matplotlib.axes import Axes
from matplotlib.figure import Figure
from matplotlib.patches import Patch, Polygon as PolygonPatch

from common.runtime_config import ExperimentConfig
from env.scenes import BaselineInspiredSceneFactory, SceneSpec
from env.success import articulated_body_polygons


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Render the current HybridPPO scenes with start and goal vehicles.")
    parser.add_argument(
        "--level",
        choices=("Debug", "Warmup", "Normal", "All"),
        default="All",
        help="Which level to render. Use All to render all implemented levels.",
    )
    parser.add_argument("--seed", type=int, default=0, help="Random seed passed to the scene generator.")
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Optional output image path. If omitted, the figure is shown interactively.",
    )
    return parser.parse_args()


def _scene_levels(level: str) -> List[str]:
    return [str(level)] if str(level) != "All" else ["Debug", "Warmup", "Normal"]


def _build_scene(level: str, seed: int) -> Tuple[SceneSpec, ExperimentConfig]:
    config = ExperimentConfig()
    scene_factory = BaselineInspiredSceneFactory(config.env.scene_presets, vehicle_config=config.vehicle)
    scene = scene_factory.generate(str(level), np.random.default_rng(int(seed)))
    return scene, config


def _polygon_patch(points: Sequence[Tuple[float, float]], facecolor: str, edgecolor: str, alpha: float, label: str = "") -> PolygonPatch:
    return PolygonPatch(list(points), closed=True, facecolor=facecolor, edgecolor=edgecolor, linewidth=1.2, alpha=alpha, label=label)


def _add_shapely_polygon(ax: Axes, polygon, facecolor: str, edgecolor: str, alpha: float) -> None:
    if polygon.is_empty:
        return
    ax.add_patch(
        _polygon_patch(
            [(float(x), float(y)) for x, y in polygon.exterior.coords],
            facecolor=facecolor,
            edgecolor=edgecolor,
            alpha=alpha,
        )
    )
    for interior in polygon.interiors:
        ax.add_patch(
            _polygon_patch(
                [(float(x), float(y)) for x, y in interior.coords],
                facecolor=ax.get_facecolor(),
                edgecolor="none",
                alpha=1.0,
            )
        )


def _add_vehicle(ax: Axes, state, vehicle_config, front_colors: Tuple[str, str], rear_colors: Tuple[str, str], label: str) -> None:
    front_poly, rear_poly = articulated_body_polygons(state, vehicle_config)
    ax.add_patch(
        _polygon_patch(
            [(float(x), float(y)) for x, y in front_poly.exterior.coords],
            facecolor=front_colors[0],
            edgecolor=front_colors[1],
            alpha=0.85,
            label=label,
        )
    )
    ax.add_patch(
        _polygon_patch(
            [(float(x), float(y)) for x, y in rear_poly.exterior.coords],
            facecolor=rear_colors[0],
            edgecolor=rear_colors[1],
            alpha=0.75,
        )
    )


def _render_scene(ax: Axes, scene: SceneSpec, config: ExperimentConfig) -> None:
    xmin, xmax, ymin, ymax = scene.world_bounds
    ax.set_facecolor("#f5f3ef")
    for obstacle in scene.obstacles:
        _add_shapely_polygon(ax, obstacle, facecolor="#4a4a4a", edgecolor="#2b2b2b", alpha=0.95)

    _add_vehicle(
        ax,
        scene.start_state,
        config.vehicle,
        front_colors=("#4db6ac", "#1f6f69"),
        rear_colors=("#7fd4cb", "#1f6f69"),
        label="Start vehicle",
    )
    _add_vehicle(
        ax,
        scene.goal_state,
        config.vehicle,
        front_colors=("#f4a261", "#9c4f16"),
        rear_colors=("#f7bf8a", "#9c4f16"),
        label="Goal vehicle",
    )

    ax.set_xlim(float(xmin), float(xmax))
    ax.set_ylim(float(ymin), float(ymax))
    ax.set_aspect("equal", adjustable="box")
    ax.set_title(f"{scene.level} | seed={scene.metadata.get('seed', 'n/a')}")
    ax.set_xlabel("x [m]")
    ax.set_ylabel("y [m]")
    ax.grid(True, linestyle=":", linewidth=0.5, alpha=0.4)


def _legend_handles() -> List[Patch]:
    return [
        Patch(facecolor="#4a4a4a", edgecolor="#2b2b2b", label="Obstacle"),
        Patch(facecolor="#4db6ac", edgecolor="#1f6f69", label="Start vehicle"),
        Patch(facecolor="#f4a261", edgecolor="#9c4f16", label="Goal vehicle"),
    ]


def _create_figure(levels: Iterable[str], seed: int) -> Figure:
    level_list = list(levels)
    figure, axes = plt.subplots(1, len(level_list), figsize=(7.0 * len(level_list), 7.0), squeeze=False)
    for axis, level in zip(axes[0], level_list):
        scene, config = _build_scene(level, seed)
        scene.metadata.setdefault("seed", int(seed))
        _render_scene(axis, scene, config)
    figure.legend(handles=_legend_handles(), loc="upper right")
    figure.suptitle("HybridPPO Current Scene Examples", fontsize=14)
    figure.tight_layout()
    return figure


def main() -> None:
    args = _parse_args()
    figure = _create_figure(_scene_levels(str(args.level)), int(args.seed))
    if args.output:
        output_path = os.path.abspath(str(args.output))
        output_dir = os.path.dirname(output_path)
        if output_dir:
            os.makedirs(output_dir, exist_ok=True)
        figure.savefig(output_path, dpi=200, bbox_inches="tight")
        print(output_path)
        plt.close(figure)
        return
    plt.show()


if __name__ == "__main__":
    main()