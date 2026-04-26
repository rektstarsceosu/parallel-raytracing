from __future__ import annotations

import argparse
import time

from src.parallel import render_image_parallel
from src.rendering import LivePreview, RenderConfig, SCENE_PRESETS
from src.serial import render_image

FIXED_WIDTH = 640
FIXED_HEIGHT = 360
FIXED_FOV = 60.0
FIXED_PROCESSES = 4
FIXED_ROWS_PER_CHUNK = 8


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Ray tracing runner with fixed project settings.")
    parser.add_argument("--mode", choices=["sequential", "parallel"], required=True)
    parser.add_argument("--scene", choices=sorted(SCENE_PRESETS.keys()), default="simple")
    parser.add_argument("--preview", action="store_true", help="Show live render progress.")
    return parser


def main() -> None:
    parser = _build_parser()
    args = parser.parse_args()

    scene = SCENE_PRESETS[args.scene]()
    config = RenderConfig(width=FIXED_WIDTH, height=FIXED_HEIGHT, fov_degrees=FIXED_FOV)
    preview = LivePreview(FIXED_WIDTH, FIXED_HEIGHT) if args.preview else None

    start = time.perf_counter()
    if args.mode == "sequential":
        render_image(
            scene,
            config,
            progress_callback=preview.update if preview is not None else None,
        )
    else:
        render_image_parallel(
            scene,
            config,
            processes=FIXED_PROCESSES,
            rows_per_chunk=FIXED_ROWS_PER_CHUNK,
            progress_callback=preview.update if preview is not None else None,
        )
    elapsed = time.perf_counter() - start
    if preview is not None:
        preview.close_after_delay(1.0)
    print(f"Rendered {args.scene} in {elapsed:.3f}s")


if __name__ == "__main__":
    main()
