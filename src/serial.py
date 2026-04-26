from __future__ import annotations

import time

from src.rendering import Camera, ProgressCallback, RenderConfig, Scene, Vec3, trace_ray


def render_image(scene: Scene, config: RenderConfig, progress_callback: ProgressCallback | None = None) -> list[list[Vec3]]:
    camera = Camera.from_config(config.width, config.height, config.fov_degrees)
    image: list[list[Vec3]] = []
    started = time.perf_counter()

    for y in range(config.height):
        row: list[Vec3] = []
        v = 1.0 - (y + 0.5) / config.height
        for x in range(config.width):
            u = (x + 0.5) / config.width
            ray = camera.get_ray(u, v)
            row.append(trace_ray(ray, scene, depth=0, max_depth=config.max_bounces))
        image.append(row)
        if progress_callback is not None:
            progress_callback(image, y + 1, config.height, time.perf_counter() - started)
    return image
