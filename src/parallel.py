from __future__ import annotations

from dataclasses import dataclass
from multiprocessing import Pool, cpu_count
from multiprocessing.pool import Pool as PoolType
import time

from src.rendering import Camera, ProgressCallback, RenderConfig, Scene, Vec3, trace_ray


@dataclass(frozen=True)
class _ChunkTask:
    start_row: int
    end_row: int
    scene: Scene
    config: RenderConfig


def _render_chunk(task: _ChunkTask) -> tuple[int, list[list[Vec3]]]:
    camera = Camera.from_config(task.config.width, task.config.height, task.config.fov_degrees)
    chunk: list[list[Vec3]] = []
    for y in range(task.start_row, task.end_row):
        row: list[Vec3] = []
        v = 1.0 - (y + 0.5) / task.config.height
        for x in range(task.config.width):
            u = (x + 0.5) / task.config.width
            ray = camera.get_ray(u, v)
            row.append(trace_ray(ray, task.scene, depth=0, max_depth=task.config.max_bounces))
        chunk.append(row)
    return task.start_row, chunk


def _build_tasks(scene: Scene, config: RenderConfig, rows_per_chunk: int) -> list[_ChunkTask]:
    tasks: list[_ChunkTask] = []
    start = 0
    while start < config.height:
        end = min(config.height, start + rows_per_chunk)
        tasks.append(_ChunkTask(start_row=start, end_row=end, scene=scene, config=config))
        start = end
    return tasks


def render_image_parallel(
    scene: Scene,
    config: RenderConfig,
    processes: int | None = None,
    rows_per_chunk: int = 8,
    progress_callback: ProgressCallback | None = None,
    pool: PoolType | None = None,
) -> list[list[Vec3]]:
    rows_per_chunk = max(1, rows_per_chunk)
    tasks = _build_tasks(scene, config, rows_per_chunk)

    image: list[list[Vec3]] = [[Vec3(0.0, 0.0, 0.0) for _ in range(config.width)] for _ in range(config.height)]
    started = time.perf_counter()
    rows_done = 0

    if pool is None:
        if processes is None:
            processes = max(1, cpu_count() - 1)
        processes = max(1, processes)
        with Pool(processes=processes) as owned_pool:
            for start_row, chunk in owned_pool.imap_unordered(_render_chunk, tasks):
                for offset, row in enumerate(chunk):
                    image[start_row + offset] = row
                rows_done += len(chunk)
                if progress_callback is not None:
                    progress_callback(image, rows_done, config.height, time.perf_counter() - started)
    else:
        for start_row, chunk in pool.imap_unordered(_render_chunk, tasks):
            for offset, row in enumerate(chunk):
                image[start_row + offset] = row
            rows_done += len(chunk)
            if progress_callback is not None:
                progress_callback(image, rows_done, config.height, time.perf_counter() - started)
    return image
