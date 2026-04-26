from __future__ import annotations

import json
from pathlib import Path
import queue
import tempfile
import threading
import time
import tkinter as tk
from multiprocessing import Pool
from tkinter import ttk

import matplotlib.pyplot as plt
import numpy as np

from src.parallel import render_image_parallel
from src.rendering import RenderConfig, SCENE_PRESETS, Vec3
from src.serial import render_image


class RayTracingUI:
    def __init__(self, root: tk.Tk) -> None:
        self.root = root
        self.root.title("Parallel Ray Tracing - All Scenes Dashboard")
        self.root.geometry("720x480")

        self.fixed_width = 640
        self.fixed_height = 360
        self.fixed_processes = 4
        self.fixed_rows_per_chunk = 8

        self.status_var = tk.StringVar(
            value=(
                f"Ready. Fixed configuration: {self.fixed_width}x{self.fixed_height}, "
                f"parallel processes={self.fixed_processes}, rows_per_chunk={self.fixed_rows_per_chunk}"
            )
        )

        self._progress_queue: queue.Queue[tuple[str, dict[str, object]]] = queue.Queue()
        self._active_thread: threading.Thread | None = None
        self._image_refs: list[tk.PhotoImage] = []

        self._build_layout()
        self.root.after(100, self._drain_progress_queue)

    def _build_layout(self) -> None:
        top_bar = ttk.Frame(self.root, padding=12)
        top_bar.pack(fill=tk.X)
        ttk.Button(top_bar, text="Render All Scenes (Sequential + Parallel)", command=self._start_render_all).pack(side=tk.LEFT)
        ttk.Label(top_bar, textvariable=self.status_var).pack(side=tk.LEFT, padx=12)

        container = ttk.Frame(self.root)
        container.pack(fill=tk.BOTH, expand=True)

        self.canvas = tk.Canvas(container)
        scrollbar = ttk.Scrollbar(container, orient="vertical", command=self.canvas.yview)
        self.results_frame = ttk.Frame(self.canvas, padding=12)
        self.canvas_window = self.canvas.create_window((0, 0), window=self.results_frame, anchor="nw")

        def _on_results_configure(_: tk.Event) -> None:
            self.canvas.configure(scrollregion=self.canvas.bbox("all"))

        def _on_canvas_configure(event: tk.Event) -> None:
            self.canvas.itemconfig(self.canvas_window, width=event.width)

        self.results_frame.bind("<Configure>", _on_results_configure)
        self.canvas.bind("<Configure>", _on_canvas_configure)
        self.canvas.configure(yscrollcommand=scrollbar.set)

        self.canvas.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        self._bind_mousewheel_scroll()

    def _bind_mousewheel_scroll(self) -> None:
        def _on_mousewheel(event: tk.Event) -> None:
            delta = getattr(event, "delta", 0)
            if delta != 0:
                self.canvas.yview_scroll(int(-delta / 120), "units")
            else:
                # Linux fallback events use num=4 (up) and num=5 (down).
                num = getattr(event, "num", 0)
                if num == 4:
                    self.canvas.yview_scroll(-1, "units")
                elif num == 5:
                    self.canvas.yview_scroll(1, "units")

        self.canvas.bind_all("<MouseWheel>", _on_mousewheel)
        self.canvas.bind_all("<Button-4>", _on_mousewheel)
        self.canvas.bind_all("<Button-5>", _on_mousewheel)

    def _set_busy(self, busy: bool) -> None:
        self.root.config(cursor="watch" if busy else "")

    def _start_render_all(self) -> None:
        if self._active_thread and self._active_thread.is_alive():
            self.status_var.set("A render is already running.")
            return
        self._set_busy(True)
        self.status_var.set("Rendering all scenes with fixed settings...")
        self._active_thread = threading.Thread(target=self._render_all_worker, daemon=True)
        self._active_thread.start()

    def _config(self) -> RenderConfig:
        return RenderConfig(
            width=self.fixed_width,
            height=self.fixed_height,
            fov_degrees=60.0,
            max_bounces=1,
        )

    def _progress_cb(self, scene_name: str, mode: str, rows_done: int, total_rows: int, elapsed: float) -> None:
        self._progress_queue.put(
            (
                "progress",
                {
                    "scene_name": scene_name,
                    "mode": mode,
                    "rows_done": rows_done,
                    "total_rows": total_rows,
                    "elapsed": elapsed,
                },
            )
        )

    @staticmethod
    def _save_png(image: list[list[Vec3]]) -> str:
        height = len(image)
        width = len(image[0]) if height else 0
        rgb = np.zeros((height, width, 3), dtype=np.float32)
        for y, row in enumerate(image):
            for x, col in enumerate(row):
                c = col.clamp()
                rgb[y, x, 0] = c.x
                rgb[y, x, 1] = c.y
                rgb[y, x, 2] = c.z
        with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tmp:
            png_path = tmp.name
        plt.imsave(png_path, rgb)
        return png_path

    @staticmethod
    def _clear_children(frame: ttk.Frame) -> None:
        for child in frame.winfo_children():
            child.destroy()

    @staticmethod
    def _benchmark_output_path() -> Path:
        return Path(__file__).resolve().parent.parent / "benchmark_results.json"

    @classmethod
    def _save_benchmark_results(cls, rows: list[dict[str, object]]) -> Path:
        output_path = cls._benchmark_output_path()
        with output_path.open("w", encoding="utf-8") as f:
            json.dump(rows, f, indent=2)
        return output_path

    def _render_all_worker(self) -> None:
        try:
            cfg = self._config()
            scene_names = [name for name in sorted(SCENE_PRESETS.keys()) if name != "stress"]
            all_results: list[dict[str, object]] = []
            benchmark_rows: list[dict[str, object]] = []
            # Reuse one pool across all scenes to avoid worker startup per render.
            with Pool(processes=self.fixed_processes) as shared_pool:
                for scene_name in scene_names:
                    scene_seq = SCENE_PRESETS[scene_name]()
                    seq_start = time.perf_counter()
                    seq_image = render_image(
                        scene_seq,
                        cfg,
                        progress_callback=lambda _, rows, total, elapsed, s=scene_name: self._progress_cb(
                            s, "sequential", rows, total, elapsed
                        ),
                    )
                    seq_time = time.perf_counter() - seq_start
                    seq_png = self._save_png(seq_image)

                    scene_par = SCENE_PRESETS[scene_name]()
                    par_start = time.perf_counter()
                    par_image = render_image_parallel(
                        scene_par,
                        cfg,
                        rows_per_chunk=self.fixed_rows_per_chunk,
                        progress_callback=lambda _, rows, total, elapsed, s=scene_name: self._progress_cb(
                            s, "parallel", rows, total, elapsed
                        ),
                        pool=shared_pool,
                    )
                    par_time = time.perf_counter() - par_start

                    speedup = seq_time / max(par_time, 1e-9)
                    all_results.append(
                        {
                            "scene": scene_name,
                            "seq_time": seq_time,
                            "par_time": par_time,
                            "speedup": speedup,
                            "image_path": seq_png,
                        }
                    )
                    benchmark_rows.append(
                        {
                            "scene": scene_name,
                            "width": self.fixed_width,
                            "height": self.fixed_height,
                            "mode": "sequential",
                            "processes": 1,
                            "seconds": seq_time,
                            "speedup": 1.0,
                        }
                    )
                    benchmark_rows.append(
                        {
                            "scene": scene_name,
                            "width": self.fixed_width,
                            "height": self.fixed_height,
                            "mode": "parallel",
                            "processes": self.fixed_processes,
                            "seconds": par_time,
                            "speedup": speedup,
                        }
                    )

            output_path = self._save_benchmark_results(benchmark_rows)
            self._progress_queue.put(("done_all", {"results": all_results, "output_path": str(output_path)}))
        except Exception as exc:  # pragma: no cover - UI error path
            self._progress_queue.put(("error", {"message": str(exc)}))

    def _build_scene_result_card(self, scene_result: dict[str, object]) -> None:
        card = ttk.Frame(self.results_frame, padding=10)
        card.pack(fill=tk.X, pady=(0, 14))

        scene_name = str(scene_result["scene"])
        seq_time = float(scene_result["seq_time"])
        par_time = float(scene_result["par_time"])
        speedup = float(scene_result["speedup"])
        image_path = str(scene_result["image_path"])

        ttk.Label(card, text=f"Scene: {scene_name}", font=("TkDefaultFont", 11, "bold")).pack(anchor="w", pady=(0, 6))

        image = tk.PhotoImage(file=image_path)
        self._image_refs.append(image)
        ttk.Label(card, image=image).pack(anchor="w")

        ttk.Label(
            card,
            text=(
                f"Sequential: {seq_time:.3f}s | Parallel: {par_time:.3f}s | "
                f"Speedup: {speedup:.3f}x | Resolution: {self.fixed_width}x{self.fixed_height}"
            ),
        ).pack(anchor="w", pady=(8, 0))

    def _drain_progress_queue(self) -> None:
        try:
            while True:
                event, payload = self._progress_queue.get_nowait()
                if event == "progress":
                    self.status_var.set(
                        f"Rendering {payload['scene_name']} [{payload['mode']}] - "
                        f"rows {payload['rows_done']}/{payload['total_rows']}, "
                        f"time {float(payload['elapsed']):.2f}s"
                    )
                elif event == "done_all":
                    self._clear_children(self.results_frame)
                    self._image_refs.clear()
                    for result in payload["results"]:  # type: ignore[index]
                        self._build_scene_result_card(result)
                    self._set_busy(False)
                    self.status_var.set(
                        "Completed all scenes. "
                        f"Saved benchmark_results.json to {payload['output_path']}."
                    )
                elif event == "error":
                    self._set_busy(False)
                    self.status_var.set("Error during run.")
                    ttk.Label(self.results_frame, text=f"Error: {payload['message']}").pack(anchor="w")
        except queue.Empty:
            pass
        self.root.after(100, self._drain_progress_queue)


def main() -> None:
    root = tk.Tk()
    RayTracingUI(root)
    root.mainloop()


if __name__ == "__main__":
    main()
