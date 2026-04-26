# Parallel Ray Tracing in Python (CPU)

This project implements and compares two CPU ray tracing approaches in Python: a serial renderer and a multiprocessing parallel renderer.

## Requirements

- Python 3.10+ (tested on Python 3.13)
- Dependencies from `requirements.txt`

Install:

1. `python -m venv .venv`
2. Activate the virtual environment:
   - Windows: `.venv\\Scripts\\activate`
   - Linux/macOS: `source .venv/bin/activate`
3. `pip install -r requirements.txt`

## How to Run

### Main UI (recommended)

Run:

`python -m src.ui`

What it does:

- renders all project scenes with fixed settings
- runs both serial and parallel versions for each scene
- shows one image and timing results per scene
- writes `benchmark_results.json` automatically after the full run
  (single-run schema uses `seconds`, not `mean/std`)

Fixed settings used in this project:

- resolution: `640x360`
- field of view: `60`
- parallel worker count: `4`
- rows per chunk: `8`

## Computation Performed

For each pixel, the renderer:

1. generates a camera ray
2. computes object intersections (sphere/plane)
3. finds nearest hit point and surface normal
4. applies lighting (ambient + diffuse + specular)
5. casts shadow rays to test light visibility
6. writes final RGB color to the image buffer

The serial and parallel versions use the same rendering math and scene setup. Only execution strategy differs.

## Serial Approach

The serial algorithm in `src/serial.py`:

- processes image rows in a single process
- computes all pixels sequentially
- updates progress row by row

Characteristics:

- simple and deterministic baseline
- no inter-process overhead
- slower for heavy scenes because only one CPU core is used

## Parallel Approach

The parallel algorithm in `src/parallel.py`:

- splits the image into row chunks
- sends chunks to worker processes using multiprocessing
- merges returned chunks into the final image
- reuses one process pool across all scenes in UI batch mode

Characteristics:

- better throughput on compute-heavy scenes
- same output quality as serial approach
- includes multiprocessing overhead (startup, scheduling, data transfer)
