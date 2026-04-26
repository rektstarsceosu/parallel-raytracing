from __future__ import annotations

from dataclasses import dataclass
import math
import time
from typing import Optional, Protocol

import matplotlib.pyplot as plt
import numpy as np


@dataclass(frozen=True)
class Vec3:
    x: float
    y: float
    z: float

    def __add__(self, other: "Vec3") -> "Vec3":
        return Vec3(self.x + other.x, self.y + other.y, self.z + other.z)

    def __sub__(self, other: "Vec3") -> "Vec3":
        return Vec3(self.x - other.x, self.y - other.y, self.z - other.z)

    def __mul__(self, value: float) -> "Vec3":
        return Vec3(self.x * value, self.y * value, self.z * value)

    def __rmul__(self, value: float) -> "Vec3":
        return self * value

    def __truediv__(self, value: float) -> "Vec3":
        return Vec3(self.x / value, self.y / value, self.z / value)

    def dot(self, other: "Vec3") -> float:
        return self.x * other.x + self.y * other.y + self.z * other.z

    def norm(self) -> float:
        return math.sqrt(self.dot(self))

    def normalize(self) -> "Vec3":
        n = self.norm()
        if n == 0:
            return self
        return self / n

    def clamp(self, low: float = 0.0, high: float = 1.0) -> "Vec3":
        return Vec3(
            max(low, min(high, self.x)),
            max(low, min(high, self.y)),
            max(low, min(high, self.z)),
        )


@dataclass(frozen=True)
class Ray:
    origin: Vec3
    direction: Vec3

    def at(self, t: float) -> Vec3:
        return self.origin + self.direction * t


@dataclass(frozen=True)
class Material:
    color: Vec3
    ambient: float = 0.1
    diffuse: float = 0.7
    specular: float = 0.2
    shininess: float = 32.0


@dataclass(frozen=True)
class HitRecord:
    t: float
    point: Vec3
    normal: Vec3
    material: Material


class Hittable(Protocol):
    def intersect(self, ray: Ray, t_min: float, t_max: float) -> Optional[HitRecord]:
        ...


@dataclass(frozen=True)
class Sphere:
    center: Vec3
    radius: float
    material: Material

    def intersect(self, ray: Ray, t_min: float, t_max: float) -> Optional[HitRecord]:
        oc = ray.origin - self.center
        a = ray.direction.dot(ray.direction)
        half_b = oc.dot(ray.direction)
        c = oc.dot(oc) - self.radius * self.radius
        discriminant = half_b * half_b - a * c
        if discriminant < 0:
            return None
        sqrt_d = discriminant**0.5

        root = (-half_b - sqrt_d) / a
        if root < t_min or root > t_max:
            root = (-half_b + sqrt_d) / a
            if root < t_min or root > t_max:
                return None

        point = ray.at(root)
        normal = (point - self.center) / self.radius
        return HitRecord(t=root, point=point, normal=normal.normalize(), material=self.material)


@dataclass(frozen=True)
class Plane:
    point: Vec3
    normal: Vec3
    material: Material

    def intersect(self, ray: Ray, t_min: float, t_max: float) -> Optional[HitRecord]:
        denominator = self.normal.dot(ray.direction)
        if abs(denominator) < 1e-8:
            return None
        t = (self.point - ray.origin).dot(self.normal) / denominator
        if t < t_min or t > t_max:
            return None
        hit_point = ray.at(t)
        outward = self.normal.normalize()
        if outward.dot(ray.direction) > 0:
            outward = outward * -1.0
        return HitRecord(t=t, point=hit_point, normal=outward, material=self.material)


@dataclass(frozen=True)
class PointLight:
    position: Vec3
    intensity: float = 1.0
    color: Vec3 = Vec3(1.0, 1.0, 1.0)


@dataclass(frozen=True)
class Scene:
    objects: tuple[Hittable, ...]
    lights: tuple[PointLight, ...]
    background: Vec3 = Vec3(0.2, 0.2, 0.25)

    def hit(self, ray: Ray, t_min: float = 1e-4, t_max: float = float("inf")) -> Optional[HitRecord]:
        closest = t_max
        result: Optional[HitRecord] = None
        for obj in self.objects:
            hit = obj.intersect(ray, t_min=t_min, t_max=closest)
            if hit is not None:
                closest = hit.t
                result = hit
        return result


@dataclass(frozen=True)
class RenderConfig:
    width: int = 640
    height: int = 360
    fov_degrees: float = 60.0
    max_bounces: int = 1


@dataclass(frozen=True)
class Camera:
    origin: Vec3
    lower_left: Vec3
    horizontal: Vec3
    vertical: Vec3

    @classmethod
    def from_config(cls, width: int, height: int, fov_degrees: float, origin: Vec3 | None = None) -> "Camera":
        if origin is None:
            origin = Vec3(0.0, 0.0, 0.0)
        aspect_ratio = width / max(1, height)
        theta = fov_degrees * math.pi / 180.0
        half_height = theta / 2.0
        viewport_height = 2.0 * math.tan(half_height)
        viewport_width = aspect_ratio * viewport_height
        focal_length = 1.0
        horizontal = Vec3(viewport_width, 0.0, 0.0)
        vertical = Vec3(0.0, viewport_height, 0.0)
        lower_left = origin - horizontal / 2.0 - vertical / 2.0 - Vec3(0.0, 0.0, focal_length)
        return cls(origin=origin, lower_left=lower_left, horizontal=horizontal, vertical=vertical)

    def get_ray(self, u: float, v: float) -> Ray:
        direction = (self.lower_left + self.horizontal * u + self.vertical * v - self.origin).normalize()
        return Ray(origin=self.origin, direction=direction)


def reflect(v: Vec3, n: Vec3) -> Vec3:
    return v - n * (2.0 * v.dot(n))


def trace_ray(ray: Ray, scene: Scene, depth: int, max_depth: int) -> Vec3:
    hit = scene.hit(ray)
    if hit is None:
        return scene.background

    ambient = hit.material.color * hit.material.ambient
    color = ambient
    view_dir = (ray.direction * -1.0).normalize()

    for light in scene.lights:
        to_light = light.position - hit.point
        distance_to_light = to_light.norm()
        light_dir = to_light.normalize()
        shadow_ray = Ray(hit.point + hit.normal * 1e-4, light_dir)
        occluder = scene.hit(shadow_ray, t_min=1e-4, t_max=distance_to_light - 1e-4)
        if occluder is not None:
            continue

        diffuse_term = max(0.0, hit.normal.dot(light_dir))
        diffuse = hit.material.color * (hit.material.diffuse * diffuse_term * light.intensity)

        reflected = reflect(light_dir * -1.0, hit.normal).normalize()
        spec_term = max(0.0, view_dir.dot(reflected)) ** hit.material.shininess
        specular = light.color * (hit.material.specular * spec_term * light.intensity)
        color = color + diffuse + specular

    if depth >= max_depth:
        return color.clamp()
    return color.clamp()


def simple_sphere_scene() -> Scene:
    red = Material(color=Vec3(0.9, 0.2, 0.2), specular=0.25)
    gray = Material(color=Vec3(0.8, 0.8, 0.8), diffuse=0.8, specular=0.1)
    objects = (
        Sphere(center=Vec3(0.0, 0.0, -3.0), radius=0.8, material=red),
        Plane(point=Vec3(0.0, -0.8, 0.0), normal=Vec3(0.0, 1.0, 0.0), material=gray),
    )
    lights = (PointLight(position=Vec3(-2.0, 3.0, 0.0), intensity=1.0),)
    return Scene(objects=objects, lights=lights, background=Vec3(0.15, 0.2, 0.3))


def two_spheres_scene() -> Scene:
    blue = Material(color=Vec3(0.2, 0.4, 0.9), specular=0.35)
    green = Material(color=Vec3(0.3, 0.85, 0.4), specular=0.2)
    floor = Material(color=Vec3(0.85, 0.85, 0.85), diffuse=0.8)
    objects = (
        Sphere(center=Vec3(-0.8, -0.1, -3.5), radius=0.7, material=blue),
        Sphere(center=Vec3(0.9, 0.0, -2.8), radius=0.6, material=green),
        Plane(point=Vec3(0.0, -0.9, 0.0), normal=Vec3(0.0, 1.0, 0.0), material=floor),
    )
    lights = (
        PointLight(position=Vec3(-3.0, 3.5, -0.5), intensity=1.1),
        PointLight(position=Vec3(2.0, 2.0, 0.5), intensity=0.5),
    )
    return Scene(objects=objects, lights=lights, background=Vec3(0.1, 0.15, 0.2))


def corridor_scene() -> Scene:
    wall = Material(color=Vec3(0.7, 0.7, 0.8), diffuse=0.75, specular=0.05)
    shiny = Material(color=Vec3(0.9, 0.75, 0.2), specular=0.4)
    matte = Material(color=Vec3(0.2, 0.6, 0.6), specular=0.15)
    objects = (
        Plane(point=Vec3(0.0, -1.0, 0.0), normal=Vec3(0.0, 1.0, 0.0), material=wall),
        Plane(point=Vec3(0.0, 0.0, -8.0), normal=Vec3(0.0, 0.0, 1.0), material=wall),
        Sphere(center=Vec3(-1.3, -0.2, -4.0), radius=0.8, material=shiny),
        Sphere(center=Vec3(1.1, -0.5, -3.0), radius=0.5, material=matte),
        Sphere(center=Vec3(0.1, -0.3, -5.5), radius=0.7, material=matte),
    )
    lights = (PointLight(position=Vec3(0.0, 2.5, -1.0), intensity=1.2),)
    return Scene(objects=objects, lights=lights, background=Vec3(0.08, 0.08, 0.1))


def many_spheres_scene() -> Scene:
    floor = Material(color=Vec3(0.75, 0.75, 0.75), diffuse=0.85, specular=0.05)
    objects = [Plane(point=Vec3(0.0, -1.1, 0.0), normal=Vec3(0.0, 1.0, 0.0), material=floor)]
    for i in range(-3, 4):
        for j in range(3):
            color = Vec3(0.2 + 0.1 * (i + 3), 0.2 + 0.25 * j, 0.7 - 0.08 * (i + 3))
            mat = Material(color=color.clamp(), diffuse=0.7, specular=0.2, shininess=24.0)
            objects.append(Sphere(center=Vec3(i * 0.55, -0.65 + j * 0.45, -3.2 - j * 0.9), radius=0.25, material=mat))
    lights = (
        PointLight(position=Vec3(-2.5, 3.0, 0.0), intensity=1.0),
        PointLight(position=Vec3(2.5, 2.2, -1.5), intensity=0.8),
    )
    return Scene(objects=tuple(objects), lights=lights, background=Vec3(0.06, 0.07, 0.1))


def stress_scene() -> Scene:
    objects = []
    floor = Material(color=Vec3(0.8, 0.8, 0.8), diffuse=0.85, specular=0.05)
    objects.append(Plane(point=Vec3(0.0, -1.0, 0.0), normal=Vec3(0.0, 1.0, 0.0), material=floor))
    for x in range(-4, 5):
        for z in range(4, 13):
            radius = 0.17 + ((x + z) % 3) * 0.04
            color = Vec3(0.3 + (x + 4) * 0.06, 0.2 + (z - 4) * 0.035, 0.5)
            mat = Material(color=color.clamp(), diffuse=0.7, specular=0.2)
            objects.append(Sphere(center=Vec3(x * 0.35, -0.8 + radius, -float(z)), radius=radius, material=mat))
    lights = (
        PointLight(position=Vec3(-3.0, 5.0, 1.0), intensity=1.0),
        PointLight(position=Vec3(2.5, 3.5, -4.0), intensity=0.8),
    )
    return Scene(objects=tuple(objects), lights=lights, background=Vec3(0.03, 0.04, 0.07))


SCENE_PRESETS = {
    "simple": simple_sphere_scene,
    "double": two_spheres_scene,
    "corridor": corridor_scene,
    "many": many_spheres_scene,
    "stress": stress_scene,
}


class ProgressCallback(Protocol):
    def __call__(self, image: list[list[Vec3]], rows_done: int, total_rows: int, elapsed_seconds: float) -> None:
        ...


class LivePreview:
    def __init__(self, width: int, height: int, update_interval_rows: int = 8) -> None:
        self.update_interval_rows = max(1, update_interval_rows)
        self._last_rows = 0
        self._fig, self._ax = plt.subplots()
        self._buffer = np.zeros((height, width, 3), dtype=np.float32)
        self._img = self._ax.imshow(self._buffer, vmin=0.0, vmax=1.0)
        self._ax.set_title("Ray tracing live preview")
        self._ax.set_axis_off()
        self._text = self._ax.text(
            0.01,
            0.99,
            "rows: 0/0\ntime: 0.00s",
            transform=self._ax.transAxes,
            va="top",
            ha="left",
            color="white",
            bbox={"facecolor": "black", "alpha": 0.5, "pad": 4},
        )
        plt.ion()
        plt.show(block=False)

    def _copy_image(self, image: list[list[Vec3]]) -> None:
        for y, row in enumerate(image):
            for x, col in enumerate(row):
                self._buffer[y, x, 0] = col.x
                self._buffer[y, x, 1] = col.y
                self._buffer[y, x, 2] = col.z

    def update(self, image: list[list[Vec3]], rows_done: int, total_rows: int, elapsed_seconds: float) -> None:
        should_draw = rows_done == total_rows or (rows_done - self._last_rows) >= self.update_interval_rows
        if not should_draw:
            return
        self._last_rows = rows_done
        self._copy_image(image)
        self._img.set_data(self._buffer)
        self._text.set_text(f"rows: {rows_done}/{total_rows}\ntime: {elapsed_seconds:.2f}s")
        self._fig.canvas.draw_idle()
        self._fig.canvas.flush_events()
        plt.pause(0.001)

    def close_after_delay(self, seconds: float = 1.0) -> None:
        end = time.perf_counter() + seconds
        while time.perf_counter() < end:
            self._fig.canvas.flush_events()
            plt.pause(0.05)
