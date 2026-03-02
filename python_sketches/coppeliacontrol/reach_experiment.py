"""
reach_experiment.py
───────────────────
Reusable experiment module for the YuMi joystick controller.

Defines:
  • ReachTarget   – a single spatial goal (sphere in sim + HUD feedback)
  • Experiment    – orchestrates a sequence of ReachTargets, logs results

Usage (minimal):
    from reach_experiment import Experiment

    exp = Experiment(sim, trials=[
        {"pos": [0.4, 0.1, 0.8], "radius": 0.04},
        {"pos": [0.3, -0.1, 0.9], "radius": 0.03},
    ])

    # inside your sim loop, after computing wrist_pos each step:
    exp.update(wrist_pos, dt=1/60)

    # inside your pygame draw pass:
    exp.draw(screen, fonts)

    # check exp.finished to know when all trials are done
"""

from __future__ import annotations
import time
import math
import random
import pygame
import numpy as np
from dataclasses import dataclass, field
from typing import List, Optional, Tuple


# ─────────────────────────────────────────────────────────────────────────────
# Colour palette (matches the controller UI feel)
# ─────────────────────────────────────────────────────────────────────────────
C_IDLE = (100, 160, 220)  # calm blue  – far from target
C_WARM = (220, 190, 60)  # amber      – getting close
C_HOT = (100, 220, 130)  # green      – inside target zone
C_SUCCESS = (100, 220, 130)
C_FAIL = (220, 80, 80)
C_TEXT_DIM = (130, 130, 130)
C_TEXT_BRT = (210, 210, 210)
C_PANEL_BG = (22, 22, 28, 200)  # semi-transparent (needs Surface)


# ─────────────────────────────────────────────────────────────────────────────
# Hemisphere sampler
# ─────────────────────────────────────────────────────────────────────────────
def sample_hemisphere_positions(
    shoulder,
    arm_length,
    n,
    min_reach=0.4,
    max_reach=0.85,
    min_elevation=-10.0,
    max_elevation=80.0,
    az_min=-45.0,
    az_max=45.0,
    az_centre=-90.0,
    seed=None,
):
    rng = random.Random(seed)
    positions = []
    r_min = arm_length * min_reach
    r_max = arm_length * max_reach
    el_lo = math.radians(min_elevation)
    el_hi = math.radians(max_elevation)
    centre_rad = math.radians(az_centre)
    az_lo = centre_rad + math.radians(az_min)
    az_hi = centre_rad + math.radians(az_max)
    while len(positions) < n:
        el = rng.uniform(el_lo, el_hi)
        az = rng.uniform(az_lo, az_hi)
        r = rng.uniform(r_min, r_max)
        x = shoulder[0] + r * math.cos(el) * math.cos(az)
        y = shoulder[1] + r * math.cos(el) * math.sin(az)
        z = shoulder[2] + r * math.sin(el)
        positions.append([x, y, z])
    return positions


# ─────────────────────────────────────────────────────────────────────────────
# ReachTarget
# ─────────────────────────────────────────────────────────────────────────────
@dataclass
class ReachTarget:
    sim: object
    position: List[float]
    radius: float = 0.04
    dwell_time: float = 0.5
    timeout: float = 15.0
    label: str = "Target"

    _dummy: Optional[int] = field(default=None, init=False, repr=False)
    _elapsed: float = field(default=0.0, init=False, repr=False)
    _dwell_acc: float = field(default=0.0, init=False, repr=False)
    _inside: bool = field(default=False, init=False, repr=False)
    _result: Optional[str] = field(default=None, init=False, repr=False)
    _flash_t: float = field(default=0.0, init=False, repr=False)
    _started: bool = field(default=False, init=False, repr=False)
    _start_wrist: Optional[List[float]] = field(default=None, init=False, repr=False)
    _settle_frames: int = field(default=0, init=False, repr=False)
    MOVE_THRESHOLD: float = field(default=0.02, init=False, repr=False)
    SETTLE_FRAMES: int = field(default=10, init=False, repr=False)

    def __post_init__(self):
        self._spawn_dummy()

    def _spawn_dummy(self):
        sim = self.sim
        self._dummy = sim.createDummy(self.radius * 2)
        if self._dummy is None or self._dummy < 0:
            raise RuntimeError(
                f"createDummy returned invalid handle ({self._dummy}). "
                "Ensure the simulation is running before creating an Experiment."
            )
        sim.setObjectPosition(self._dummy, self.position)
        sim.setObjectAlias(self._dummy, self.label)

    def remove(self):
        if self._dummy is not None:
            try:
                self.sim.removeObject(self._dummy)
            except Exception:
                pass
            self._dummy = None

    def update(self, wrist_pos: List[float], dt: float) -> Optional[str]:
        if self._result is not None:
            # Keep counting down the flash even after result is set,
            # so _maybe_advance knows when the animation is done.
            self._flash_t = max(0.0, self._flash_t - dt)
            return self._result

        if not self._started:
            self._settle_frames += 1
            if self._settle_frames <= self.SETTLE_FRAMES:
                return None
            if self._start_wrist is None:
                self._start_wrist = list(wrist_pos)
            moved = math.sqrt(
                sum((wrist_pos[i] - self._start_wrist[i]) ** 2 for i in range(3))
            )
            if moved >= self.MOVE_THRESHOLD:
                self._started = True
            dist = math.sqrt(
                sum((wrist_pos[i] - self.position[i]) ** 2 for i in range(3))
            )
            self._inside = dist <= self.radius
            return None

        self._elapsed += dt

        dist = math.sqrt(sum((wrist_pos[i] - self.position[i]) ** 2 for i in range(3)))
        self._inside = dist <= self.radius

        if self._inside:
            self._dwell_acc += dt
            if self._dwell_acc >= self.dwell_time:
                self._result = "success"
                self._flash_t = 0.6
                self.remove()
        else:
            self._dwell_acc = max(0.0, self._dwell_acc - dt * 2)

        if self.timeout > 0 and self._elapsed >= self.timeout:
            if self._result is None:  # don't overwrite a success
                self._result = "timeout"
                self._flash_t = 0.6
                self.remove()

        return self._result

    @property
    def finished(self) -> bool:
        return self._result is not None

    @property
    def success(self) -> bool:
        return self._result == "success"

    @property
    def time_remaining(self) -> float:
        if self.timeout <= 0:
            return float("inf")
        if not self._started:
            return self.timeout
        return max(0.0, self.timeout - self._elapsed)

    @property
    def dwell_fraction(self) -> float:
        return min(1.0, self._dwell_acc / self.dwell_time)

    def distance_to(self, wrist_pos: List[float]) -> float:
        return math.sqrt(sum((wrist_pos[i] - self.position[i]) ** 2 for i in range(3)))

    def draw(
        self, surf: pygame.Surface, wrist_pos: List[float], fonts: dict, dt: float = 0.0
    ):
        if self.finished and self._flash_t <= 0:
            return

        dist = self.distance_to(wrist_pos)
        W, H = surf.get_size()

        if self._flash_t > 0 and self._result:
            alpha = int(160 * (self._flash_t / 0.6))
            col = C_SUCCESS if self._result == "success" else C_FAIL
            flash = pygame.Surface((W, H), pygame.SRCALPHA)
            flash.fill((*col, alpha))
            surf.blit(flash, (0, 0))
            msg = "✓  TARGET REACHED" if self._result == "success" else "✗  TIMED OUT"
            lbl = fonts["lg"].render(msg, True, (255, 255, 255))
            surf.blit(
                lbl, (W // 2 - lbl.get_width() // 2, H // 2 - lbl.get_height() // 2)
            )
            return

        ratio = min(1.0, dist / (self.radius * 6))
        if ratio < 0.4:
            hud_col = _lerp_col(C_HOT, C_WARM, ratio / 0.4)
        else:
            hud_col = _lerp_col(C_WARM, C_IDLE, (ratio - 0.4) / 0.6)

        panel_w, panel_h = 260, 110
        px, py = W - panel_w - 10, 250
        panel = pygame.Surface((panel_w, panel_h), pygame.SRCALPHA)
        panel.fill((22, 22, 28, 190))
        surf.blit(panel, (px, py))
        pygame.draw.rect(surf, hud_col, (px, py, panel_w, panel_h), 1)

        surf.blit(fonts["md"].render(self.label, True, C_TEXT_BRT), (px + 10, py + 8))
        dist_txt = f"dist  {dist * 100:.1f} cm"
        surf.blit(fonts["sm"].render(dist_txt, True, hud_col), (px + 10, py + 28))

        bar_x, bar_y = px + 10, py + 50
        bar_w, bar_h = panel_w - 20, 10
        fill = int(bar_w * (1.0 - ratio))
        pygame.draw.rect(surf, (55, 55, 55), (bar_x, bar_y, bar_w, bar_h))
        if fill > 0:
            pygame.draw.rect(surf, hud_col, (bar_x, bar_y, fill, bar_h))

        if self._inside:
            ring_cx, ring_cy = px + panel_w - 28, py + 72
            ring_r = 18
            pygame.draw.circle(surf, (55, 55, 55), (ring_cx, ring_cy), ring_r, 3)
            arc_rect = pygame.Rect(
                ring_cx - ring_r, ring_cy - ring_r, ring_r * 2, ring_r * 2
            )
            end_angle = -math.pi / 2 + self.dwell_fraction * 2 * math.pi
            _draw_arc(surf, C_HOT, arc_rect, -math.pi / 2, end_angle, 3)
            pct = fonts["sm"].render(f"{int(self.dwell_fraction*100)}%", True, C_HOT)
            surf.blit(
                pct, (ring_cx - pct.get_width() // 2, ring_cy - pct.get_height() // 2)
            )

        if self.timeout > 0:
            if not self._started:
                t_txt = fonts["sm"].render("move to start timer", True, (160, 140, 60))
            else:
                tr = self.time_remaining
                t_col = C_FAIL if tr < 3.0 else C_TEXT_DIM
                t_txt = fonts["sm"].render(f"time  {tr:.1f}s", True, t_col)
            surf.blit(t_txt, (px + 10, py + 68 + 20))


# ─────────────────────────────────────────────────────────────────────────────
# Experiment  – sequence of ReachTargets
# ─────────────────────────────────────────────────────────────────────────────
class Experiment:
    def __init__(self, sim, trials: List[dict]):
        self.sim = sim
        self._trial_defs = trials
        self._index = 0
        self._results: List[dict] = []
        self._active: Optional[ReachTarget] = None
        self._start_time = time.time()
        self._trial_start = time.time()
        self._result_logged = False
        self._spawn_next()

    @classmethod
    def from_hemisphere(
        cls,
        sim,
        shoulder_pos: List[float],
        arm_length: float,
        n_trials: int = 6,
        radius: float = 0.04,
        dwell_time: float = 0.5,
        timeout: float = 20.0,
        seed: int = None,
        min_reach: float = 0.4,
        max_reach: float = 0.85,
        min_elevation: float = -20.0,
        max_elevation: float = 60.0,
        az_min: float = -45.0,
        az_max: float = 45.0,
        az_centre: float = -90.0,
    ) -> "Experiment":
        positions = sample_hemisphere_positions(
            shoulder=shoulder_pos,
            arm_length=arm_length,
            n=n_trials,
            min_reach=min_reach,
            max_reach=max_reach,
            min_elevation=min_elevation,
            max_elevation=max_elevation,
            az_min=az_min,
            az_max=az_max,
            az_centre=az_centre,
            seed=seed,
        )
        trials = [
            {
                "pos": pos,
                "radius": radius,
                "dwell_time": dwell_time,
                "timeout": timeout,
                "label": f"Trial {i+1}",
            }
            for i, pos in enumerate(positions)
        ]
        return cls(sim, trials)

    # ── spawn ─────────────────────────────────────────────────────────────────
    def _spawn_next(self):
        if self._index >= len(self._trial_defs):
            self._active = None
            return
        cfg = self._trial_defs[self._index]
        self._active = ReachTarget(
            sim=self.sim,
            position=cfg["pos"],
            radius=cfg.get("radius", 0.04),
            dwell_time=cfg.get("dwell_time", 0.5),
            timeout=cfg.get("timeout", 15.0),
            label=cfg.get("label", f"Trial {self._index + 1}"),
        )
        self._trial_start = time.time()

    # ── update ────────────────────────────────────────────────────────────────
    def update(self, wrist_pos: List[float], dt: float):
        """Call once per sim step."""
        if self._active is None:
            return

        result = self._active.update(wrist_pos, dt)

        # Log result the first time it appears
        if result is not None and not self._result_logged:
            self._result_logged = True
            self._results.append(
                {
                    "trial": self._index + 1,
                    "label": self._active.label,
                    "result": result,
                    "duration": time.time() - self._trial_start,
                }
            )
            self._index += 1

        # FIX: advance to the next trial once the flash animation finishes.
        # Previously this was only called from draw() (the pygame path), so
        # the OpenCV path would log the result and increment _index but never
        # spawn the next target — leaving _active stuck on the finished trial.
        self._maybe_advance()

    def _maybe_advance(self):
        """Spawn next trial once the current flash animation has finished."""
        if (
            self._active is not None
            and self._active.finished
            and self._active._flash_t <= 0
        ):
            self._result_logged = False
            self._spawn_next()

    # ── draw (pygame path — unchanged) ────────────────────────────────────────
    def draw(
        self, surf: pygame.Surface, wrist_pos: List[float], fonts: dict, dt: float = 0.0
    ):
        """Call once per pygame frame inside your draw pass."""
        # _maybe_advance is now driven by update(), but calling it here too is
        # harmless and keeps the pygame path working identically to before.
        self._maybe_advance()

        W, H = surf.get_size()

        total = len(self._trial_defs)
        done = len(self._results)
        bar_w = W - 20
        pygame.draw.rect(surf, (45, 45, 55), (10, H - 18, bar_w, 8))
        if total:
            filled = int(bar_w * done / total)
            pygame.draw.rect(surf, C_SUCCESS, (10, H - 18, filled, 8))
        prog_txt = fonts["sm"].render(
            f"trial {min(done+1, total)} / {total}  ·  "
            + (f"{done} done" if done else ""),
            True,
            C_TEXT_DIM,
        )
        surf.blit(prog_txt, (10, H - 34))

        if self._active is not None:
            self._active.draw(surf, wrist_pos, fonts, dt)

        if self.finished:
            _draw_summary(surf, self._results, fonts)

    # ── properties ────────────────────────────────────────────────────────────
    @property
    def finished(self) -> bool:
        return self._active is None and self._index >= len(self._trial_defs)

    @property
    def results(self) -> List[dict]:
        return list(self._results)

    def summary(self) -> str:
        lines = ["─── Experiment Results ───"]
        for r in self._results:
            lines.append(
                f"  {r['trial']:2d}. {r['label']:20s}  "
                f"{r['result']:8s}  {r['duration']:.2f}s"
            )
        n_ok = sum(1 for r in self._results if r["result"] == "success")
        lines.append(
            f"\n  {n_ok}/{len(self._results)} succeeded  "
            f"| total {time.time()-self._start_time:.1f}s"
        )
        return "\n".join(lines)


# ─────────────────────────────────────────────────────────────────────────────
# Drawing utilities (private)
# ─────────────────────────────────────────────────────────────────────────────
def _lerp_col(a, b, t):
    t = max(0.0, min(1.0, t))
    return tuple(int(a[i] + (b[i] - a[i]) * t) for i in range(3))


def _draw_arc(surf, colour, rect, start_angle, end_angle, width=2):
    if abs(end_angle - start_angle) < 0.01:
        return
    try:
        pygame.draw.arc(surf, colour, rect, start_angle, end_angle, width)
    except Exception:
        pass


def _draw_summary(surf: pygame.Surface, results: List[dict], fonts: dict):
    W, H = surf.get_size()
    overlay = pygame.Surface((W, H), pygame.SRCALPHA)
    overlay.fill((10, 10, 18, 200))
    surf.blit(overlay, (0, 0))

    title = fonts["lg"].render("EXPERIMENT COMPLETE", True, C_SUCCESS)
    surf.blit(title, (W // 2 - title.get_width() // 2, 60))

    n_ok = sum(1 for r in results if r["result"] == "success")
    score = fonts["md"].render(
        f"{n_ok} / {len(results)}  trials succeeded", True, C_TEXT_BRT
    )
    surf.blit(score, (W // 2 - score.get_width() // 2, 100))

    for i, r in enumerate(results):
        col = C_SUCCESS if r["result"] == "success" else C_FAIL
        txt = (
            f"  {r['trial']:2d}.  {r['label']:18s}  "
            f"{r['result']:8s}  {r['duration']:.2f}s"
        )
        lbl = fonts["sm"].render(txt, True, col)
        surf.blit(lbl, (W // 2 - lbl.get_width() // 2, 135 + i * 20))
