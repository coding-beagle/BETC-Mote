"""
experiment_io.py
================
Factory functions for creating experiments and saving results to CSV.
"""

import csv
import datetime
import sys, os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from reach_experiment import Experiment, TransportExperiment

from .config import (
    ROBOT_ARM_LENGTH,
    EXP_N_TRIALS,
    EXP_RADIUS,
    EXP_DWELL_TIME,
    EXP_TIMEOUT,
    EXP_MIN_REACH,
    EXP_MAX_REACH,
    EXP_MIN_ELEVATION,
    EXP_MAX_ELEVATION,
    EXP_AZ_MIN,
    EXP_AZ_MAX,
    EXP_SEED,
    TRANSPORT_N_TRIALS,
    TRANSPORT_PICK_RADIUS,
    TRANSPORT_DROP_RADIUS,
    TRANSPORT_TIMEOUT,
    TRANSPORT_MIN_REACH,
    TRANSPORT_MAX_REACH,
    TRANSPORT_MIN_ELEV,
    TRANSPORT_MAX_ELEV,
    TRANSPORT_AZ_MIN,
    TRANSPORT_AZ_MAX,
    TRANSPORT_SEED,
    MODE_REACH,
    MODE_TRANSPORT,
)


# ── factories ─────────────────────────────────────────────────────────────────


def make_reach_experiment(sim, robot_shoulder_world):
    return Experiment.from_hemisphere(
        sim,
        shoulder_pos=robot_shoulder_world,
        arm_length=ROBOT_ARM_LENGTH,
        n_trials=EXP_N_TRIALS,
        radius=EXP_RADIUS,
        dwell_time=EXP_DWELL_TIME,
        timeout=EXP_TIMEOUT,
        min_reach=EXP_MIN_REACH,
        max_reach=EXP_MAX_REACH,
        min_elevation=EXP_MIN_ELEVATION,
        max_elevation=EXP_MAX_ELEVATION,
        az_min=EXP_AZ_MIN,
        az_max=EXP_AZ_MAX,
        seed=EXP_SEED,
    )


def make_transport_experiment(sim, robot_shoulder_world, start_pos=None):
    return TransportExperiment.from_random(
        sim,
        shoulder_pos=robot_shoulder_world,
        arm_length=ROBOT_ARM_LENGTH,
        n_trials=TRANSPORT_N_TRIALS,
        pick_radius=TRANSPORT_PICK_RADIUS,
        drop_radius=TRANSPORT_DROP_RADIUS,
        timeout=TRANSPORT_TIMEOUT,
        min_reach=TRANSPORT_MIN_REACH,
        max_reach=TRANSPORT_MAX_REACH,
        min_elevation=TRANSPORT_MIN_ELEV,
        max_elevation=TRANSPORT_MAX_ELEV,
        az_min=TRANSPORT_AZ_MIN,
        az_max=TRANSPORT_AZ_MAX,
        seed=TRANSPORT_SEED,
        start_pos=start_pos,
    )


# ── persistence ───────────────────────────────────────────────────────────────


def save_results(experiment, mode: str):
    """Persist experiment results to a timestamped CSV."""
    results = experiment.results
    if not results:
        return
    ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"poseEstimation{mode.capitalize()}Results/{mode}_results_{ts}.csv"

    if mode == MODE_REACH:
        _save_reach(filename, experiment, results)
    else:
        _save_transport(filename, results)

    print(f"Results saved to {filename}")
    print(experiment.summary())


def _save_reach(filename, experiment, results):
    fieldnames = [
        "trial",
        "label",
        "result",
        "duration_s",
        "target_x",
        "target_y",
        "target_z",
    ]
    trial_defs = {i + 1: t for i, t in enumerate(experiment._trial_defs)}
    with open(filename, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for r in results:
            pos = trial_defs.get(r["trial"], {}).get("pos", [None, None, None])
            writer.writerow(
                {
                    "trial": r["trial"],
                    "label": r["label"],
                    "result": r["result"],
                    "duration_s": round(r["duration"], 3),
                    "target_x": round(pos[0], 4) if pos[0] is not None else "",
                    "target_y": round(pos[1], 4) if pos[1] is not None else "",
                    "target_z": round(pos[2], 4) if pos[2] is not None else "",
                }
            )


def _save_transport(filename, results):
    fieldnames = [
        "trial",
        "label",
        "result",
        "duration_s",
        "cube_x",
        "cube_y",
        "cube_z",
        "drop_x",
        "drop_y",
        "drop_z",
        "start_x",
        "start_y",
        "start_z",
        "dist_start_to_cube",
        "dist_start_to_drop",
        "phase_approach_s",
        "phase_grip_s",
        "phase_carry_s",
        "phase_place_s",
    ]
    with open(filename, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for r in results:
            cp = r.get("cube_pos", [None, None, None])
            dp = r.get("drop_pos", [None, None, None])
            spos = r.get("start_pos") or [None, None, None]
            sp = r.get("phase_splits", {})
            writer.writerow(
                {
                    "trial": r["trial"],
                    "label": r["label"],
                    "result": r["result"],
                    "duration_s": round(r["duration"], 3),
                    "cube_x": round(cp[0], 4) if cp[0] is not None else "",
                    "cube_y": round(cp[1], 4) if cp[1] is not None else "",
                    "cube_z": round(cp[2], 4) if cp[2] is not None else "",
                    "drop_x": round(dp[0], 4) if dp[0] is not None else "",
                    "drop_y": round(dp[1], 4) if dp[1] is not None else "",
                    "drop_z": round(dp[2], 4) if dp[2] is not None else "",
                    "start_x": round(spos[0], 4) if spos[0] is not None else "",
                    "start_y": round(spos[1], 4) if spos[1] is not None else "",
                    "start_z": round(spos[2], 4) if spos[2] is not None else "",
                    "dist_start_to_cube": (
                        round(r["dist_start_to_cube"], 4)
                        if r.get("dist_start_to_cube") is not None
                        else ""
                    ),
                    "dist_start_to_drop": (
                        round(r["dist_start_to_drop"], 4)
                        if r.get("dist_start_to_drop") is not None
                        else ""
                    ),
                    "phase_approach_s": round(sp.get("approach", 0.0), 3),
                    "phase_grip_s": round(sp.get("grip", 0.0), 3),
                    "phase_carry_s": round(sp.get("carry", 0.0), 3),
                    "phase_place_s": round(sp.get("place", 0.0), 3),
                }
            )
