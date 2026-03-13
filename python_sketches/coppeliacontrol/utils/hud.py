"""
hud.py
======
All OpenCV overlay / HUD drawing functions.

  draw_mode_select_hud()  — lobby menu
  draw_experiment_hud()   — unified HUD for Reach and Transport experiments
"""

import cv2

from .config import MODE_REACH, MODE_TRANSPORT, MODE_OBSTACLE


# ── colour helper ─────────────────────────────────────────────────────────────


def _cv_col(r, g, b):
    """Convert RGB tuple to OpenCV BGR tuple."""
    return (b, g, r)


# ── mode-select screen ────────────────────────────────────────────────────────


def draw_mode_select_hud(frame):
    """Overlay a mode-selection menu on the HUD camera frame."""
    H, W = frame.shape[:2]
    overlay = frame.copy()
    cv2.rectangle(overlay, (0, 0), (W, H), (10, 10, 18), -1)
    cv2.addWeighted(overlay, 0.55, frame, 0.45, 0, frame)

    lines = [
        ("YuMi Pose Control", 0.9, (210, 210, 210), 2),
        ("Select experiment mode:", 0.65, (160, 160, 160), 1),
        ("", 0.0, (0, 0, 0), 0),
        ("[R]  Reach Experiment", 0.7, _cv_col(100, 220, 130), 2),
        ("[T]  Transport Experiment", 0.7, _cv_col(100, 180, 255), 2),
        ("[O]  Obstacle Transport", 0.7, _cv_col(220, 100, 80), 2),
        ("", 0.0, (0, 0, 0), 0),
        ("Transport: open hand = gripper open", 0.45, (140, 140, 140), 1),
        ("          closed fist = gripper closed", 0.45, (140, 140, 140), 1),
        ("", 0.0, (0, 0, 0), 0),
        ("[Q]  Quit", 0.5, (130, 130, 130), 1),
    ]
    y = H // 2 - 120
    for text, scale, col, thick in lines:
        if not text:
            y += 14
            continue
        (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, scale, thick)
        cv2.putText(
            frame,
            text,
            (W // 2 - tw // 2, y),
            cv2.FONT_HERSHEY_SIMPLEX,
            scale,
            col,
            thick,
        )
        y += th + 16


# ── experiment HUD ────────────────────────────────────────────────────────────


def draw_experiment_hud(frame, experiment, wrist_pos, dt, mode):
    """Unified HUD for both Reach and Transport experiments."""
    H, W = frame.shape[:2]

    # Mode badge (top-left)
    if mode == MODE_REACH:
        badge_txt = "REACH"
        badge_col = _cv_col(100, 220, 130)
    elif mode == MODE_OBSTACLE:
        badge_txt = "OBSTACLE"
        badge_col = _cv_col(220, 100, 80)
    else:
        badge_txt = "TRANSPORT"
        badge_col = _cv_col(100, 180, 255)

    cv2.rectangle(frame, (0, 0), (130, 28), (22, 22, 28), -1)
    cv2.putText(frame, badge_txt, (6, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.55, badge_col, 1)

    # Switch hint (bottom-right)
    hint = "[R] Reach  [T] Transport  [Q] Quit"
    cv2.putText(
        frame, hint, (W - 330, H - 22), cv2.FONT_HERSHEY_SIMPLEX, 0.38, (80, 80, 80), 1
    )

    total = len(experiment._trial_defs)
    done = len(experiment.results)

    # Progress bar (bottom)
    bar_y = H - 12
    cv2.rectangle(frame, (10, bar_y), (W - 10, bar_y + 8), (45, 45, 55), -1)
    if total and done:
        fill_x = 10 + int((W - 20) * done / total)
        cv2.rectangle(
            frame, (10, bar_y), (fill_x, bar_y + 8), _cv_col(100, 220, 130), -1
        )
    cv2.putText(
        frame,
        f"Trial {min(done+1, total)} / {total}",
        (10, bar_y - 4),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.45,
        (130, 130, 130),
        1,
    )

    # Finished overlay
    if experiment.finished:
        overlay = frame.copy()
        cv2.rectangle(overlay, (0, 0), (W, H), (18, 10, 10), -1)
        cv2.addWeighted(overlay, 0.6, frame, 0.4, 0, frame)
        n_ok = sum(1 for r in experiment.results if r["result"] == "success")
        cv2.putText(
            frame,
            "EXPERIMENT COMPLETE",
            (W // 2 - 160, H // 2 - 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.9,
            _cv_col(100, 220, 130),
            2,
        )
        cv2.putText(
            frame,
            f"{n_ok} / {total}  trials succeeded",
            (W // 2 - 120, H // 2 + 10),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.65,
            (210, 210, 210),
            1,
        )
        cv2.putText(
            frame,
            "Press [R] or [T] to restart",
            (W // 2 - 140, H // 2 + 40),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (130, 130, 130),
            1,
        )
        return

    active = experiment._active
    if active is None:
        return

    if mode == MODE_REACH:
        _draw_reach_hud(frame, active, W)
    else:
        _draw_transport_hud(frame, active, wrist_pos, W)


# ── reach sub-HUD ─────────────────────────────────────────────────────────────


def _draw_reach_hud(frame, active, W):
    dist = active.distance_to(active._pos if hasattr(active, "_pos") else [0, 0, 0])
    # recalculate from the caller — active.distance_to needs wrist_pos, keep signature clean
    # (wrist_pos is passed from draw_experiment_hud via active)
    dist = active.distance_to(getattr(active, "_last_wrist", [0, 0, 0]))
    ratio = min(1.0, dist / (active.radius * 6))

    def lerp(a, b, t):
        t = max(0.0, min(1.0, t))
        return tuple(int(a[i] + (b[i] - a[i]) * t) for i in range(3))

    C_HOT = (100, 220, 130)
    C_WARM = (220, 190, 60)
    C_IDLE = (100, 160, 220)
    hud_rgb = (
        lerp(C_HOT, C_WARM, ratio / 0.4)
        if ratio < 0.4
        else lerp(C_WARM, C_IDLE, (ratio - 0.4) / 0.6)
    )
    hud_col = _cv_col(*hud_rgb)

    if active._flash_t > 0:
        _draw_flash(frame, active, W)
        return

    px, py = W - 270, 30
    cv2.rectangle(frame, (px - 8, py - 4), (W - 8, py + 115), (22, 22, 28), -1)
    cv2.rectangle(frame, (px - 8, py - 4), (W - 8, py + 115), hud_col, 1)
    cv2.putText(
        frame,
        active.label,
        (px, py + 18),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.55,
        (210, 210, 210),
        1,
    )
    cv2.putText(
        frame,
        f"dist  {dist*100:.1f} cm",
        (px, py + 38),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.5,
        hud_col,
        1,
    )

    bar_x, bar_y2 = px, py + 52
    bar_w = 250
    fill = int(bar_w * (1.0 - ratio))
    cv2.rectangle(frame, (bar_x, bar_y2), (bar_x + bar_w, bar_y2 + 8), (55, 55, 55), -1)
    if fill > 0:
        cv2.rectangle(frame, (bar_x, bar_y2), (bar_x + fill, bar_y2 + 8), hud_col, -1)

    if active._inside and active.dwell_fraction > 0:
        cx, cy = W - 35, py + 88
        r = 18
        cv2.circle(frame, (cx, cy), r, (55, 55, 55), 2)
        cv2.ellipse(
            frame,
            (cx, cy),
            (r, r),
            -90,
            0,
            int(360 * active.dwell_fraction),
            _cv_col(100, 220, 130),
            2,
        )
        cv2.putText(
            frame,
            f"{int(active.dwell_fraction*100)}%",
            (cx - 14, cy + 5),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.35,
            _cv_col(100, 220, 130),
            1,
        )

    if active.timeout > 0:
        if not active._started:
            cv2.putText(
                frame,
                "move to start timer",
                (px, py + 108),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.4,
                _cv_col(160, 140, 60),
                1,
            )
        else:
            tr = active.time_remaining
            t_col = _cv_col(220, 80, 80) if tr < 3.0 else (130, 130, 130)
            cv2.putText(
                frame,
                f"time  {tr:.1f}s",
                (px, py + 108),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.45,
                t_col,
                1,
            )


def _draw_flash(frame, active, W):
    H = frame.shape[0]
    flash_rgb = (100, 220, 130) if active._result == "success" else (220, 80, 80)
    alpha = active._flash_t / 0.6
    overlay = frame.copy()
    cv2.rectangle(overlay, (0, 0), (W, H), _cv_col(*flash_rgb), -1)
    cv2.addWeighted(overlay, alpha * 0.45, frame, 1 - alpha * 0.45, 0, frame)
    msg = "TARGET REACHED" if active._result == "success" else "TIMED OUT"
    cv2.putText(
        frame,
        msg,
        (W // 2 - 110, H // 2),
        cv2.FONT_HERSHEY_SIMPLEX,
        1.1,
        (255, 255, 255),
        2,
    )


# ── transport sub-HUD ─────────────────────────────────────────────────────────


def _draw_transport_hud(frame, active, wrist_pos, W):
    import math

    PHASE_LABELS = {
        "approach": "1. Move wrist to cube  (open hand)",
        "grip": "2. Close fist to grip",
        "carry": "3. Carry cube to drop zone",
        "place": "4. Open hand to release",
        "done": "Done!",
    }

    phase = getattr(active, "_phase", "approach")
    phase_text = PHASE_LABELS.get(phase, phase)

    C_ORANGE = _cv_col(230, 130, 40)
    C_HOT = _cv_col(100, 220, 130)
    phase_col = C_ORANGE if phase in ("approach", "grip") else C_HOT

    if active._flash_t > 0 and active._result:
        H = frame.shape[0]
        flash_rgb = (100, 220, 130) if active._result == "success" else (220, 80, 80)
        alpha = active._flash_t / 0.8
        overlay = frame.copy()
        cv2.rectangle(overlay, (0, 0), (W, H), _cv_col(*flash_rgb), -1)
        cv2.addWeighted(overlay, alpha * 0.45, frame, 1 - alpha * 0.45, 0, frame)
        msg = "DELIVERED!" if active._result == "success" else "TIMED OUT"
        cv2.putText(
            frame,
            msg,
            (W // 2 - 100, H // 2),
            cv2.FONT_HERSHEY_SIMPLEX,
            1.1,
            (255, 255, 255),
            2,
        )
        return

    px, py = W - 290, 30
    panel_h = 140
    cv2.rectangle(frame, (px - 8, py - 4), (W - 8, py + panel_h), (22, 22, 28), -1)
    cv2.rectangle(frame, (px - 8, py - 4), (W - 8, py + panel_h), phase_col, 1)

    cv2.putText(
        frame,
        active.label,
        (px, py + 18),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.55,
        (210, 210, 210),
        1,
    )
    cv2.putText(
        frame, phase_text, (px, py + 38), cv2.FONT_HERSHEY_SIMPLEX, 0.46, phase_col, 1
    )

    cur_cube = getattr(active, "_current_cube_pos", active.cube_pos)
    d_cube = math.sqrt(sum((wrist_pos[i] - cur_cube[i]) ** 2 for i in range(3)))
    d_drop = math.sqrt(sum((wrist_pos[i] - active.drop_pos[i]) ** 2 for i in range(3)))

    if phase in ("approach", "grip"):
        d_txt = f"cube  {d_cube*100:.1f} cm"
        d_col = C_HOT if d_cube <= active.pick_radius else C_ORANGE
    else:
        d_txt = f"drop  {d_drop*100:.1f} cm"
        d_col = C_HOT if d_drop <= active.drop_radius else _cv_col(220, 190, 60)
    cv2.putText(frame, d_txt, (px, py + 58), cv2.FONT_HERSHEY_SIMPLEX, 0.5, d_col, 1)

    gripped = getattr(active, "_gripped", False)
    g_txt = (
        "hand: CLOSED  (open to release)"
        if gripped
        else "hand: OPEN  (close fist to grip)"
    )
    g_col = C_HOT if gripped else _cv_col(100, 160, 220)
    cv2.putText(frame, g_txt, (px, py + 78), cv2.FONT_HERSHEY_SIMPLEX, 0.45, g_col, 1)

    # Phase dots
    phases = ["approach", "grip", "carry", "place"]
    phase_idx = phases.index(phase) if phase in phases else 0
    dot_y = py + 108
    for i, ph in enumerate(phases):
        dot_x = px + i * 62 + 15
        done_ph = phase_idx > i
        active_ph = phase_idx == i
        col = (
            _cv_col(100, 220, 130)
            if done_ph
            else (phase_col if active_ph else (55, 55, 55))
        )
        cv2.circle(frame, (dot_x, dot_y), 7, col, -1)
        cv2.putText(
            frame,
            str(i + 1),
            (dot_x - 4, dot_y + 4),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.35,
            (20, 20, 20) if (done_ph or active_ph) else (100, 100, 100),
            1,
        )
        if i < len(phases) - 1:
            line_col = _cv_col(100, 220, 130) if done_ph else (55, 55, 55)
            cv2.line(frame, (dot_x + 7, dot_y), (dot_x + 55, dot_y), line_col, 2)

    tr = active.time_remaining
    t_col = _cv_col(220, 80, 80) if tr < 5.0 else (130, 130, 130)
    cv2.putText(
        frame,
        f"time  {tr:.1f}s",
        (px, py + 128),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.45,
        t_col,
        1,
    )
