"""
hand_gesture.py
===============
Finger-curl detection and the per-finger curl-meter HUD widget.
"""

import cv2

from .config import FINGER_CURL_THRESHOLD, FINGER_CLOSED_COUNT

# MediaPipe hand landmark indices
_FINGER_NAMES = ["Index", "Middle", "Ring", "Pinky"]
_MCP_IDS = [5, 9, 13, 17]
_TIP_IDS = [8, 12, 16, 20]
_WRIST_ID = 0


# ── curl ratios ───────────────────────────────────────────────────────────────


def compute_finger_curls(hand_landmarks):
    """
    Return a list of 4 curl ratios (tip_dist / mcp_dist) for Index → Pinky.

    ratio < FINGER_CURL_THRESHOLD  →  curled / closed
    ratio >= FINGER_CURL_THRESHOLD →  extended / open

    Returns None if landmarks are invalid.
    """
    lm = hand_landmarks.landmark

    def dist3(a, b):
        return (
            (lm[a].x - lm[b].x) ** 2
            + (lm[a].y - lm[b].y) ** 2
            + (lm[a].z - lm[b].z) ** 2
        ) ** 0.5

    ratios = []
    for mcp, tip in zip(_MCP_IDS, _TIP_IDS):
        d_mcp = dist3(_WRIST_ID, mcp)
        ratios.append(1.0 if d_mcp < 1e-6 else min(2.0, dist3(_WRIST_ID, tip) / d_mcp))
    return ratios


def classify_hand_open(hand_landmarks) -> bool:
    """True = open hand, False = closed fist."""
    ratios = compute_finger_curls(hand_landmarks)
    curled = sum(1 for r in ratios if r < FINGER_CURL_THRESHOLD)
    return curled < FINGER_CLOSED_COUNT


# ── HUD widget ────────────────────────────────────────────────────────────────


def draw_curl_meter(frame, curl_ratios, origin_xy, label_prefix=""):
    """
    Draw a compact per-finger curl meter at origin_xy (top-left of panel).

    Each finger gets a horizontal bar:
      - Full bar width = ratio of 2.0 (fully extended beyond MCP)
      - Threshold line marks FINGER_CURL_THRESHOLD
      - Bar colour: green (extended) → red (curled)
    """
    if curl_ratios is None:
        return

    ox, oy = origin_xy
    BAR_W = 110  # max bar width in pixels
    BAR_H = 9
    ROW_STEP = 17
    MAX_RATIO = 2.0

    panel_h = ROW_STEP * 4 + 10
    cv2.rectangle(
        frame, (ox - 4, oy - 4), (ox + BAR_W + 74, oy + panel_h), (22, 22, 28), -1
    )
    cv2.rectangle(
        frame, (ox - 4, oy - 4), (ox + BAR_W + 74, oy + panel_h), (55, 55, 65), 1
    )

    if label_prefix:
        cv2.putText(
            frame,
            label_prefix,
            (ox, oy - 8),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.35,
            (110, 110, 110),
            1,
        )

    thresh_x = ox + int(BAR_W * FINGER_CURL_THRESHOLD / MAX_RATIO)

    for i, (name, ratio) in enumerate(zip(_FINGER_NAMES, curl_ratios)):
        by = oy + i * ROW_STEP

        cv2.rectangle(frame, (ox, by), (ox + BAR_W, by + BAR_H), (45, 45, 45), -1)

        fill_w = max(1, int(BAR_W * min(ratio, MAX_RATIO) / MAX_RATIO))
        t = max(
            0.0,
            min(
                1.0,
                1.0
                - (ratio - FINGER_CURL_THRESHOLD)
                / max(1e-6, 1.0 - FINGER_CURL_THRESHOLD),
            ),
        )
        r_ch = int(40 + t * (220 - 40))
        g_ch = int(200 - t * (200 - 60))
        cv2.rectangle(frame, (ox, by), (ox + fill_w, by + BAR_H), (40, g_ch, r_ch), -1)

        cv2.line(
            frame, (thresh_x, by - 1), (thresh_x, by + BAR_H + 1), (200, 200, 200), 1
        )

        is_curled = ratio < FINGER_CURL_THRESHOLD
        lbl_col = (60, 60, 220) if is_curled else (60, 200, 100)
        cv2.putText(
            frame,
            f"{name[0]}  {ratio:.2f}{'  curl' if is_curled else ''}",
            (ox + BAR_W + 4, by + BAR_H - 1),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.32,
            lbl_col,
            1,
        )
