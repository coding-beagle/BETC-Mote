from math import pi, sqrt
import pygame
import numpy as np
from coppeliasim_zmqremoteapi_client import RemoteAPIClient

# ── constants ────────────────────────────────────────────────────────────────
ROBOT_ARM_LENGTH = 0.21492 + 0.24129

MOVE_SPEED = 0.005  # metres per sim step
ROT_SPEED = 0.02  # radians per sim step

GRIPPER_SIGNAL = "gripper_close"
TRIGGER_THRESHOLD = 0.1
DEADZONE = 0.1
MODE_BUTTON = 5  # RB
RESET_BUTTON = 1  # B

COL_POSITION = (100, 220, 100)
COL_ROTATION = (220, 160, 60)

# ── display layout ────────────────────────────────────────────────────────────
W, H = 700, 420
STICK_R = 50  # radius of stick gate circle
STICK_DOT_R = 8  # radius of stick position dot
TRIG_W = 30
TRIG_H = 80

# Stick centre positions  (x, y)
LS_CX, LS_CY = 140, 260
RS_CX, RS_CY = 370, 260

# Trigger positions  (top-left x, y)
LT_X, LT_Y = 55, 60
RT_X, RT_Y = 610, 60

# Button positions  {index: (cx, cy, label)}
BUTTON_LAYOUT = {
    0: (580, 200, "A"),
    1: (605, 175, "B"),
    2: (555, 175, "X"),
    3: (580, 150, "Y"),
    4: (120, 110, "LB"),
    5: (580, 110, "RB"),
}


# ── helpers ───────────────────────────────────────────────────────────────────
def vec_add(a, b):
    return [a[i] + b[i] for i in range(3)]


def vec_clamp_to_sphere(origin, point, radius):
    diff = [point[i] - origin[i] for i in range(3)]
    length = sum(x**2 for x in diff) ** 0.5
    if length > radius:
        scale = radius / length
        diff = [d * scale for d in diff]
    return [origin[i] + diff[i] for i in range(3)]


def apply_deadzone(value, deadzone):
    if abs(value) < deadzone:
        return 0.0
    sign = 1 if value > 0 else -1
    return sign * (abs(value) - deadzone) / (1.0 - deadzone)


def euler_to_quaternion(roll, pitch, yaw):
    cr, sr = np.cos(roll / 2), np.sin(roll / 2)
    cp, sp = np.cos(pitch / 2), np.sin(pitch / 2)
    cy, sy = np.cos(yaw / 2), np.sin(yaw / 2)
    return [
        sr * cp * cy - cr * sp * sy,
        cr * sp * cy + sr * cp * sy,
        cr * cp * sy - sr * sp * cy,
        cr * cp * cy + sr * sp * sy,
    ]


def quaternion_multiply(q1, q2):
    x1, y1, z1, w1 = q1
    x2, y2, z2, w2 = q2
    return [
        w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
        w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
        w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2,
        w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2,
    ]


def normalise_quaternion(q):
    n = sum(v**2 for v in q) ** 0.5
    return [v / n for v in q] if n > 1e-9 else q


# ── drawing helpers ───────────────────────────────────────────────────────────
def draw_stick(surf, cx, cy, ax, ay, label, active_col):
    """Draw a stick gate, dot, and cross-hair."""
    # Gate
    pygame.draw.circle(surf, (70, 70, 70), (cx, cy), STICK_R, 2)
    # Cross-hair
    pygame.draw.line(surf, (50, 50, 50), (cx - STICK_R, cy), (cx + STICK_R, cy), 1)
    pygame.draw.line(surf, (50, 50, 50), (cx, cy - STICK_R), (cx, cy + STICK_R), 1)
    # Dot
    dot_x = int(cx + ax * STICK_R)
    dot_y = int(cy + ay * STICK_R)
    pygame.draw.circle(surf, active_col, (dot_x, dot_y), STICK_DOT_R)
    pygame.draw.circle(surf, (255, 255, 255), (dot_x, dot_y), STICK_DOT_R, 1)
    # Label
    lbl = pygame.font.SysFont("monospace", 13).render(label, True, (160, 160, 160))
    surf.blit(lbl, (cx - lbl.get_width() // 2, cy + STICK_R + 6))


def draw_trigger(surf, tx, ty, value, label, active_col):
    """Draw a vertical trigger bar."""
    pygame.draw.rect(surf, (60, 60, 60), (tx, ty, TRIG_W, TRIG_H), 2)
    fill_h = int(TRIG_H * value)
    if fill_h > 0:
        pygame.draw.rect(
            surf, active_col, (tx + 2, ty + TRIG_H - fill_h, TRIG_W - 4, fill_h)
        )
    lbl = pygame.font.SysFont("monospace", 13).render(label, True, (160, 160, 160))
    surf.blit(lbl, (tx + TRIG_W // 2 - lbl.get_width() // 2, ty + TRIG_H + 4))


def draw_button(surf, cx, cy, label, pressed):
    col_fill = (220, 180, 50) if pressed else (55, 55, 55)
    col_border = (255, 220, 100) if pressed else (110, 110, 110)
    col_text = (20, 20, 20) if pressed else (160, 160, 160)
    pygame.draw.circle(surf, col_fill, (cx, cy), 16)
    pygame.draw.circle(surf, col_border, (cx, cy), 16, 2)
    lbl = pygame.font.SysFont("monospace", 11, bold=pressed).render(
        label, True, col_text
    )
    surf.blit(lbl, (cx - lbl.get_width() // 2, cy - lbl.get_height() // 2))


def draw_gamepad_body(surf):
    """Draw a simple controller silhouette."""
    pygame.draw.ellipse(surf, (45, 45, 45), (60, 140, 580, 240))
    # Grip bumps
    pygame.draw.ellipse(surf, (45, 45, 45), (50, 310, 160, 100))
    pygame.draw.ellipse(surf, (45, 45, 45), (490, 310, 160, 100))


# ── CoppeliaSim setup ─────────────────────────────────────────────────────────
print("Connecting to CoppeliaSim...")
client = RemoteAPIClient()
sim = client.require("sim")
simIK = client.require("simIK")

rightShoulderAbduct = sim.getObject("/rightJoint1")
rightWristLink = sim.getObject(
    "/rightJoint1/rightLink1/rightJoint2/rightLink2"
    "/rightJoint3/rightLink3/rightJoint4/rightLink4"
    "/rightJoint5/rightLink5/rightJoint6/rightLink6"
    "/rightJoint7/rightLink7"
)

target = sim.createDummy(0.02)
sim.setObjectAlias(target, "WristTarget")

robot_shoulder_world = sim.getObjectPosition(rightShoulderAbduct, -1)
print(f"Robot shoulder origin: {robot_shoulder_world}")

target_pos = [
    robot_shoulder_world[0] + 0.3,
    robot_shoulder_world[1],
    robot_shoulder_world[2],
]
target_quat = [0.0, 0.0, 0.0, 1.0]

# ── IK setup ──────────────────────────────────────────────────────────────────
ikEnv = simIK.createEnvironment()

ikGroupUndamped = simIK.createGroup(ikEnv)
simIK.setGroupCalculation(ikEnv, ikGroupUndamped, simIK.method_pseudo_inverse, 0, 6)
simIK.addElementFromScene(
    ikEnv,
    ikGroupUndamped,
    rightShoulderAbduct,
    rightWristLink,
    target,
    simIK.constraint_pose,
)

ikGroupDamped = simIK.createGroup(ikEnv)
simIK.setGroupCalculation(
    ikEnv, ikGroupDamped, simIK.method_damped_least_squares, 1, 99
)
simIK.addElementFromScene(
    ikEnv,
    ikGroupDamped,
    rightShoulderAbduct,
    rightWristLink,
    target,
    simIK.constraint_position,
)

# ── Pygame setup ──────────────────────────────────────────────────────────────
pygame.init()
pygame.joystick.init()

if pygame.joystick.get_count() == 0:
    raise RuntimeError("No joystick detected. Please connect a joystick and retry.")

joystick = pygame.joystick.Joystick(0)
joystick.init()
print(f"Using joystick: {joystick.get_name()}")

screen = pygame.display.set_mode((W, H))
pygame.display.set_caption("YuMi 7-DOF Joystick Control")
font_sm = pygame.font.SysFont("monospace", 13)
font_md = pygame.font.SysFont("monospace", 15)
font_lg = pygame.font.SysFont("monospace", 17, bold=True)
clock = pygame.time.Clock()

try:
    print("Starting simulation...")
    sim.setStepping(True)
    sim.startSimulation()
    sim.setInt32Signal(GRIPPER_SIGNAL, 1)
    print("Simulation started OK")

    running = True
    gripper_open = False

    while running:
        # ── events ────────────────────────────────────────────────────────────
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            elif event.type == pygame.KEYDOWN and event.key == pygame.K_q:
                running = False
            elif event.type == pygame.JOYBUTTONDOWN and event.button == 7:
                running = False

        # ── read inputs ───────────────────────────────────────────────────────
        ls_x_raw = joystick.get_axis(0)
        ls_y_raw = joystick.get_axis(1)
        rs_x_raw = joystick.get_axis(2)
        rs_y_raw = joystick.get_axis(3)
        lt_raw = joystick.get_axis(4)
        rt_raw = joystick.get_axis(5)

        ls_x = apply_deadzone(ls_x_raw, DEADZONE)
        ls_y = apply_deadzone(ls_y_raw, DEADZONE)
        rs_y = apply_deadzone(rs_y_raw, DEADZONE)
        lt = (lt_raw + 1.0) / 2.0
        rt = (rt_raw + 1.0) / 2.0

        rotation_mode = joystick.get_button(MODE_BUTTON)
        reset_rot = joystick.get_button(RESET_BUTTON)

        num_buttons = joystick.get_numbuttons()
        button_states = [joystick.get_button(i) for i in range(num_buttons)]

        # ── gripper ───────────────────────────────────────────────────────────
        if lt > TRIGGER_THRESHOLD:
            sim.setInt32Signal(GRIPPER_SIGNAL, 1)
            gripper_open = False
        elif rt > TRIGGER_THRESHOLD:
            sim.setInt32Signal(GRIPPER_SIGNAL, 0)
            gripper_open = True

        # ── position mode ─────────────────────────────────────────────────────
        if not rotation_mode:
            target_pos = vec_add(
                target_pos,
                [
                    ls_y * MOVE_SPEED,
                    ls_x * MOVE_SPEED,
                    -rs_y * MOVE_SPEED,
                ],
            )
            target_pos = vec_clamp_to_sphere(
                robot_shoulder_world, target_pos, ROBOT_ARM_LENGTH
            )

        # ── rotation mode ─────────────────────────────────────────────────────
        else:
            d_pitch = ls_y * ROT_SPEED
            d_yaw = ls_x * ROT_SPEED
            d_roll = rs_y * ROT_SPEED

            if d_pitch or d_yaw or d_roll:
                delta = euler_to_quaternion(d_roll, d_pitch, d_yaw)
                target_quat = normalise_quaternion(
                    quaternion_multiply(target_quat, delta)
                )

            if reset_rot:
                target_quat = [0.0, 0.0, 0.0, 1.0]

        # ── push to sim ───────────────────────────────────────────────────────
        sim.setObjectPosition(target, target_pos)
        sim.setObjectQuaternion(target, target_quat)

        res, *_ = simIK.handleGroup(ikEnv, ikGroupUndamped, {"syncWorlds": True})
        if res != simIK.result_success:
            simIK.handleGroup(ikEnv, ikGroupDamped, {"syncWorlds": True})

        sim.step()

        # ── draw ──────────────────────────────────────────────────────────────
        screen.fill((22, 22, 28))

        # Controller body
        # draw_gamepad_body(screen)

        # Mode banner
        mode_label = "[ ROTATION MODE ]" if rotation_mode else "[ POSITION MODE ]"
        mode_col = COL_ROTATION if rotation_mode else COL_POSITION
        hint_label = (
            "holding RB"
            if rotation_mode
            else "hold RB to rotate  |  B to reset orientation"
        )
        screen.blit(font_lg.render(mode_label, True, mode_col), (10, 8))
        screen.blit(font_sm.render(hint_label, True, (130, 130, 130)), (10, 32))

        # Sticks — colour by mode
        ls_col = COL_ROTATION if rotation_mode else COL_POSITION
        rs_col = COL_ROTATION if rotation_mode else COL_POSITION

        ls_label = "pitch / yaw" if rotation_mode else "X / Y pos"
        rs_label = "roll" if rotation_mode else "Z pos"

        draw_stick(screen, LS_CX, LS_CY, ls_x_raw, ls_y_raw, ls_label, ls_col)
        draw_stick(screen, RS_CX, RS_CY, rs_x_raw, rs_y_raw, rs_label, rs_col)

        # Triggers
        lt_col = (220, 100, 100) if lt > TRIGGER_THRESHOLD else (100, 160, 220)
        rt_col = (100, 220, 130) if rt > TRIGGER_THRESHOLD else (100, 160, 220)
        draw_trigger(screen, LT_X, LT_Y, lt, "LT close", lt_col)
        draw_trigger(screen, RT_X, RT_Y, rt, "RT open", rt_col)

        # Buttons
        for idx, (bx, by, blabel) in BUTTON_LAYOUT.items():
            pressed = button_states[idx] if idx < num_buttons else False
            draw_button(screen, bx, by, blabel, pressed)

        # Telemetry panel (right side)
        px, py = 490, 150
        gripper_label = "OPEN" if gripper_open else "CLOSED"
        # tele_lines = [
        #     ("TARGET", None),
        #     (f" X {target_pos[0]:+.3f}", None),
        #     (f" Y {target_pos[1]:+.3f}", None),
        #     (f" Z {target_pos[2]:+.3f}", None),
        #     ("QUAT", None),
        #     (f" x {target_quat[0]:+.3f}  y {target_quat[1]:+.3f}", None),
        #     (f" z {target_quat[2]:+.3f}  w {target_quat[3]:+.3f}", None),
        #     (
        #         f"GRIP {gripper_label}",
        #         (100, 220, 130) if gripper_open else (220, 100, 100),
        #     ),
        # ]
        # for i, (txt, col) in enumerate(tele_lines):
        #     c = col if col else (190, 190, 190)
        #     screen.blit(font_sm.render(txt, True, c), (px, py + i * 18))

        pygame.display.flip()
        clock.tick(60)

except KeyboardInterrupt:
    print("Interrupted.")

finally:
    sim.clearInt32Signal(GRIPPER_SIGNAL)
    sim.stopSimulation()
    pygame.quit()
    print("Simulation stopped.")
