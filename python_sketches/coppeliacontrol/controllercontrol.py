from math import pi, sqrt
import pygame
import numpy as np
from coppeliasim_zmqremoteapi_client import RemoteAPIClient

# ── constants ────────────────────────────────────────────────────────────────
ROBOT_ARM_LENGTH = 0.21492 + 0.24129

MOVE_SPEED = 0.005  # metres per sim step
ROT_SPEED = 0.02  # radians per sim step

# Signal name read by the Lua gripper script each actuation step.
# 1 = close, 0 = open
GRIPPER_SIGNAL = "gripper_close"

TRIGGER_THRESHOLD = 0.1
DEADZONE = 0.1

# Button index for mode toggle (RB on a standard gamepad)
MODE_BUTTON = 5
RESET_BUTTON = 1

# Colours for mode indicator
COL_POSITION = (100, 220, 100)  # green
COL_ROTATION = (220, 160, 60)  # amber


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
    """Convert Euler angles (radians) to quaternion [x, y, z, w]."""
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


# ── CoppeliaSim setup ─────────────────────────────────────────────────────────
print("Connecting to CoppeliaSim...")
client = RemoteAPIClient()
sim = client.require("sim")
simIK = client.require("simIK")

rightShoulderAbduct = sim.getObject("/rightJoint1")
rightWristLink = sim.getObject(
    "/rightJoint1/rightLink1/rightJoint2/rightLink2"
    "/rightJoint3/rightLink3/rightJoint4/rightLink4"
    "/rightJoint5/rightLink5/rightJoint6/rightLink6/rightJoint7/rightLink7"
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

# ── Pygame / joystick setup ───────────────────────────────────────────────────
pygame.init()
pygame.joystick.init()

if pygame.joystick.get_count() == 0:
    raise RuntimeError("No joystick detected. Please connect a joystick and retry.")

joystick = pygame.joystick.Joystick(0)
joystick.init()
print(f"Using joystick: {joystick.get_name()}")
print(
    "Controls:\n"
    "  ── POSITION mode (RB not held) ──\n"
    "  Left  stick X/Y  → end-effector Y / X  (left/right, forward/back)\n"
    "  Right stick Y    → end-effector Z       (up/down)\n"
    "\n"
    "  ── ROTATION mode (hold RB) ──\n"
    "  Left  stick X    → yaw\n"
    "  Left  stick Y    → pitch\n"
    "  Right stick Y    → roll\n"
    "\n"
    "  ── Always active ──\n"
    "  RT (axis 5)      → gripper open\n"
    "  LT (axis 4)      → gripper close  (wins if both pressed)\n"
    "  Q / Start btn    → quit\n"
)

screen = pygame.display.set_mode((460, 260))
pygame.display.set_caption("YuMi 7-DOF Joystick Control")
font = pygame.font.SysFont("monospace", 16)
font_large = pygame.font.SysFont("monospace", 18, bold=True)

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

        # ── read axes & mode button ───────────────────────────────────────────
        # Axes:
        #   0  left  stick X (left/right)
        #   1  left  stick Y (forward/back)
        #   2  right stick X (unused in both modes)
        #   3  right stick Y (Z in position mode, roll in rotation mode)
        #   4  LT             (gripper close, -1 rest → +1 full)
        #   5  RT             (gripper open,  -1 rest → +1 full)
        ls_x = apply_deadzone(joystick.get_axis(0), DEADZONE)
        ls_y = apply_deadzone(joystick.get_axis(1), DEADZONE)
        rs_y = apply_deadzone(joystick.get_axis(3), DEADZONE)
        lt = (joystick.get_axis(4) + 1.0) / 2.0
        rt = (joystick.get_axis(5) + 1.0) / 2.0

        rotation_mode = joystick.get_button(MODE_BUTTON)
        reset_rot = joystick.get_button(RESET_BUTTON)

        # ── gripper — always active ───────────────────────────────────────────
        if lt > TRIGGER_THRESHOLD:
            sim.setInt32Signal(GRIPPER_SIGNAL, 1)
            gripper_open = False
        elif rt > TRIGGER_THRESHOLD:
            sim.setInt32Signal(GRIPPER_SIGNAL, 0)
            gripper_open = True

        # ── position mode ─────────────────────────────────────────────────────
        if not rotation_mode:
            dx = ls_y * MOVE_SPEED  # forward / back  (CoppeliaSim X)
            dy = ls_x * MOVE_SPEED  # left   / right  (CoppeliaSim Y)
            dz = -rs_y * MOVE_SPEED  # up     / down   (CoppeliaSim Z)

            target_pos = vec_add(target_pos, [dx, dy, dz])
            target_pos = vec_clamp_to_sphere(
                robot_shoulder_world, target_pos, ROBOT_ARM_LENGTH
            )

        # ── rotation mode ─────────────────────────────────────────────────────
        else:
            d_pitch = ls_y * ROT_SPEED  # left stick Y → pitch
            d_yaw = ls_x * ROT_SPEED  # left stick X → yaw
            d_roll = rs_y * ROT_SPEED  # right stick Y → roll

            if d_pitch or d_yaw or d_roll:
                delta = euler_to_quaternion(d_roll, d_pitch, d_yaw)
                target_quat = normalise_quaternion(
                    quaternion_multiply(target_quat, delta)
                )
            elif reset_rot:
                target_quat = [0, 0, 0, 1]

        # ── push to sim ───────────────────────────────────────────────────────
        sim.setObjectPosition(target, target_pos)
        sim.setObjectQuaternion(target, target_quat)

        # Always use the full pose-constrained damped group so orientation is
        # actually enforced by the IK solver in both modes.
        res, *_ = simIK.handleGroup(ikEnv, ikGroupUndamped, {"syncWorlds": True})
        if res != simIK.result_success:
            simIK.handleGroup(ikEnv, ikGroupDamped, {"syncWorlds": True})

        sim.step()

        # ── display ───────────────────────────────────────────────────────────
        screen.fill((30, 30, 30))

        mode_label = (
            "[ ROTATION MODE — hold RB ]"
            if rotation_mode
            else "[ POSITION MODE — hold RB to rotate ]"
        )
        mode_col = COL_ROTATION if rotation_mode else COL_POSITION
        screen.blit(font_large.render(mode_label, True, mode_col), (10, 8))

        gripper_label = "OPEN" if gripper_open else "CLOSED"
        lines = [
            f"Target  X: {target_pos[0]:.4f}  Y: {target_pos[1]:.4f}  Z: {target_pos[2]:.4f}",
            f"Quat    x: {target_quat[0]:.3f}  y: {target_quat[1]:.3f}",
            f"  z: {target_quat[2]:.3f}  w: {target_quat[3]:.3f}",
            f"Reset rot: {'Active' if reset_rot else 'Inactive'}",
            f"L-stick ({ls_x:+.2f}, {ls_y:+.2f})   R-stick Y {rs_y:+.2f}",
            f"Gripper: {gripper_label}  (LT={lt:.2f}  RT={rt:.2f})",
            "Q / Start button to quit",
        ]
        for i, line in enumerate(lines):
            screen.blit(font.render(line, True, (200, 200, 200)), (10, 38 + i * 22))

        pygame.display.flip()

except KeyboardInterrupt:
    print("Interrupted.")

finally:
    sim.clearInt32Signal(GRIPPER_SIGNAL)
    sim.stopSimulation()
    pygame.quit()
    print("Simulation stopped.")
