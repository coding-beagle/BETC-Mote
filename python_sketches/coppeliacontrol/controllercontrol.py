from math import pi, sqrt
import pygame
import numpy as np
from coppeliasim_zmqremoteapi_client import RemoteAPIClient

# ── constants ────────────────────────────────────────────────────────────────
ROBOT_ARM_LENGTH = 0.21492 + 0.24129

MOVE_SPEED = 0.005  # metres per sim step
ROT_SPEED = 0.02  # radians per sim step

# Signal name read by the Lua gripper script each actuation step.
# 1 = close, 0 = open  (matches the `close` boolean in the Lua script)
GRIPPER_SIGNAL = "gripper_close"

# Trigger threshold — how far a trigger must be pressed to count as active
TRIGGER_THRESHOLD = 0.1

DEADZONE = 0.1


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
    """Convert Euler angles (radians) to a quaternion [x, y, z, w]."""
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
    "/rightJoint5/rightLink5/rightJoint6/rightLink6"
)

# Create IK target dummy
target = sim.createDummy(0.02)
sim.setObjectAlias(target, "WristTarget")

robot_shoulder_world = sim.getObjectPosition(rightShoulderAbduct, -1)
print(f"Robot shoulder origin: {robot_shoulder_world}")

# Initial end-effector target state
target_pos = [
    robot_shoulder_world[0] + 0.3,
    robot_shoulder_world[1],
    robot_shoulder_world[2],
]
target_quat = [0.0, 0.0, 0.0, 1.0]  # identity

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
    simIK.constraint_position,
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
    simIK.constraint_pose,
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
    "  Left  stick X/Y  → end-effector X / Y  (forward/back, left/right)\n"
    "  Right stick X    → end-effector pitch   (axis 2)\n"
    "  Right stick Y    → end-effector Z       (up/down, axis 3)\n"
    "  RT (axis 5)      → gripper open\n"
    "  LT (axis 4)      → gripper close  (wins if both pressed)\n"
    "  Q / Start btn    → quit\n"
)

screen = pygame.display.set_mode((420, 240))
pygame.display.set_caption("YuMi 7-DOF Joystick Control")
font = pygame.font.SysFont("monospace", 16)

try:
    print("Starting simulation...")
    sim.setStepping(True)
    sim.startSimulation()

    # Initialise gripper to closed so the Lua script starts in a known state
    sim.setInt32Signal(GRIPPER_SIGNAL, 1)
    print("Simulation started OK")

    running = True
    gripper_open = False  # local state for display

    while running:
        # ── events ────────────────────────────────────────────────────────────
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            elif event.type == pygame.KEYDOWN and event.key == pygame.K_q:
                running = False
            elif event.type == pygame.JOYBUTTONDOWN and event.button == 7:
                running = False

        # ── read axes ─────────────────────────────────────────────────────────
        # Left  stick: axis 0 = X (left/right),  axis 1 = Y (forward/back)
        # Right stick: axis 2 = pitch,            axis 3 = Z (up/down)
        # Triggers:    axis 4 = LT (close),        axis 5 = RT (open)
        #              Triggers rest at -1, fully pressed = +1
        ls_x = apply_deadzone(joystick.get_axis(0), DEADZONE)
        ls_y = apply_deadzone(joystick.get_axis(1), DEADZONE)
        rs_pitch = apply_deadzone(joystick.get_axis(2), DEADZONE)
        rs_z = apply_deadzone(joystick.get_axis(3), DEADZONE)

        lt = (joystick.get_axis(4) + 1.0) / 2.0  # normalise to 0–1
        rt = (joystick.get_axis(5) + 1.0) / 2.0

        # ── gripper — signal to Lua script ────────────────────────────────────
        # LT wins over RT if both pressed (safety default: close).
        # The Lua sysCall_actuation reads this every step and drives the motor.
        if lt > TRIGGER_THRESHOLD:
            sim.setInt32Signal(GRIPPER_SIGNAL, 1)  # close
            gripper_open = False
        elif rt > TRIGGER_THRESHOLD:
            sim.setInt32Signal(GRIPPER_SIGNAL, 0)  # open
            gripper_open = True
        # If neither trigger is pressed the signal is left unchanged,
        # so the gripper holds its last commanded direction (the Lua motor
        # will keep running until it hits a mechanical limit — normal behaviour).

        # ── position update ───────────────────────────────────────────────────
        dx = ls_y * MOVE_SPEED  # forward / back  (CoppeliaSim X)
        dy = ls_x * MOVE_SPEED  # left   / right  (CoppeliaSim Y)
        dz = -rs_z * MOVE_SPEED  # up     / down   (CoppeliaSim Z; stick up = +Z)

        target_pos = vec_add(target_pos, [dx, dy, dz])
        target_pos = vec_clamp_to_sphere(
            robot_shoulder_world, target_pos, ROBOT_ARM_LENGTH
        )

        # ── orientation update (pitch from right stick X) ─────────────────────
        if abs(rs_pitch) > 0.0:
            delta_quat = euler_to_quaternion(0.0, rs_pitch * ROT_SPEED, 0.0)
            target_quat = normalise_quaternion(
                quaternion_multiply(target_quat, delta_quat)
            )

        # ── push position + orientation to sim ───────────────────────────────
        sim.setObjectPosition(target, target_pos)
        sim.setObjectQuaternion(target, target_quat)

        res, *_ = simIK.handleGroup(ikEnv, ikGroupUndamped, {"syncWorlds": True})
        if res != simIK.result_success:
            simIK.handleGroup(ikEnv, ikGroupDamped, {"syncWorlds": True})

        sim.step()

        # ── display ───────────────────────────────────────────────────────────
        screen.fill((30, 30, 30))
        gripper_label = "OPEN" if gripper_open else "CLOSED"
        lines = [
            f"Target   X: {target_pos[0]:.4f}  Y: {target_pos[1]:.4f}  Z: {target_pos[2]:.4f}",
            f"Quat     x: {target_quat[0]:.3f}  y: {target_quat[1]:.3f}",
            f"         z: {target_quat[2]:.3f}  w: {target_quat[3]:.3f}",
            f"Left  stick: ({ls_x:+.2f}, {ls_y:+.2f})",
            f"Right stick: pitch={rs_pitch:+.2f}  Z={rs_z:+.2f}",
            f"Gripper: {gripper_label}  (LT={lt:.2f}  RT={rt:.2f})",
            "Q / Start button to quit",
        ]
        for i, line in enumerate(lines):
            surface = font.render(line, True, (200, 200, 200))
            screen.blit(surface, (10, 10 + i * 22))
        pygame.display.flip()

except KeyboardInterrupt:
    print("Interrupted.")

finally:
    sim.clearInt32Signal(GRIPPER_SIGNAL)
    sim.stopSimulation()
    pygame.quit()
    print("Simulation stopped.")
