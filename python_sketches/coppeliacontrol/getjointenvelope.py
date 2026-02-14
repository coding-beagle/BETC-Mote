from math import pi

from coppeliasim_zmqremoteapi_client import RemoteAPIClient

client = RemoteAPIClient()
sim = client.require("sim")

DEG_TO_RAD = pi / 180
RAD_TO_DEG = 180 / pi


def getTipPosition():
    return sim.getObjectPosition(rightElbowLink, -1)


simBase = sim.getObject("/YuMi")
rightShoulderAbduct = sim.getObject("/rightJoint1")
rightShoulderFlex = sim.getObject("/rightJoint1/rightLink1/rightJoint2")
rightElbowLink = sim.getObject("/rightJoint1/rightLink1/rightJoint2/rightLink2")
rightElbowAbduct = sim.getObject(
    "/rightJoint1/rightLink1/rightJoint2/rightLink2/rightJoint3"
)
rightElbowFlex = sim.getObject(
    "/rightJoint1/rightLink1/rightJoint2/rightLink2/rightJoint3/rightLink3/rightJoint4"
)
rightWrist = sim.getObject(
    "/rightJoint1/rightLink1/rightJoint2/rightLink2/rightJoint3/rightLink3/rightJoint4/rightLink4/rightJoint5/rightLink5/rightJoint6"
)
rightWristLink = sim.getObject(
    "/rightJoint1/rightLink1/rightJoint2/rightLink2/rightJoint3/rightLink3/rightJoint4/rightLink4/rightJoint5/rightLink5"
)

leftShoulderAbduct = sim.getObject("/leftJoint1")
leftShoulderFlex = sim.getObject("/leftJoint1/leftLink1/leftJoint2")
leftElbowLink = sim.getObject("/leftJoint1/leftLink1/leftJoint2/leftLink2")
leftElbowAbduct = sim.getObject("/leftJoint1/leftLink1/leftJoint2/leftLink2/leftJoint3")
leftElbowFlex = sim.getObject(
    "/leftJoint1/leftLink1/leftJoint2/leftLink2/leftJoint3/leftLink3/leftJoint4"
)
leftWrist = sim.getObject(
    "/leftJoint1/leftLink1/leftJoint2/leftLink2/leftJoint3/leftLink3/leftJoint4/leftLink4/leftJoint5/leftLink5/leftJoint6"
)
leftWristLink = sim.getObject(
    "/leftJoint1/leftLink1/leftJoint2/leftLink2/leftJoint3/leftLink3/leftJoint4/leftLink4/leftJoint5/leftLink5"
)

sim.setStepping(True)

sim.startSimulation()

# move stuff out of the way
sim.setJointTargetPosition(leftShoulderAbduct, 0 * DEG_TO_RAD)
sim.setJointTargetPosition(rightElbowAbduct, 0 * DEG_TO_RAD)
sim.setJointTargetPosition(rightElbowFlex, 0 * DEG_TO_RAD)


joint_base_angle = -168.50
joint_range = 330
joint_steps = 500
joint_angle_increment = joint_range / joint_steps  # degrees

joint_link_2_angle = 0
joint_link_2_increment = 0.5

while (t := sim.getSimulationTime()) < 36:
    try:
        sim.setJointTargetPosition(rightShoulderAbduct, joint_base_angle * DEG_TO_RAD)
        joint_base_angle -= joint_angle_increment
        joint_base_angle %= 360
        print(f"Tip is at {getTipPosition()}")
        sim.step()
    except KeyboardInterrupt:
        sim.stopSimulation()
        break
sim.stopSimulation()
