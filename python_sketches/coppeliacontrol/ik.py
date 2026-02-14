from coppeliasim_zmqremoteapi_client import RemoteAPIClient
from random import uniform

client = RemoteAPIClient()
sim = client.require("sim")
simIK = client.require("simIK")


def getObjectWorldPosition(objectHandle):
    return sim.getObjectPosition(objectHandle, -1)


def addVector3(one, two):
    return [one[0] + two[0], one[1] + two[1], one[2] + two[2]]


def subVector3(one, two):
    return [one[0] - two[0], one[1] - two[1], one[2] - two[2]]


simBase = sim.getObject("/YuMi")
rightShoulderAbduct = sim.getObject("/rightJoint1")
rightShoulderFlex = sim.getObject("/rightJoint1/rightLink1/rightJoint2")
# rightElbowLink = sim.getObject("/rightJoint1/rightLink1/rightJoint2/rightLink3")
rightElbowAbduct = sim.getObject(
    "/rightJoint1/rightLink1/rightJoint2/rightLink2/rightJoint3"
)
rightElbowLink = sim.getObject(
    "/rightJoint1/rightLink1/rightJoint2/rightLink2/rightJoint3/rightLink3"
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

sphere = sim.getObject("/Sphere")

shoulderIKEnv = simIK.createEnvironment()

ikGroupUndamped = simIK.createGroup(shoulderIKEnv)
simIK.setGroupCalculation(
    shoulderIKEnv, ikGroupUndamped, simIK.method_pseudo_inverse, 0, 6
)
simIK.addElementFromScene(
    shoulderIKEnv,
    ikGroupUndamped,
    rightShoulderAbduct,
    rightElbowLink,
    sphere,
    simIK.constraint_position,
)

ikGroupDamped = simIK.createGroup(shoulderIKEnv)
simIK.setGroupCalculation(
    shoulderIKEnv, ikGroupDamped, simIK.method_damped_least_squares, 1, 99
)
simIK.addElementFromScene(
    shoulderIKEnv,
    ikGroupDamped,
    rightShoulderAbduct,
    rightElbowLink,
    sphere,
    simIK.constraint_position,
)

sim.setStepping(True)
sim.startSimulation()

MAXRADIUS = (0.21492 + 0.24129) * 2
origin = getObjectWorldPosition(rightShoulderAbduct)


def getRandomSpherePos():
    return [
        uniform(-MAXRADIUS, MAXRADIUS / 2) + origin[0],
        uniform(-MAXRADIUS, MAXRADIUS / 2) + origin[1],
        uniform(-MAXRADIUS, MAXRADIUS / 2) + origin[2],
    ]


spherePos = [0, 0, 0]


def setSphereToKnownPos():
    global spherePos
    spherePos = [0.4753182281946458, 0.03174080656038679, 0.39867188031712114]
    sim.setObjectPosition(sphere, spherePos)


def randomisePositionOfSphere():
    global spherePos
    spherePos = getRandomSpherePos()
    sim.setObjectPosition(sphere, spherePos)


print(f"Origin position is at {origin}")

while (t := sim.getSimulationTime()) < 1000:
    try:
        randomisePositionOfSphere()
        res, *_ = simIK.handleGroup(
            shoulderIKEnv, ikGroupUndamped, {"syncWorlds": True}
        )
        if res != simIK.result_success:
            simIK.handleGroup(shoulderIKEnv, ikGroupDamped, {"syncWorlds": True})
        else:
            print(f"Successfully reached {spherePos}")
        sim.step()
    except KeyboardInterrupt:
        print("Stopping sim")
        sim.stopSimulation()
        break
sim.stopSimulation()
