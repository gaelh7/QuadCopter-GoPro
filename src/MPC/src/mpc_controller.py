# !/usr/bin/env python3
# ---------------------------------------------------
from pyexpat.errors import XML_ERROR_ABORTED
import rospy
from gazebo_msgs.msg import ModelStates
from std_msgs.msg import Float64MultiArray, Float32, Float64
from geometry_msgs.msg import Pose
from tf.transformations import euler_from_quaternion
import numpy as np
from itertools import product
from multiprocessing.pool import ThreadPool

goalX = 0
goalY = 0
goalZ = 1
goalRoll = 0
goalPitch = 0
goalYaw = 0

m_p = 0.0055 # Mass of propeller (kg)
r_p = 0.20 # Radius of propeller (m)
m_L = 0.01 # Mass of quadcopter arm (kg)
m_b = 0.01 # Mass of the central box (kg)
arm_L = 0.02 # Length of arm (m)
a = 0.12 # Side length of box (m)
h = 0.01 # Height of box (m)
totalMass = 0.5 # (kg)
Ip = m_p * r_p # Moment of inertia of propellors

# Total moment of inertia about local z-axis
Iqy = 2*m_L*arm_L + (1/6.)*m_b*(a ** 2) + 4*m_p*arm_L
inv_Iqy = Iqy**(-1) # Precompute inverse for speed

# Total moment of inertia about local x and y-axes
Iqpr = np.sqrt(2)*m_L*arm_L + (1/12.)*m_b*((a**2)+(h**2)) + 2*np.sqrt(2)*m_p*arm_L
inv_Iqpr = Iqpr**(-1)

global R
global coms

aird = 1.2041
k = 1.05 # TODO: The hell is this?

def A(dt):
    # x, y, z, dx, dy, dz, rl, pt, yw, drl, dpt, dyw
    return np.array([
        [1, 0, 0, dt, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 1, 0, 0, dt, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 1, 0, 0, dt, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 1, 0, 0, dt, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 1, 0, 0, dt, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, dt],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1]
    ])

def v0(state):
    # Calculate normal component of air velocity infront of the propellers
    # assume static air wrt global frame
    # https://www.grc.nasa.gov/www/k-12/airplane/propth.html
    global R

    unitVec = np.array([0,0,1]).reshape(3,1)

    # Update Rotation Matrices for global to local
    Rx = np.array([1, 0, 0, 0, np.cos(state[6]), -np.sin(state[6]), 0, np.sin(state[6]), np.cos(state[6])]).reshape(3,3)
    Ry = np.array([np.cos(state[7]), 0, np.sin(state[7]), 0, 1, 0, -np.sin(state[7]), 0, np.cos(state[7])]).reshape(3,3)
    Rz = np.array([np.cos(state[8]), -np.sin(state[8]), 0, np.sin(state[8]), np.cos(state[8]), 0, 0, 0, 1]).reshape(3,3)

    R = np.matmul(Rz, Ry, Rx)

    # Generate Unit vector of the Kwad's angular orientation
    angleUnitVec = np.matmul(R, unitVec)
    return np.dot(angleUnitVec.reshape(3), state[3:6])

def forces(state, commands):
    # Compute the upward thrust from every motor
    return 0.5*aird*np.pi*(r_p**2)*k*commands*(k*commands + 2*v0(state))

def stepDynamics(state, commands, dt):
    # Using our model, propagate the dynamics forward in time
    f = forces(state, commands)

    newState = np.copy(state)

    # Then update based on commands
    dcommands = commands - coms
    dang_mom = Ip*(dcommands[0] + dcommands[3] - dcommands[1] - dcommands[2])
    dyaw_local = dang_mom*inv_Iqy
    dpitch_local = dt*np.sqrt(0.5)*(f[1] + f[3] - f[0] - f[2])*inv_Iqpr
    droll_local = dt*np.sqrt(0.5)*(f[0] + f[2] - f[1] - f[3])*inv_Iqpr

    # TODO: Should we use a different rotation matrix for angles.
    glob_c = np.matmul(R, np.array([droll_local, dpitch_local, dyaw_local]))

    newState[9] += glob_c[0]
    newState[10] += glob_c[1]
    newState[11] += glob_c[2]

    unitVec = np.array([0,0,1]).reshape(3,1)
    angleUnitVec = np.matmul(R, unitVec).reshape(3)

    # Total thrust from props
    fTotal = np.sum(f)

    # Update acceleration in X, Y, Z global frames
    newState[3] += dt*(fTotal*np.dot(angleUnitVec,[1,0,0])/totalMass)
    newState[4] += dt*(fTotal*np.dot(angleUnitVec,[0,1,0])/totalMass)
    newState[5] += dt*(-9.81 + fTotal*np.dot(angleUnitVec,[0,0,1])/totalMass)

    # Now update according to state dynamics
    return np.matmul(A(dt), newState)

def cost(state):
    # Calculate position cost based off of how far to goal
    global goalX, goalY, goalZ, goalRoll, goalPitch, goalYaw
    xErr = state[0] - goalX
    yErr = state[1] - goalY
    zErr = state[2] - goalZ
    rollErr = state[6] - goalRoll
    pitchErr = state[7] - goalPitch
    yawErr = state[8] - goalYaw

    # Tuning Weights on different State Vars
    coeffs = [1, 1, 1, 0, 0, 1]

    cost = \
        coeffs[0]*abs(xErr) + \
        coeffs[1]*abs(yErr) + \
        coeffs[2]*abs(zErr) + \
        coeffs[3]*abs(rollErr) + \
        coeffs[4]*abs(pitchErr) + \
        coeffs[5]*abs(yawErr)

    return float(cost)

def step(state, commands, dt):
    def func((frontRightDU, frontLeftDU, backRightDU, backLeftDU)):
        newState = state
        # Propagate dynamics 3x
        new_com = np.array([
            commands[0]+frontRightDU,
            commands[1]+frontLeftDU,
            commands[2]+backLeftDU,
            commands[3]+backRightDU])
        for _ in range(1):
            newState = stepDynamics(newState, new_com, dt)
        return cost(newState), new_com
    return func

def MPC(state, commands):
    global coms
    # At the current state, try out a combination of primatives and evaluate the cost after propagating the dynamics forward
    uPrims = [-7, -3, 0, 3, 7] # TODO: Maybe include smaller primitives also?
    dt = 0.1

    pool = ThreadPool(20)
    # For each combination of primitives
    steps = pool.map(step(state, commands, dt), product(uPrims, repeat=4))
    _, bestPrims = min(steps, key=(lambda x : x[0]))
    pool.close()
    pool.terminate()

    retArr = Float64MultiArray()
    retArr.data = bestPrims.tolist()
    coms = bestPrims
    return retArr

# ---------------------------------------------------
def control_kwad(msg, args):
    # Declare global variables as you dont want these to die, reset to zero and then re-initiate when the function is called again.
    global roll, pitch, yaw

    # Assign the Float64MultiArray object to 'f' as we will have to send data of motor velocities to gazebo in this format
    #f = Float64MultiArray()
    global coms

    # Convert the quaternion data to roll, pitch, yaw data
    # The model_states contains the position, orientation, velocities of all objects in gazebo. In the simulation, there are objects like: ground, Contruction_cone, quadcopter (named as 'Kwad') etc. So 'msg.pose[ind]' will access the 'Kwad' object's pose information i.e the quadcopter's pose.
    ind = msg.name.index('Kwad')

    orientationObj = msg.pose[ind].orientation
    positionObj = msg.pose[ind].position
    deltaOrientationObj = msg.twist[ind].angular
    deltaPositionObj = msg.twist[ind].linear

    x, y, z = positionObj.x, positionObj.y, positionObj.z
    dx, dy, dz = deltaPositionObj.x, deltaPositionObj.y, deltaPositionObj.z
    dRoll, dPitch, dYaw = deltaOrientationObj.x, deltaOrientationObj.y, deltaOrientationObj.z
    orientationList = [orientationObj.x, orientationObj.y, orientationObj.z, orientationObj.w]
    (roll, pitch, yaw) = (euler_from_quaternion(orientationList))

    state = np.array([
        x, y, z,
        dx, dy, dz,
        roll, pitch, yaw,
        dRoll, dPitch, dYaw
    ]).reshape((12,1))
    # send roll, pitch, yaw data to PID() for attitude-stabilisation, along with 'f', to obtain 'fUpdated'
    # Alternatively, you can add your 'control-file' with other algorithms such as Reinforcement learning, and import the main function here instead of PID().
    steadyComs = np.array([50, -50, 50, -50])
    fUpdated = MPC(state, steadyComs) #np.array(f.data))

    # The object args contains the tuple of objects (velPub, err_rollPub, err_pitchPub, err_yawPub. publish the information to namespace.
    args.publish(fUpdated)
    # print("Roll: ",roll*(180/3.141592653),"Pitch: ", pitch*(180/3.141592653),"Yaw: ", yaw*(180/3.141592653))
    # print(orientationObj)
# ----------------------------------------------------





def processGoal(msg):
    global goalX, goalY, goalZ, goalRoll, goalPitch, goalYaw
    goalX = msg.data[0]
    goalY = msg.data[1]
    goalZ = msg.data[2]
    goalYaw = msg.data[3]

coms = np.array([50, -50, 50, -50])

# Initiate the node that will control the gazebo model
rospy.init_node("Control")

# initialte publisher velPub that will publish the velocities of individual BLDC motors
velPub = rospy.Publisher('/Kwad/joint_motor_controller/command', Float64MultiArray, queue_size=4)

# initCom = Float64MultiArray()
# initCom.data = [0, 0, 0, 0]
# velPub.publish(initCom)

GoalSub = rospy.Subscriber('/Kwad/goal', Float64MultiArray, processGoal)

# Subscribe to /gazebo/model_states to obtain the pose in quaternion form
# Subscriber the run controller on every pose update
PoseSub = rospy.Subscriber('/gazebo/model_states',ModelStates,control_kwad,velPub)

rospy.spin()
