import rospy
from std_msgs.msg import Float64MultiArray



goalPub = rospy.Publisher('/Kwad/goal', Float64MultiArray, queue_size=4)

rospy.init_node('globalControl', anonymous=True)

rate = rospy.Rate(0.5)

newGoal = [2, 0, 1, 0, 0, 0]

goalPub.publish(newGoal)

rate.sleep()

newGoal = [2, 2, 1, 0, 0, 0]

goalPub.publish(newGoal)

rate.sleep()