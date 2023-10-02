from body_tracking_msgs.msg import BboxKpList, BboxKp
import rospy


class BboxKpSubscriber:
    def __init__(self):
        rospy.init_node('yolo_subscriber', anonymous=False)
        self.bbox_kp_sub = rospy.Subscriber("/refined_perception/kp_bbox", BboxKpList, self.callback)

    def callback(self, data):
        trackings = []
        for tracker in data.subjects:
            tracking = {}
            tracking["bbox"] = tracker.bbox
            tracking["kp"] = tracker.kp
            trackings.append(tracking)
        print(tracking)


if __name__ == "__main__":
    # rosnode initialization
    ros_yolo = BboxKpSubscriber()
    try:
        rospy.spin()
    except KeyboardInterrupt:
        print("Shutting down")
