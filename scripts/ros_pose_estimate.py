#!/home/tim/env/yoloV9/bin/python

import sys
# print(sys.path)

import rospy
from cv_bridge import CvBridge, CvBridgeError
from ros_yolov7.msg import BboxKpList, BboxKp
from sensor_msgs.msg import Image
import numpy as np
import cv2
import torch
from torchvision import transforms
from utils.datasets import letterbox
from utils.torch_utils import select_device
from models.experimental import attempt_load
from utils.plots import output_to_keypoint, plot_skeleton_kpts
from utils.general import non_max_suppression_kpt, strip_optimizer


class RosYolo:
    def __init__(self):

        # Variables
        self.POSEWEIGHTS = 'yolov7-w6-pose.pt'  #
        self.DEVICE = '0'
        self.TOPIC_IMG = rospy.get_param("/topic_img")

        self.bridge = CvBridge()

        # ROS Topics
        # self.image_sub = rospy.Subscriber("/usb_cam/image_raw/", Image, self.callback)
        self.image_sub = rospy.Subscriber(self.TOPIC_IMG, Image, self.callback)
        self.kp_bbox_pub = rospy.Publisher("/refined_perception/kp_bbox", BboxKpList, queue_size=10)

        # select device
        self.DEVICE = select_device(self.DEVICE)
        # Load model
        self.loading: bool = False
        self.model = attempt_load(self.POSEWEIGHTS, map_location=self.DEVICE)  # load FP32 model
        _ = self.model.eval()
        self.loading = True

    @torch.no_grad()
    def callback(self, data):
        if self.loading:
            try:
                cv_image = self.bridge.imgmsg_to_cv2(data, "bgr8")  # (480, 640, 3)
                image = cv2.cvtColor(cv_image, cv2.COLOR_BGR2RGB)
                image = letterbox(image, (640), stride=64, auto=True)[0]
                # print(image.shape) (512, 640, 3)
                image_ = image.copy()
                image = transforms.ToTensor()(image)
                image = torch.tensor(np.array([image.numpy()]))
                # convert image data to device
                image = image.to(self.DEVICE)
                # convert image to float precision (cpu)
                image = image.float()
                # get predictions
                with torch.no_grad():
                    output, _ = self.model(image)
                # Apply non-max suppression
                output = non_max_suppression_kpt(output, 0.25, 0.65, nc=self.model.yaml['nc'],
                                                 nkpt=self.model.yaml['nkpt'],
                                                 kpt_label=True)
                output = output_to_keypoint(output)
                im0 = image[0].permute(1, 2, 0) * 255
                im0 = im0.cpu().numpy().astype(np.uint8)
                # reshape image format to (BGR)
                im0 = cv2.cvtColor(im0, cv2.COLOR_RGB2BGR)
                # print(f"output.shape {output.shape}")
                bbox_per_id = []
                for idx in range(output.shape[0]):
                    plot_skeleton_kpts(im0, output[idx, 7:].T, 3)
                    # id = 10
                    # x_coord, y_coord = int(output[idx, 3 * id + 7]), int(output[idx, 3 * id + 1 + 7])
                    # cv2.circle(im0, (x_coord, y_coord), 5, (255, 122, 122), -1)
                    xmin, ymin = (output[idx, 2] - output[idx, 4] / 2), (output[idx, 3] - output[idx, 5] / 2)
                    xmax, ymax = (output[idx, 2] + output[idx, 4] / 2), (output[idx, 3] + output[idx, 5] / 2)
                    bbox_per_id.append([xmin, ymin, xmax, ymax])

                    # Plotting key points on Image
                    cv2.rectangle(im0, (int(xmin), int(ymin)), (int(xmax), int(ymax)), color=(255, 0, 0),
                                  thickness=1, lineType=cv2.LINE_AA)
                if len(output) == 0:
                    self.kp_bbox_pub.publish("empty")
                else:
                    self.publish_bbox_kp(output, bbox_per_id)
            except CvBridgeError as e:
                print(e)
            cv2.imshow("Image window", im0)
            cv2.waitKey(3)

    def publish_bbox_kp(self, output_, bbox_per_id_):
        """
        publish bbox + kp
        input: np.array(idx, 58)
        0-1 -> object id
        2-6 -> bbox [x_center, y_center, width, height, confidence]
        7-58 -> kp [x,y,confidence] 17 kps
        """
        data_to_send = BboxKpList()
        tracking_list = []
        for idx in range(output_.shape[0]):
            tracking = BboxKp()
            if output_[idx, 6] > 0.7:
                tracking.bbox = bbox_per_id_[idx]
                tracking.kp = output_[idx, 7:]
            tracking_list.append(tracking)
        data_to_send.subjects = tracking_list
        self.kp_bbox_pub.publish(data_to_send)
        # print("------------------------------------")
        # print(output_)
        # print(str_to_publish)


if __name__ == "__main__":
    start = rospy.get_param("/start_yolo_module")
    # rosnode initialization
    rospy.init_node('yolo_module', anonymous=False)
    if start:
        ros_yolo = RosYolo()
        try:
            rospy.spin()
        except KeyboardInterrupt:
            print("Shutting down")
        cv2.destroyAllWindows()
    else:
        rospy.loginfo("Not starting YOLO_MODULE")
        while not rospy.is_shutdown():
            rospy.Rate(1)
