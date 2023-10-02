#!/home/tim/env/yoloV9/bin/python

import sys
# print(sys.path)

import rospy
from cv_bridge import CvBridge, CvBridgeError
from body_tracking_msgs.msg import BboxKpList, BboxKp
from sensor_msgs.msg import Image, CompressedImage
import numpy as np
import cv2
import torch
from torchvision import transforms
import time
from utils.datasets import letterbox
from utils.torch_utils import select_device
from models.experimental import attempt_load
from utils.plots import output_to_keypoint, plot_skeleton_kpts
from utils.general import non_max_suppression_kpt, strip_optimizer

from deep_sort.deep_sort import DeepSort


class RosYolo:
    def __init__(self):

        # Variables
        self.POSEWEIGHTS = rospy.get_param("/path_models") + 'yolov7-w6-pose.pt'  #
        self.DEEPSORT_WEIGHT = rospy.get_param("/path_models") + 'ckpt.t7'  #
        self.DEVICE = '0'
        self.TOPIC_IMG = rospy.get_param("/topic_img")

        self.bridge = CvBridge()

        # select device
        self.DEVICE = select_device(self.DEVICE)
        # Load model
        self.loading: bool = False
        self.model = attempt_load(self.POSEWEIGHTS, map_location=self.DEVICE)  # load FP32 model
        _ = self.model.eval()
        self.loading = True
        self.cpt = 0

        # DeepSort initialization
        self.deepsort = DeepSort(self.DEEPSORT_WEIGHT, max_dist=0.2, min_confidence=0.3, nms_max_overlap=1.0,
                                 max_iou_distance=0.7, max_age=70, n_init=4, nn_budget=100, use_cuda=True)

        # ROS Topics
        # self.image_sub = rospy.Subscriber("/usb_cam/image_raw/", Image, self.callback)
        self.image_sub = rospy.Subscriber(self.TOPIC_IMG, CompressedImage, self.callback)
        self.kp_bbox_pub = rospy.Publisher("/refined_perception/kp_bbox", BboxKpList, queue_size=0)

    @torch.no_grad()
    def callback(self, data):
        if self.loading:
            try:
                pixels = np.asarray(bytearray(data.data), dtype='uint8')
                cv_image = cv2.imdecode(pixels, cv2.IMREAD_COLOR)
                # cv_image = self.bridge.imgmsg_to_cv2(data, "bgr8")  # (480, 640, 3)
                # cv_image = np.asarray(self.bridge.imgmsg_to_cv2(data, '8UC1'))
                image = cv2.cvtColor(cv_image, cv2.COLOR_BGR2RGB)
                image = letterbox(image, (640), stride=64, auto=True)[0]
                # print(image.shape)   # (512, 640, 3)
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
                deepsort_bboxes = []
                kp_per_id = []
                confidence_per_id = []
                for idx in range(output.shape[0]):
                    plot_skeleton_kpts(im0, output[idx, 7:].T, 3)
                    # id = 10
                    # x_coord, y_coord = int(output[idx, 3 * id + 7]), int(output[idx, 3 * id + 1 + 7])
                    # cv2.circle(im0, (x_coord, y_coord), 5, (255, 122, 122), -1)
                    xmin, ymin = (output[idx, 2] - output[idx, 4] / 2), (output[idx, 3] - output[idx, 5] / 2)
                    xmax, ymax = (output[idx, 2] + output[idx, 4] / 2), (output[idx, 3] + output[idx, 5] / 2)
                    bbox_per_id.append([xmin, ymin, xmax, ymax])
                    deepsort_bboxes.append([output[idx, 2],output[idx, 3],output[idx, 4],output[idx, 5]])
                    kp_per_id.append(output[idx, 7:].tolist())
                    confidence_per_id.append(output[idx, 6])
                    # Plotting key points on Image
                    cv2.rectangle(im0, (int(xmin), int(ymin)), (int(xmax), int(ymax)), color=(255, 0, 0),
                                  thickness=1, lineType=cv2.LINE_AA)
                if len(output) == 0:
                    self.kp_bbox_pub.publish(BboxKpList())
                else:
                    deepsort_tracking = self.deepsort.update(np.array(deepsort_bboxes), confidence_per_id, image_)
                    if len(bbox_per_id)==len(deepsort_tracking):
                        assignment_list = self.pair_dpbbox(bbox_per_id, deepsort_tracking)
                        for id, bbox in enumerate(bbox_per_id):
                            if id < len(deepsort_tracking):
                                deepsort_id = deepsort_tracking[assignment_list[id]][-1]
                                cv2.putText(img=im0, text=str(deepsort_id), org=(int(bbox[0]), int(bbox[1])),
                                            fontFace=cv2.FONT_HERSHEY_TRIPLEX,
                                            fontScale=1, color=(255, 0, 0), thickness=2)
                        self.publish_bbox_kp(bbox_per_id, kp_per_id, deepsort_tracking, assignment_list)
            except CvBridgeError as e:
                print(e)
            cv2.imshow("Image window", im0)
            cv2.waitKey(3)

    def publish_bbox_kp(self, bbox_per_id_, kp_per_id_, deepsort_tracking_, assignment_list):
        """
        bbox_per_id: [xmin, ymin, xmax, ymax]
        deepsort_tracking: [x1,y1,x2,y2,track_id]

        publish bbox[x_min, y_min, x_max, y_max] + kp
        input: np.array(idx, 58)
        0-1 -> object id
        2-6 -> bbox [x_center, y_center, width, height, confidence]
        7-58 -> kp [x,y,confidence] 17 kps
        """
        data_to_send = BboxKpList()
        tracking_list = []
        for idx in range(len(bbox_per_id_)):
            tracking = BboxKp()
            if idx < len(deepsort_tracking_):
                tracking.bbox = bbox_per_id_[idx]
                tracking.kp = kp_per_id_[idx]
                tracking.id = deepsort_tracking_[assignment_list[idx]][-1]
            tracking_list.append(tracking)
        if len(tracking_list) > 0:
            data_to_send.subjects = tracking_list
            self.kp_bbox_pub.publish(data_to_send)

    def pair_dpbbox(self, bbox_per_id, bbox_deepsort):
        """ pair bbox output from deepsort with bbox output from yolo
        return list of deepsort corresponding id """
        iou_list = []
        for i, bbox in enumerate(bbox_per_id):
            ious = []
            for j, deepbbox in enumerate(bbox_deepsort):
                bb1 = self.convert_bbox_dict(deepbbox[:4])
                bb2 = self.convert_bbox_dict(bbox)
                ious.append(self.get_iou(bb1, bb2))
            iou_list.append(ious)
        assignment_list = np.argmax(np.array(iou_list),axis=1)
        return assignment_list

    @staticmethod
    def get_iou(bb1, bb2, convert=False):
        """
        Calculate the Intersection over Union (IoU) of two bounding boxes.

        Parameters
        ----------
        bb1 : dict
            Keys: {'x1', 'x2', 'y1', 'y2'}
            The (x1, y1) position is at the top left corner,
            the (x2, y2) position is at the bottom right corner
        bb2 : dict
            Keys: {'x1', 'x2', 'y1', 'y2'}
            The (x, y) position is at the top left corner,
            the (x2, y2) position is at the bottom right corner

        Returns
        -------
        float
            in [0, 1]
        """
        # if convert:
        #     bb1 = convert_bbox_dict(bb1)
        #     bb2 = convert_bbox_dict(bb2)
        assert bb1['x1'] < bb1['x2']
        assert bb1['y1'] < bb1['y2']
        assert bb2['x1'] < bb2['x2']
        assert bb2['y1'] < bb2['y2']

        # determine the coordinates of the intersection rectangle
        x_left = max(bb1['x1'], bb2['x1'])
        y_top = max(bb1['y1'], bb2['y1'])
        x_right = min(bb1['x2'], bb2['x2'])
        y_bottom = min(bb1['y2'], bb2['y2'])

        if x_right < x_left or y_bottom < y_top:
            return 0.0

        # The intersection of two axis-aligned bounding boxes is always an
        # axis-aligned bounding box
        intersection_area = (x_right - x_left) * (y_bottom - y_top)

        # compute the area of both AABBs
        bb1_area = (bb1['x2'] - bb1['x1']) * (bb1['y2'] - bb1['y1'])
        bb2_area = (bb2['x2'] - bb2['x1']) * (bb2['y2'] - bb2['y1'])

        # compute the intersection over union by taking the intersection
        # area and dividing it by the sum of prediction + ground-truth
        # areas - the interesection area
        iou = intersection_area / float(bb1_area + bb2_area - intersection_area)
        assert iou >= 0.0
        assert iou <= 1.0
        return iou

    @staticmethod
    def convert_bbox_dict(bbox):
        """
        take as input bbox list
        [x,y,width,height]
        return bbox dict {x1,y1,x2,y2}
        """
        bbox_dict = {}
        bbox_dict['x1'] = bbox[0]
        bbox_dict['y1'] = bbox[1]
        bbox_dict['x2'] = bbox[2]
        bbox_dict['y2'] = bbox[3]
        return bbox_dict


if __name__ == "__main__":
    # rosnode initialization
    rospy.init_node('yolo_module', anonymous=False)
    ros_yolo = RosYolo()
    time.sleep(0.2)
    try:
        rospy.spin()
    except KeyboardInterrupt:
        print("Shutting down")
    cv2.destroyAllWindows()
