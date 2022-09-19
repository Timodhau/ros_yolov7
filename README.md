# ros_yolov7
Ros package publishing ros yolov7 output

Based on https://github.com/RizwanMunawar/yolov7-pose-estimation

Follow the tutorial to install yolov7 required packages

Configure launch file:
/topic_img for your input topic
/path_yolo_model path to yolo weight downloaded following https://github.com/RizwanMunawar/yolov7-pose-estimation tutorial
user launch-prefix if you want to start the node with a python env, else remove it

