# ros yolov7 bounding boxes + body keypoints
Ros package publishing ros yolov7 output tested with ros noetic but should would with others ros versions aswell <br/>

Based on https://github.com/RizwanMunawar/yolov7-pose-estimation <br/>

Follow the tutorial to install yolov7 required packages <br/>

Configure launch file: <br/>
<ul>
  <li>/topic_img for your input topic</li>
  <li>/path_yolo_model path to yolo weight downloaded following https://github.com/RizwanMunawar/yolov7-pose-estimation tutorial</li>
  <li>use launch-prefix if you want to start the node with a python env, else remove it</li>
</ul>

start with: 
<pre><code>roslaunch ros_yolov7 pose_estimate.launch
</code></pre>

An example of subscriber is in src/ directory

Packages installations
    
    cd /src
    pip install -r requirements.txt
    pip install --extra-index-url https://rospypi.github.io/simple/ rospy rosbag rospkg
    pip install scikit-learn==0.22.2 --upgrade
# ros module documentation
## formats used
### BboxKp
- int64 id : a user's id
- float64[] bbox : a user's bounding boxes
- float64[] kp : a user's keypoints
### BboxKpList
- BboxKp[] subjects : a list of BboxKp
## ROS topics
### subscribed to
- [topic_img] (the parameter defined in the launch file) : Image, the image stream feeded to the service
### published
- /refined_perception/kp_bbox : BboxKpList, the users' bounding boxes and keypoints, useful for identifying them and interpret their position/body posture
