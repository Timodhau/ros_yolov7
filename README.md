# ros_yolov7
Ros package publishing ros yolov7 output tested with ros noetic but should would with others ros version aswell <br/>

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
