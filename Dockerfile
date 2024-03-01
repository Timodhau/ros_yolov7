FROM ros:noetic
ARG CLIENT_SRC_FOLDER
ARG CLIENT_WSP_FOLDER
ARG HOST_WSP_FOLDER
ARG HOST_SRC_FOLDER
ARG REQUIREMENTS_FILE
ARG ROS_MODULE_NAME
RUN apt-get update && apt-get install -y --no-install-recommends \
    git \
    libv4l-dev \
    portaudio19-dev \
    python3.8 \
    python3-empy \
    python3-pip \
    ros-$(rosversion -d)-cv-bridge \
    ros-$(rosversion -d)-image-geometry \
    ros-$(rosversion -d)-libuvc-camera \
    ros-$(rosversion -d)-tf \
    && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*
COPY .$REQUIREMENTS_FILE .
RUN pip3 install --ignore-installed -r requirements.txt
RUN mkdir -p $CLIENT_SRC_FOLDER
COPY $HOST_SRC_FOLDER/$ROS_MODULE_NAME $CLIENT_SRC_FOLDER/$ROS_MODULE_NAME
COPY $HOST_SRC_FOLDER/mudialbot_msgs $CLIENT_SRC_FOLDER/mudialbot_msgs
WORKDIR $CLIENT_WSP_FOLDER
RUN [ "/bin/sh", "-c", ". /opt/ros/${ROS_DISTRO}/setup.sh && catkin_make" ]
