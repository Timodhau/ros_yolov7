FROM ros:noetic
ARG HOST_WORKSPACE_FOLDER
ARG VIRTUAL_ENV
ARG WORKSPACE_NAME
ARG REQUIREMENTS_FILE
ARG ROS_MODULE_NAME
ARG CLIENT_SRC_FOLDER
ARG HOST_WORKSPACE_SRC_FOLDER
# https://pythonspeed.com/articles/activate-virtualenv-dockerfile/
# changer le shell par défaut pour pouvoir lancer `source`
RUN rm /bin/sh && ln -s /bin/bash /bin/sh
RUN apt-get update && apt-get install -y --no-install-recommends \
    git \
    python3.8 \
    python3-pip \
    python3-venv \
    python3-empy \
    ros-$(rosversion -d)-cv-bridge \
    ros-$(rosversion -d)-image-geometry \
    ros-$(rosversion -d)-libuvc-camera \
    ros-$(rosversion -d)-tf \
    libv4l-dev \
    portaudio19-dev \
    && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*
RUN python3 -m venv $VIRTUAL_ENV
COPY .$REQUIREMENTS_FILE .
RUN pip3 install --upgrade --force-reinstall -v "pip==23.0.1"
RUN pip3 install --ignore-installed -r requirements.txt
RUN rm requirements.txt
RUN mkdir -p $CLIENT_SRC_FOLDER
COPY $HOST_WORKSPACE_SRC_FOLDER/$ROS_MODULE_NAME $CLIENT_SRC_FOLDER/$ROS_MODULE_NAME
COPY $HOST_WORKSPACE_SRC_FOLDER/mudialbot_msgs $CLIENT_SRC_FOLDER/mudialbot_msgs
WORKDIR $WORKSPACE_NAME
ENV PATH="$PATH:$VIRTUAL_ENV"
RUN [ "/bin/bash", "-c", "source /opt/ros/${ROS_DISTRO}/setup.bash && catkin_make" ]
RUN env

