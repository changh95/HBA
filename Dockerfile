FROM nvidia/opengl:1.2-glvnd-devel-ubuntu20.04

ENV NVIDIA_VISIBLE_DEVICES \
    ${NVIDIA_VISIBLE_DEVICES:-all}
ENV NVIDIA_DRIVER_CAPABILITIES \
    ${NVIDIA_DRIVER_CAPABILITIES:+$NVIDIA_DRIVER_CAPABILITIES,}graphics

ARG DEBIAN_FRONTEND=noninteractive

RUN apt-get update && apt-get install -y --no-install-recommends \
    wget nano build-essential libomp-dev clang lld git python3 python3-pip vim sudo curl gnupg
RUN apt-get install -y lsb-release &&\
    sudo sh -c 'echo "deb http://packages.ros.org/ros/ubuntu $(lsb_release -sc) main" > /etc/apt/sources.list.d/ros-latest.list' &&\
    curl -s https://raw.githubusercontent.com/ros/rosdistro/master/ros.asc | sudo apt-key add - &&\
    apt-get update -y &&\
    apt install -y ros-noetic-desktop-full ros-noetic-geodesy ros-noetic-pcl-ros ros-noetic-nmea-msgs \
    ros-noetic-rviz ros-noetic-tf-conversions ros-noetic-gtsam libtbb-dev libceres-dev \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*
RUN pip3 install rosdep
RUN rosdep init && rosdep update

RUN pip install catkin_tools catkin_tools_fetch rosinstall_generator

SHELL ["/bin/bash", "-c"]

RUN mkdir -p ~/catkin_ws/src

RUN echo "source /opt/ros/noetic/setup.bash" >> /root/.bashrc \
    && echo "source /root/catkin_ws/devel/setup.bash" >> /root/.bashrc

WORKDIR /root/catkin_ws

# Example run command with volume mount for development:
# docker run -it --net=host --env DISPLAY=$DISPLAY --volume /tmp/.X11-unix:/tmp/.X11-unix --volume ~/data/data:/data --volume $(pwd):/root/catkin_ws/src/HBA --gpus=all apr:hba

# Alternative with privileged mode if needed:
# docker run -it -e QT_XCB_GL_INTEGRATION=xcb_egl --net=host --env DISPLAY=$DISPLAY --volume /tmp/.X11-unix:/tmp/.X11-unix --volume ~/data/data:/data --volume $(pwd):/root/catkin_ws/src/HBA --gpus=all --privileged apr:hba

# Make sure to catkin_make after building the docker image