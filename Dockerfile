FROM nvidia/opengl:1.2-glvnd-devel-ubuntu22.04

ENV NVIDIA_VISIBLE_DEVICES \
    ${NVIDIA_VISIBLE_DEVICES:-all}
ENV NVIDIA_DRIVER_CAPABILITIES \
    ${NVIDIA_DRIVER_CAPABILITIES:+$NVIDIA_DRIVER_CAPABILITIES,}graphics

ARG DEBIAN_FRONTEND=noninteractive

RUN apt-get update && apt-get install -y --no-install-recommends \
    wget nano build-essential libomp-dev clang lld git vim sudo curl gnupg cmake software-properties-common python3 python3-pip libboost-all-dev

RUN add-apt-repository ppa:borglab/gtsam-develop && \
    apt install -y libgtsam-dev libgtsam-unstable-dev libpcl-dev libeigen3-dev

RUN python3 --version && pip3 install rerun-sdk

SHELL ["/bin/bash", "-c"]

RUN mkdir -p ~/hba

WORKDIR /root/hba

# Example run command with volume mount for development:
# docker run -it --net=host --env DISPLAY=$DISPLAY --volume /tmp/.X11-unix:/tmp/.X11-unix --volume ~/data/data:/data --volume $(pwd):/root/hba --gpus=all apr:hba

# Alternative with privileged mode if needed:
# docker run -it -e QT_XCB_GL_INTEGRATION=xcb_egl --net=host --env DISPLAY=$DISPLAY --volume /tmp/.X11-unix:/tmp/.X11-unix --volume ~/data/data:/data --volume $(pwd):/root/hba --gpus=all --privileged apr:hba

# Make sure to catkin_make after building the docker image