# Microduck ROS 2 sim in a container, using the same pixi environment CI tests.
#
#   docker build -t microduck_ros2 .
#   xhost +local:docker          # allow the container to open RViz on your screen
#   docker run --rm -it --net=host -e DISPLAY -v /tmp/.X11-unix:/tmp/.X11-unix microduck_ros2
#   docker run --rm -it --net=host -e DISPLAY -v /tmp/.X11-unix:/tmp/.X11-unix microduck_ros2 pixi run nav
#
# The robot meshes (CC BY-NC-SA 4.0) and Pollen's policies are downloaded at build
# time into this local image. Don't push the image to a public registry.
FROM ubuntu:24.04

ENV DEBIAN_FRONTEND=noninteractive \
    PIXI_HOME=/opt/pixi \
    PATH=/opt/pixi/bin:$PATH \
    MICRODUCK_RL_PATH=/opt/microduck_rl \
    MICRODUCK_POLICIES_PATH=/opt/microduck-policies

RUN apt-get update && apt-get install -y --no-install-recommends \
        ca-certificates curl git python3 libgl1 libegl1 libxkbcommon-x11-0 libxcb-cursor0 \
        libxcb-icccm4 libxcb-image0 libxcb-keysyms1 libxcb-randr0 libxcb-render-util0 \
        libxcb-shape0 libxcb-xinerama0 libxcb-xkb1 \
    && rm -rf /var/lib/apt/lists/* \
    && curl -fsSL https://pixi.sh/install.sh | bash

WORKDIR /microduck_ros2
COPY pixi.toml ./
RUN pixi install

COPY . .
RUN pixi run fetch && pixi run build && pixi run map

CMD ["pixi", "run", "sim"]
