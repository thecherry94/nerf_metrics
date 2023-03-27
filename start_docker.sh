#!/bin/sh
uid=$(eval "id -u")
gid=$(eval "id -g")
docker build --build-arg UID="$uid" --build-arg GID="$gid" -t nerf_evaluation .

docker run --rm -it --privileged --net=host -v $PWD:/opt/project:rw --ipc=host -e DISPLAY="$DISPLAY" --gpus all nerf_evaluation
