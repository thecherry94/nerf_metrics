FROM pytorch/pytorch:latest
ENV TZ=Europe/Berlin
RUN ln -snf /usr/share/zoneinfo/$TZ /etc/localtime && echo $TZ > /etc/timezone

RUN apt update && apt install -y -qq --no-install-recommends libgl1 ffmpeg libsm6 libxext6
RUN apt update && apt-get -y install sudo

ARG USER=student
ARG PASSWORD=student
ARG UID=1000
ARG GID=1000
ENV UID=${UID}
ENV GID=${GID}
ENV USER=${USER}
RUN groupadd -g "$GID" "$USER"  && \
    useradd -m -u "$UID" -g "$GID" --shell $(which bash) "$USER" -G sudo && \
    echo "$USER:$PASSWORD" | chpasswd && \
    echo "%sudo ALL=(ALL) NOPASSWD: ALL" > /etc/sudoers.d/sudogrp

USER $USER
RUN pip install -U scikit-image
RUN pip install lpips
RUN pip install pandas
RUN pip install opencv-contrib-python
RUN pip install matplotlib
