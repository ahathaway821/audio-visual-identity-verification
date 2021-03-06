 
# FROM arm64v8/ubuntu:xenial-20180123
FROM arm64v8/ubuntu:bionic


# AUTHOR bmwshop@gmail.com
# This is the base container for the Jetson TX2 board with drivers (with cuda)

# base URL for NVIDIA libs
ARG URL=http://169.44.201.108:7002/jetpacks/4.3

ENV DEBIAN_FRONTEND=noninteractive
# Update packages, install some useful packages
RUN apt-get update && apt-get install -y qemu-user-static gnupg2 apt-utils lbzip2 curl sudo unp python3 python libegl1 libx11-6 libxext6 libgles2 libwayland-egl1 libxkbcommon0 libasound2 libglib2.0 libgstreamer1.0 libgstreamer-plugins-bad1.0 libgstreamer-plugins-base1.0 libevdev2 libinput10 libunwind8 device-tree-compiler 
WORKDIR /tmp

# Install drivers first
RUN curl -sL $URL/Jetson_Linux_R32.3.1_aarch64.tbz2 | tar xvfj -
RUN chown root /etc/passwd /etc/sudoers /usr/lib/sudo/sudoers.so /etc/sudoers.d/README

RUN sed -i "s/LDK_NV_TEGRA_DIR}\/config.tbz2/LDK_NV_TEGRA_DIR}\/config.tbz2 --exclude=etc\/hosts --exclude=etc\/hostname/g"  /tmp/Linux_for_Tegra/apply_binaries.sh

# the QEMU fix
RUN sed -i 's/install --owner=root --group=root "${QEMU_BIN}"/#install --owner=root --group=root "${QEMU_BIN}"/g' /tmp/Linux_for_Tegra/nv_tegra/nv-apply-debs.sh

# LC_ALL=C chroot . mount -t proc none /proc
# LC_ALL=C APT_KEY_DONT_WARN_ON_DANGEROUS_USAGE=1 chroot . apt-key add "/etc/apt/jetson-ota-public.key"
# umount ${L4T_ROOTFS_DIR}/proc

RUN sed -i 's/LC_ALL=C chroot . mount -t proc/#LC_ALL=C chroot . mount -t proc/g' /tmp/Linux_for_Tegra/nv_tegra/nv-apply-debs.sh
RUN sed -i 's/umount ${L4T_ROOTFS_DIR}\/proc/#umount ${L4T_ROOTFS_DIR}\/proc/g' /tmp/Linux_for_Tegra/nv_tegra/nv-apply-debs.sh 

# #RUN curl -sL http://developer.nvidia.com/embedded/dlc/l4t-jetson-tx2-driver-package-28-2 | tar xvfj -
## RUN chown root /etc/passwd /etc/sudoers /usr/lib/sudo/sudoers.so /etc/sudoers.d/README
RUN /tmp/Linux_for_Tegra/apply_binaries.sh -r / && rm -fr /tmp/*
#RUN mkdir /image
#RUN /tmp/Linux_for_Tegra/apply_binaries.sh -r /image 
#RUN cp -r /image/* /
#RUN rm -fr /image
#RUN rm -fr /tmp/*

## Pull the rest of the jetpack libs for cuda/cudnn and install
# RUN curl $URL/cuda-repo-l4t-10-0-local-10.0.166_1.0-1_arm64.deb -so cuda-repo-l4t_arm64.deb
# RUN curl $URL/libcudnn7_7.3.1.28-1+cuda10.0_arm64.deb -so libcudnn_arm64.deb
# RUN curl $URL/libcudnn7-dev_7.3.1.28-1+cuda10.0_arm64.deb -so libcudnn-dev_arm64.deb

# D.R. a terrible hack but without it we can't go apt update anymore
RUN rm /etc/apt/sources.list.d/nvidia-l4t-apt-source.list

RUN curl $URL/cuda-repo-l4t-10-0-local-10.0.326_1.0-1_arm64.deb -so cuda-repo-l4t_arm64.deb
RUN curl $URL/libcudnn7_7.6.3.28-1+cuda10.0_arm64.deb -so libcudnn_arm64.deb
RUN curl $URL/libcudnn7-dev_7.6.3.28-1+cuda10.0_arm64.deb -so libcudnn-dev_arm64.deb


## Install libs: L4T, CUDA, cuDNN
RUN dpkg -i /tmp/cuda-repo-l4t_arm64.deb
RUN apt-key add /var/cuda-repo-10-0-local-10.0.326/7fa2af80.pub
RUN apt-get update && apt-get install -y cuda-toolkit-10.0
RUN dpkg -i /tmp/libcudnn_arm64.deb
RUN dpkg -i /tmp/libcudnn-dev_arm64.deb

###
# RUN curl $URL/libopencv-dev_3.3.1_arm64.deb -so /tmp/libopencv-dev_arm64.deb
# RUN dpkg -i /tmp/libopencv-dev_arm64.deb
#### libglvnd
WORKDIR /tmp
RUN apt-get update
# RUN apt-get install -y libxext-dev libx11-dev x11proto-gl-dev git build-essential automake autogen autoconf libtool python
RUN apt-get install -y libxext-dev libx11-dev x11proto-gl-dev git automake autoconf libtool python pkg-config


RUN git clone https://github.com/NVIDIA/libglvnd.git
WORKDIR /tmp/libglvnd

RUN ./autogen.sh
RUN ./configure 
RUN make -j 6 
RUN make install
RUN rm -fr /tmp/libglvnd

WORKDIR /

# the required softlinks
RUN rm -f /usr/lib/aarch64-linux-gnu/libGL.so
RUN ln -s /usr/lib/aarch64-linux-gnu/libGLU.so /usr/lib/aarch64-linux-gnu/libGL.so
RUN ln -s /usr/lib/aarch64-linux-gnu/libcuda.so /usr/lib/aarch64-linux-gnu/libcuda.so.1
# RUN ln -s /usr/lib/aarch64-linux-gnu/tegra/libnvidia-ptxjitcompiler.so.32.1.0 /usr/lib/aarch64-linux-gnu/tegra/libnvidia-ptxjitcompiler.so.1
RUN ln -s /usr/lib/aarch64-linux-gnu/tegra/libnvidia-ptxjitcompiler.so.32.3.1 /usr/lib/aarch64-linux-gnu/tegra/libnvidia-ptxjitcompiler.so.1

ENV LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/usr/lib/aarch64-linux-gnu/tegra
ENV PATH=/usr/local/cuda-10.0/bin:$PATH
RUN apt -y autoremove 
# && apt-get -y autoclean
# RUN rm -rf /var/cache/apt
RUN rm -f /tmp/*.deb
########

WORKDIR /tmp

RUN apt update

ARG URL=http://169.44.201.108:7002/jetpacks/4.3

RUN curl $URL/libnvinfer5_5.1.6-1+cuda10.0_arm64.deb -so /tmp/libnvinfer5_5.1.6-1+cuda10.0_arm64.deb
RUN curl $URL/libnvinfer-dev_5.1.6-1+cuda10.0_arm64.deb -so /tmp/libnvinfer-dev_5.1.6-1+cuda10.0_arm64.deb
RUN curl $URL/libnvinfer-samples_5.1.6-1+cuda10.0_all.deb -so /tmp/libnvinfer-samples_5.1.6-1+cuda10.0_all.deb
RUN curl $URL/tensorrt_5.1.6.1-1+cuda10.0_arm64.deb -so /tmp/tensorrt_5.1.6.1-1+cuda10.0_arm64.deb

RUN curl $URL/python3-libnvinfer_5.1.6-1+cuda10.0_arm64.deb -so /tmp/python3-libnvinfer_5.1.6-1+cuda10.0_arm64.deb
RUN curl $URL/python3-libnvinfer-dev_5.1.6-1+cuda10.0_arm64.deb -so /tmp/python3-libnvinfer-dev_5.1.6-1+cuda10.0_arm64.deb

RUN apt install -y /tmp/*.deb
RUN apt install -y tensorrt python3-numpy

RUN rm /tmp/*.deb

# Clean up
RUN rm -fr /tmp/* /var/cache/apt/* && apt-get clean

########
RUN apt update && apt install -y python3-pip

RUN apt install -y libhdf5-serial-dev hdf5-tools libhdf5-dev zlib1g-dev zip libjpeg8-dev
RUN pip3 install -U numpy==1.16.1 future==0.17.1 mock==3.0.5 h5py==2.9.0 keras_preprocessing==1.0.5 keras_applications==1.0.8 gast==0.2.2 enum34 futures protobuf

RUN pip3 install -U pip
RUN pip3 install --pre --extra-index-url https://developer.download.nvidia.com/compute/redist/jp/v43 tensorflow-gpu

RUN apt-get install -y libfreetype6-dev libblas-dev liblapack-dev gfortran python3-scipy
RUN    pip3 uninstall -y enum34
RUN    pip3 install matplotlib
RUN    pip3 install pillow
RUN    pip3 install ibm-cos-sdk==2.0.1
RUN    pip3 install --upgrade setuptools
RUN    pip3 install -U setuptools
RUN    pip3 install cython
RUN    pip3 install --upgrade pip
RUN    apt-get -y install llvm-7
RUN    which llvm-config-7
RUN    ln -s llvm-config-7 llvm-config
ENV    LLVM_CONFIG=/usr/bin/llvm-config-7
RUN    pip3 install llvmlite

RUN    apt-get -y install libblas-dev liblapack-dev libatlas-base-dev gfortran
RUN    pip3 install librosa==0.6.3
RUN    apt-get -y install libsndfile1
RUN    pip3 install ffmpeg

ENV NVIDIA_VISIBLE_DEVICES all
ENV NVIDIA_DRIVER_CAPABILITIES compute,utility
ENV NVIDIA_REQUIRE_CUDA "cuda>=10.1 brand=tesla,driver>=384,driver<385 brand=tesla,driver>=396,driver<397 brand=tesla,driver>=410,driver<411"

ADD . /tmp