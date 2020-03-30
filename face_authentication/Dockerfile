FROM w251/tensorrt:dev-tx2-4.3_b132

ARG J_URL=http://169.44.201.108:7002/jetpacks/4.3

RUN apt update && apt install -y python3-pip
RUN apt-get update && apt install -y git pkg-config wget build-essential cmake unzip
RUN apt install -y curl

RUN curl $J_URL/libopencv_3.3.1-2-g31ccdfe11_arm64.deb  -so libopencv_3.3.1-2-g31ccdfe11_arm64.deb
RUN curl $J_URL/libopencv-dev_3.3.1-2-g31ccdfe11_arm64.deb -so libopencv-dev_3.3.1-2-g31ccdfe11_arm64.deb
RUN curl $J_URL/libopencv-python_3.3.1-2-g31ccdfe11_arm64.deb -so libopencv-python_3.3.1-2-g31ccdfe11_arm64.deb

RUN apt remove -y libopencv-calib3d-dev libopencv-core-dev 
RUN apt install -y  libtbb-dev libavcodec-dev libavformat-dev libgstreamer1.0-dev libgstreamer-plugins-base1.0-dev libgtk2.0-dev
RUN apt install -y libswscale-dev libv4l-dev

RUN dpkg -i *.deb

RUN apt install -y libhdf5-serial-dev hdf5-tools libhdf5-dev zlib1g-dev zip libjpeg8-dev  vim
RUN pip3 install --upgrade pip
RUN apt install -y python3-scipy
RUN pip3 install -U numpy==1.16.1 future==0.17.1 mock==3.0.5 h5py==2.9.0 keras_preprocessing==1.0.5 keras_applications==1.0.8 gast==0.2.2 enum34 futures protobuf 
#RUN pip3 install keras
RUN pip3 install keras_vggface 

# tf2
# RUN pip3 install --pre --extra-index-url https://developer.download.nvidia.com/compute/redist/jp/v43 tensorflow-gpu
RUN pip3 install -U pip
# if we want to go back to tensorflow 1
RUN pip3 install --pre --extra-index-url https://developer.download.nvidia.com/compute/redist/jp/v43 "tensorflow-gpu<2"

ADD data/ /tmp/data
ADD pretrained_models /tmp/pretrained_models
ADD face_identify_demo.py /tmp
ADD precompute_features.py /tmp
ADD precompute_features.pickle /tmp
ADD requirements.txt /tmp
