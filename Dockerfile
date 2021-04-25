FROM nvcr.io/nvidia/pytorch:20.03-py3
WORKDIR /
RUN apt-get update

RUN apt-get install python3 -y
RUN apt-get install python3-pip -y
RUN apt-get install python3-dev -y
RUN apt-get install libncurses5 libncurses5-dbg libncurses5-dev libncursesw5 build-essential gcc gcc-multilib g++ -y
RUN apt-get install wget -y
RUN python3 -m pip install --upgrade pip
RUN python3 -m pip install torch==1.7.1 torchvision==0.8.2 torchaudio==0.7.2
RUN apt-get install git -y
ENV DEBIAN_FRONTEND noninteractive

RUN git clone https://github.com/raikarsagar/tacotron2_waveglow --branch training_taco2 --single-branch --depth 1 /tacotron2

WORKDIR /tacotron2
#RUN python3 -m pip install llvmlite --ignore-installed
RUN python3 -m pip install -r requirements.txt
RUN mkdir /tacotron2/mount
RUN sed -i -- 's,DUMMY,/tacotron2/mount/LJSpeech-1.1/wavs,g' filelists/*.txt
