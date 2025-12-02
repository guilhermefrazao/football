FROM nvcr.io/nvidia/pytorch:22.12-py3

ENV DEBIAN_FRONTEND=noninteractive
RUN apt-get update && apt-get --no-install-recommends install -yq git cmake build-essential \
  libgl1-mesa-dev libsdl2-dev \
  libsdl2-image-dev libsdl2-ttf-dev libsdl2-gfx-dev libboost-all-dev \
  libdirectfb-dev libst-dev mesa-utils xvfb x11vnc \
  python3-pip

ENV LD_LIBRARY_PATH=/opt/hpcx/ompi/lib:/opt/hpcx/ucx/lib:${LD_LIBRARY_PATH}

# RUN python3 -m pip install --upgrade pip setuptools wheel
RUN python3 -m pip install psutil
RUN python3 -m pip install wheel==0.37.0 setuptools==57.5.0


COPY requirements.txt /tmp/requirements.txt
RUN python3 -m pip install -r /tmp/requirements.txt --no-build-isolation

COPY . /gfootball
RUN chmod +x /gfootball/gfootball/build_game_engine.sh
RUN cd /gfootball && python3 -m pip install .
RUN python3 -m pip install opencv-fixer
RUN python3 -c "from opencv_fixer import AutoFix; AutoFix()"
WORKDIR '/gfootball'

RUN chmod -R 777 /gfootball
