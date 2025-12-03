#!/bin/bash

# pip cache purge
# apt-get clean
# apt update -y && apt-get install git -y
# conda activate base
# pip install matplotlib
rm -rf dist/ build/ triton.egg-info/

python setup_on_maca.py install -v

cd test
python vector_add.py 2>&1 | tee test.log 


# 创建容器
# docker run -itd \
#     --net=host \
#     --uts=host \
#     --ipc=host \
#     --device=/dev/dri \
#     --device=/dev/mxcd  \
#     --device=/dev/infiniband \
#     --privileged=true \
#     --group-add=video \
#     --name ${docker_name} \
#     --security-opt seccomp=unconfined \
#     --security-opt apparmor=unconfined \
#     --shm-size 160gb \
#     --ulimit memlock=-1 \
#     -v /datapool:/datapool  \
#     -w ${workspace_name} \
#     cr.metax-tech.com/public-ai-release/maca/sglang:maca.ai2.33.1.7-torch2.6-py310-ubuntu22.04-amd64
