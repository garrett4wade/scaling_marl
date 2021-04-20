# BASE IMAGE
FROM nvcr.io/nvidia/pytorch:21.02-py3

WORKDIR /workspace
SHELL ["/bin/bash","-c"]

# Instal basic utilities
RUN apt-get update && \
    apt-get install -y --no-install-recommends git htop screen && \
    #  wget unzip bzip2 build-essential ca-certificates && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

RUN pip config set global.index-url https://pypi.tuna.tsinghua.edu.cn/simple && \
    pip install pip -U && \
    pip install gym pysc2 absl-py wandb tensorboardx cffi && \
    pip install git+https://github.com/oxwhirl/smac.git && \
    rm -rf $(pip cache dir)

# starcraft2 environment
COPY ./StarCraftII ./StarCraftII
ENV SC2PATH="/workspace/StarCraftII"

CMD [ "/bin/bash" ]
