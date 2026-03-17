FROM ubuntu:24.04

ENV DEBIAN_FRONTEND=noninteractive
COPY ./ubuntu.sources /etc/apt/sources.list.d/ubuntu.sources
COPY ./99-verify-peer.conf /etc/apt/apt.conf.d/99-verify-peer.conf
RUN apt-get update
RUN apt-get install -y \
  ca-certificates \
  openssh-server \
  unminimize
RUN rm -rf /etc/apt/apt.conf.d/99-verify-peer.conf
RUN apt-get update
RUN yes | unminimize
RUN apt-get update
RUN apt-get install -y \
  ibverbs-providers \
  libibverbs1 \
  librdmacm1 \
  nvme-cli \
  pciutils \
  rdma-core \
  sudo
RUN update-pciids
# Default prompt for login shells (user@host:path$)
RUN echo 'export PS1="\\u@\\h:\\w\\$ "' >> /etc/profile
