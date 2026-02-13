# syntax=docker/dockerfile:1

FROM mcr.microsoft.com/devcontainers/python:3.14-trixie

ARG DEVCONTAINER_WORKSPACE_NAME
ARG ACT_VERSION=v0.2.69

# Install system dependencies as root
RUN apt-get update \
 && apt-get install -y --no-install-recommends \
    build-essential \
    ca-certificates \
    cmake \
    curl \
    gdb \
    git \
    jq \
    ninja-build \
    pkg-config \
    python3-dev \
    docker.io \
 && rm -rf /var/lib/apt/lists/*

# Install uv, just, and act globally so they are available without per-user bootstrap
RUN set -eux; \
    curl -LsSf https://astral.sh/uv/install.sh | sh; \
    install -Dm755 /root/.local/bin/uv /usr/local/bin/uv; \
    curl --proto "=https" --tlsv1.2 -sSf https://just.systems/install.sh | bash -s -- --to /usr/local/bin; \
    curl -sSL https://raw.githubusercontent.com/nektos/act/master/install.sh | bash -s -- -b /usr/local/bin "${ACT_VERSION}"

# Ensure correct PATH for vscode user (retain user-local bins for compatibility)
ENV PATH=/home/vscode/.local/bin:/home/vscode/.just/bin:$PATH

# Switch to non-root user (devcontainers base image already has vscode)
USER vscode

# Set workdir explicitly (devcontainers usually do this, but be explicit)
WORKDIR /workspaces/${DEVCONTAINER_WORKSPACE_NAME}
