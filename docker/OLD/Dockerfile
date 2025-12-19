# Base image: CUDA 12.2 runtime on Ubuntu 22.04
FROM nvidia/cuda:12.2.2-runtime-ubuntu22.04

# Avoid interactive prompts during package installation
ENV DEBIAN_FRONTEND=noninteractive

# Install basic system tools
# (tini = proper init process inside the container)
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
        wget \
        git \
        ca-certificates \
        bash \
        tini && \
    rm -rf /var/lib/apt/lists/*

# Create non-root user "monkey"
ARG USERNAME=monkey
ARG UID=1000
ARG GID=1000

RUN groupadd -g ${GID} ${USERNAME} && \
    useradd -m -u ${UID} -g ${GID} -s /bin/bash ${USERNAME}

# Set HOME for the monkey user
ENV HOME=/home/${USERNAME}

# Install Miniforge (conda + mamba, conda-forge based) to /opt/conda
ENV CONDA_DIR=/opt/conda

# We copy the installer from the build context instead of downloading it inside the container
COPY docker/Miniforge3-Linux-x86_64.sh /tmp/miniforge.sh

RUN bash /tmp/miniforge.sh -b -p ${CONDA_DIR} && \
    rm /tmp/miniforge.sh

# Make sure conda (and mamba) are on PATH for all subsequent commands
ENV PATH=${CONDA_DIR}/bin:${PATH}

# Optional: enforce strict channel priority (Miniforge already defaults to conda-forge)
RUN conda config --set channel_priority strict || true

# Project env directory inside container (should be backed by a mount)
# On the cluster, you will mount some persistent host path here (e.g. /scratch/$USER/MRSI_simulation_env)
ENV PROJECT_ENV_DIR=/opt/envs/MRSI_simulation

# Location of the environment YAML inside the mounted repo
# In the container, you will mount the repo to /workspace
# and the file should be at /workspace/env/MRSI_simulation.yml
ENV PROJECT_ENV_FILE=/workspace/env/MRSI_simulation.yml

# Use bash as default shell for RUN commands
SHELL ["bash", "-lc"]

# Create helper start script to activate the environment if it exists
RUN mkdir -p /usr/local/bin && \
    cat << 'EOF' > /usr/local/bin/start.sh
#!/bin/bash
set -e

# Load conda
source /opt/conda/etc/profile.d/conda.sh

if [ -d "$PROJECT_ENV_DIR" ]; then
    echo "Activating conda env at: $PROJECT_ENV_DIR"
    conda activate "$PROJECT_ENV_DIR" || export PATH="$PROJECT_ENV_DIR/bin:$PATH"
else
    echo "Project env directory $PROJECT_ENV_DIR does not exist yet."
    echo "You can create it inside the container with:"
    echo "  mamba env create -p $PROJECT_ENV_DIR -f $PROJECT_ENV_FILE"
fi

exec "$@"
EOF

# Make the start script executable
RUN chmod +x /usr/local/bin/start.sh

# Set working directory inside the container
WORKDIR /workspace

# Make sure "monkey" owns the working directory
RUN chown -R ${USERNAME}:${USERNAME} /workspace

# Switch to non-root user
USER ${USERNAME}

# Use tini as entrypoint for proper signal handling, then our start script
ENTRYPOINT ["/usr/bin/tini", "--", "/usr/local/bin/start.sh"]

# Default command: start an interactive shell
CMD ["bash"]
