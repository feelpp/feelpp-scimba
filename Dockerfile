
# Start with the Feel++ base image
FROM ghcr.io/feelpp/feelpp:jammy

# Set labels for metadata
LABEL maintainer="Rayen Tlili <rayen.tlili@etu.unistra.fr>"
LABEL description="Docker image with Feel++ & ScimBa"

USER root

# Install system dependencies
RUN apt-get update && apt-get install -y \
    git \
   xvfb

# Install Python libraries
RUN pip3 install torch xvfbwrapper pyvista plotly panel ipykernel matplotlib tabulate nbformat gmsh


# Clone the Scimba repository
RUN git clone https://gitlab.inria.fr/scimba/scimba.git /workspaces/2024-stage-feelpp-scimba

# Install Scimba and its dependencies
WORKDIR /workspaces/2024-stage-feelpp-scimba/scimba
RUN pip3 install scimba

# Copy the xvfb script into the container for visualization
COPY tools/load_xvfb.sh /usr/local/bin/load_xvfb.sh
RUN chmod +x /usr/local/bin/load_xvfb.sh

# Set the script to initialize the environment
CMD ["/usr/local/bin/load_xvfb.sh"]