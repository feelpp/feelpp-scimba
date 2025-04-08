FROM ghcr.io/feelpp/feelpp:jammy

#RUN apt-get update && apt-get install -y python3-pip cmake build-essential

RUN pip install scimba plotly pybind11

WORKDIR /feel
COPY . .

CMD ["bash"]
