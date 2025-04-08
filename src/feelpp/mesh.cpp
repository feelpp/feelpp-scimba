#include "mesh.hpp"
#include <cstddef>

Mesh Mesh::createUnitSquareMesh(int nx, int ny) {
    Mesh mesh;
    for (int i = 0; i < nx; ++i) {
        double x = static_cast<double>(i) / (nx - 1);
        for (int j = 0; j < ny; ++j) {
            double y = static_cast<double>(j) / (ny - 1);
            mesh.nodes.push_back(std::make_pair(x, y));
        }
    }
    return mesh;
}

const std::vector<std::pair<double, double>>& Mesh::getNodes() const {
    return nodes;
}

size_t Mesh::getNumberOfNodes() const {
    return nodes.size();
}
