#ifndef MESH_HPP
#define MESH_HPP

#include <vector>
#include <utility>
#include <cstddef> // For size_t

// A simple Mesh class that generates a unit square grid.
class Mesh {
public:
    // Generate a mesh on the unit square [0,1]x[0,1] with nx nodes in x-direction and ny nodes in y-direction.
    static Mesh createUnitSquareMesh(int nx = 10, int ny = 10);

    const std::vector<std::pair<double, double>>& getNodes() const;

    size_t getNumberOfNodes() const;
private:
    std::vector<std::pair<double, double>> nodes;
};

#endif // MESH_HPP
