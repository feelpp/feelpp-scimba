#include "pde_solver.hpp"
#include "mesh.hpp"
#include <cmath>
#include <vector>

PdeSolver::PdeSolver() { }

void PdeSolver::setupDomain() {
    // Generate a mesh on [0,1]^2 with default 10x10 nodes.
    mesh = Mesh::createUnitSquareMesh(10, 10);
}

void PdeSolver::assembleSystem() {
    // In a real FEM solver, assemble the system matrix and right-hand side vector.
    // For this example, we do not assemble any system.
}

std::vector<double> PdeSolver::solve() {
    // For demonstration, return a vector of zeros (dummy FEM solution).
    return std::vector<double>(mesh.getNumberOfNodes(), 0.0);
}

double PdeSolver::computeError(const std::vector<double>& sol1, const std::vector<double>& sol2) {
    double sum = 0;
    for (size_t i = 0; i < sol1.size(); ++i) {
        double diff = sol1[i] - sol2[i];
        sum += diff * diff;
    }
    return std::sqrt(sum / sol1.size());
}

const Mesh& PdeSolver::getMesh() const {
    return mesh;
}
