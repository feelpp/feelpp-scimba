#ifndef PDE_SOLVER_HPP
#define PDE_SOLVER_HPP

#include "mesh.hpp"
#include <vector>

// PdeSolver class encapsulates the FEM process:
// - Setting up the domain (mesh)
// - Assembling the system (placeholder in this example)
// - Solving the linear system (returns a dummy solution)
// - Computing the L2 error between two solution vectors.
class PdeSolver {
public:
    PdeSolver();
    void setupDomain();
    void assembleSystem(); // In a full implementation, assemble the stiffness matrix and load vector.
    std::vector<double> solve(); // Solve the linear system and return the solution.
    double computeError(const std::vector<double>& sol1, const std::vector<double>& sol2); // Compute L2 error.
    const Mesh& getMesh() const;
private:
    Mesh mesh;
};

#endif // PDE_SOLVER_HPP
