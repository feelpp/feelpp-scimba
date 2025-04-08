#include <iostream>
#include <cmath> // For sin, M_PI

#include "pde_solver.hpp"
#include "interpolation.hpp"

int main(int argc, char* argv[]) {
    // 1. 
    PdeSolver solver;
    solver.setupDomain();
    solver.assembleSystem();

    // 2. 
    std::vector<double> pinnData(solver.getMesh().getNumberOfNodes(), 0.0);
    const auto& nodes = solver.getMesh().getNodes();
    for (size_t i = 0; i < nodes.size(); ++i) {
        double x = nodes[i].first;
        double y = nodes[i].second;
        pinnData[i] = std::sin(M_PI * x) * std::sin(M_PI * y);
    }

    // 3. 
    Interpolator interpolator(solver.getMesh(), pinnData);
    std::vector<double> interpolatedSolution = interpolator.interpolate();

    // 4. 
    std::vector<double> femSolution = solver.solve();

    // 5. 
    double error = solver.computeError(femSolution, interpolatedSolution);
    std::cout << "L2 error between FEM and PINN interpolation: " << error << std::endl;

    return 0;
}
