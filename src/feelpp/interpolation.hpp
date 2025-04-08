#ifndef INTERPOLATION_HPP
#define INTERPOLATION_HPP

#include <vector>
#include "mesh.hpp"


class Interpolator {
public:
    Interpolator(const Mesh& mesh, const std::vector<double>& values);
    std::vector<double> interpolate();
private:
    Mesh mesh;
    std::vector<double> values;
};

#endif // INTERPOLATION_HPP
