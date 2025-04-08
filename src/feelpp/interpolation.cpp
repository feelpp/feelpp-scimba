#include "interpolation.hpp"

Interpolator::Interpolator(const Mesh& mesh, const std::vector<double>& values)
    : mesh(mesh), values(values)
{}

std::vector<double> Interpolator::interpolate() {

    return values;
}
