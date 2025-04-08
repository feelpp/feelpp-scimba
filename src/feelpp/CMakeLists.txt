# List the source files for the FEM solver
set(FEELPP_SRC
    mesh.cpp
    pde_solver.cpp
    interpolation.cpp
)

# Build a static library "pde_solver" containing mesh and solver code
add_library(pde_solver ${FEELPP_SRC})
target_include_directories(pde_solver PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})

# Build the FEM solver executable from main.cpp and link with pde_solver library
add_executable(feelpp_solver main.cpp)
target_link_libraries(feelpp_solver PRIVATE pde_solver)
