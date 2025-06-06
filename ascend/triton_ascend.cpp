#define PY_SSIZE_T_CLEAN
#include <pybind11/pybind11.h>
namespace py = pybind11;

// register ascend passes to triton
void init_triton_ascend(py::module &&m) {
// void init_triton_dicp_triton(py::module &&m) {
  // currently no extra modules needed to plug-in libtriton.so
}