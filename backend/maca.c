#include <mcr/mc_runtime.h>
#define PY_SSIZE_T_CLEAN
#include <Python.h>
#include <stdio.h>
#include <stdlib.h>
static inline void gpuAssert(mcError_t code, const char *file, int line)
{
  if (code != mcSuccess)
  {
     const char* prefix = "Triton Error [MACA]: ";
     const char* str = mcGetErrorString(code);
     char err[1024] = {0};
     strcat(err, prefix);
     strcat(err, str);
     PyErr_SetString(PyExc_RuntimeError, err);
  }
}

#define MACA_CHECK(ans) { gpuAssert((ans), __FILE__, __LINE__); if(PyErr_Occurred()) return NULL; }

static PyObject* getDeviceProperties(PyObject* self, PyObject* args){
    int device_id;
    if(!PyArg_ParseTuple(args, "i", &device_id))
        return NULL;
    // Get device handle
    MCdevice device;
    mcDeviceGet(&device, device_id);

    // create a struct to hold device properties
    int max_shared_mem;
    int multiprocessor_count;
    int sm_clock_rate;
    int mem_clock_rate;
    int mem_bus_width;
    MACA_CHECK(mcDeviceGetAttribute(&max_shared_mem, mcDeviceAttributeMaxSharedMemoryPerBlock, device));
    MACA_CHECK(mcDeviceGetAttribute(&multiprocessor_count, mcDeviceAttributeMultiProcessorCount, device));
    MACA_CHECK(mcDeviceGetAttribute(&sm_clock_rate, mcDeviceAttributeClockRate, device));
    MACA_CHECK(mcDeviceGetAttribute(&mem_clock_rate, mcDeviceAttributeMemoryClockRate, device));
    MACA_CHECK(mcDeviceGetAttribute(&mem_bus_width, mcDeviceAttributeMemoryBusWidth, device));


    return Py_BuildValue("{s:i, s:i, s:i, s:i, s:i}", "max_shared_mem", max_shared_mem,
                               "multiprocessor_count", multiprocessor_count,
                               "sm_clock_rate", sm_clock_rate,
                               "mem_clock_rate", mem_clock_rate,
                               "mem_bus_width", mem_bus_width);
}

static PyObject* loadBinary(PyObject* self, PyObject* args) {
    const char* name;
    const char* data;
    Py_ssize_t data_size;
    int shared;
    int device;
    if(!PyArg_ParseTuple(args, "ss#ii", &name, &data, &data_size, &shared, &device)) {
        return NULL;
    }
    mcFunction_t fun;
    mcModule_t mod;
    // create driver handles
    MACA_CHECK(mcModuleLoadData(&mod, data));
    MACA_CHECK(mcModuleGetFunction(&fun, mod, name));

    // get allocated registers and spilled registers from the function
    int n_regs = 0;
    int n_spills = 0;

    if(PyErr_Occurred()) {
      return NULL;
    }
    return Py_BuildValue("(KKii)", (uint64_t)mod, (uint64_t)fun, n_regs, n_spills);
}

static PyMethodDef ModuleMethods[] = {
  {"load_binary", loadBinary, METH_VARARGS, "Load provided mcfatbin into MACA driver"},
  {"get_device_properties", getDeviceProperties, METH_VARARGS, "Get the properties for a given device"},
  {NULL, NULL, 0, NULL} // sentinel
};

static struct PyModuleDef ModuleDef = {
  PyModuleDef_HEAD_INIT,
  "maca_utils",
  NULL, //documentation
  -1, //size
  ModuleMethods
};

PyMODINIT_FUNC PyInit_maca_utils(void) {
  PyObject *m = PyModule_Create(&ModuleDef);
  if(m == NULL) {
    return NULL;
  }
  PyModule_AddFunctions(m, ModuleMethods);
  return m;
}
