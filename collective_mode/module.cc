// Include the defined classes that are to be exported to python
#include "CollectiveMode.h"
#include "CollectiveModeGPU.h"

#include <hoomd/extern/pybind/include/pybind11/pybind11.h>

// specify the python module. Note that the name must expliclty match the PROJECT() name provided in CMakeLists
// (with an underscore in front)
PYBIND11_MODULE(_collective_mode, m)
    {
    export_CollectiveMode(m);

    #ifdef ENABLE_CUDA
    export_CollectiveModeGPU(m);
    #endif
    }
