// Implements various Linux Functionality as a Python Module

// Import the Module Components
const clone = @import("clone.zig");

// Python BoilerPlate
// Import the Python C Interface
const py = @cImport({
  // Use the Stable ABI; For more information see: https://docs.python.org/3/c-api/stable.html
  @cDefine("Py_LIMITED_API", "0x030A2000"); // Minimum Python Python Version of 3.12
  // Include all functions, type & macro definitions necessary for the Python/C API; For more information see: https://docs.python.org/3.12/c-api/intro.html#include-files
  @cDefine("PY_SSIZE_T_CLEAN", {});
  @cInclude("Python.h");
});

// Define Method Definitions (PyMethodDef)

// Define Module Definitions (PyModuleDef)

// Define Module Initialization Function (PyInit_*)
