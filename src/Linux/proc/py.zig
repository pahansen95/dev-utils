// Implements various Linux Functionality as a Python Module

// Import the Python C Interface
const py = @cImport({
  // Use the Stable ABI; For more information see: https://docs.python.org/3/c-api/stable.html
  @cDefine("Py_LIMITED_API", "0x03A20000"); // Minimum Python Python Version of 3.12
  // Include all functions, type & macro definitions necessary for the Python/C API; For more information see: https://docs.python.org/3.12/c-api/intro.html#include-files
  @cDefine("PY_SSIZE_T_CLEAN", {});
  // @cInclude("Python.h");
  @cInclude("/home/dev/.pyenv/versions/3.12.4/include/python3.12/Python.h"); // TODO: Swap this out for the File above when ready to compile
});

// Import the Module Components
const clone = @import("src/clone.zig");

// The Python Methods
fn no_op(self: [*c]py.PyObject, args: [*c]py.PyObject) callconv(.C) [*c]py.PyObject {
  _ = self;
  _ = args;
  return py.Py_None();
}

//
// Python BoilerPlate
//
// See the following for more information
// 
// - Python Docs: https://docs.python.org/3.12/extending/extending.html#the-module-s-method-table-and-initialization-function
// - Zig Example: https://github.com/adamserafini/zaml/blob/eef4558e6c66037e2006f185bb6e3b322e0ac7fb/zamlmodule.zig
//

// Define Method Definitions (PyMethodDef)
var ProcMethods = [_]py.PyMethodDef {
  py.PyMethodDef{
    .ml_name = "no_op",
    .ml_meth = no_op,
    .ml_flags = py.METH_NOARGS,
    .ml_doc = null, // TODO: Add docs
  },
  py.PyMethodDef{
    .ml_name = null,
    .ml_meth = null,
    .ml_flags = 0,
    .ml_doc = null,
  }
};

// Define Module Definitions (PyModuleDef); not sure what everything does yet; based on https://github.com/adamserafini/zaml/blob/eef4558e6c66037e2006f185bb6e3b322e0ac7fb/zamlmodule.zig#L36-L54
var procmodule = py.PyModuleDef{
  .m_base = py.PyModuleDef_Base{
    .ob_base = py.PyObject{
      .unnamed_0 = .{ .ob_refcnt = 1, },
      .ob_type = null,
    },
    .m_init = null,
    .m_index = 0,
    .m_copy = null,
  },
  .m_name = "proc",
  .m_doc = null,
  .m_size = -1,
  .m_slots = null,
  .m_traverse = null,
  .m_clear = null,
  .m_free = null,
  .m_methods = &ProcMethods,
};

// Define Module Initialization Function (PyInit_*)

pub export fn PyInit_proc() *py.PyObject {
  return py.PyModule_Create(&procmodule);
}
