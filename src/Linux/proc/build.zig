const std = @import("std");

// Create a Build Spec that A) creates a shared library; B) targets the `py.zig` file

pub fn build(b: *std.Build) void {
  const target = b.standardTargetOptions(.{});
  const optimize = b.standardOptimizeOption(.{});
  const py_lib = b.addSharedLibrary(.{
    .name = "PyModuleProc",
    .root_source_file = b.path("py.zig"),
    // .root_source_file = b.path("src/clone.zig"),
    .target = target,
    .optimize = optimize,
  });
  py_lib.linkLibC();
  b.installArtifact(py_lib);
}
