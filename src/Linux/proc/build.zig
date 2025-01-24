const std = @import("std");

// Create a Build Spec that A) creates a shared library; B) targets the `py.zig` file

pub fn build(b: *std.Build) void {
  const target = b.standardTargetOptions(.{});
  const optimize = b.standardOptimizeOption(.{});
  const so = b.addSharedLibrary(.{
    .name = "proc",
    .root_module = .{
      .owner = b,
      .root_source_file = b.path("proc.zig"),
      .resolved_target = target,
      .optimize = optimize,
    },
  });
  b.installArtifact(so);
}
