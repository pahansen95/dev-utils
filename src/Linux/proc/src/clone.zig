// Provides Clone Functionality

const std = @import("std");
const linux = std.os.linux;

const OSError = error {
  Generic,
};
const pid_t = linux.pid_t;

// As defined in `man 2 clone3`
// Otherwise See Kernel Source: https://github.com/torvalds/linux/blob/8883957b3c9de2087fb6cf9691c1188cccf1ac9c/include/uapi/linux/sched.h#L92-L104
// struct clone_args {
//     u64 flags;        /* Flags bit mask */
//     u64 pidfd;        /* Where to store PID file descriptor
//                         (pid_t *) */
//     u64 child_tid;    /* Where to store child TID,
//                         in child's memory (pid_t *) */
//     u64 parent_tid;   /* Where to store child TID,
//                         in parent's memory (int *) */
//     u64 exit_signal;  /* Signal to deliver to parent on
//                         child termination */
//     u64 stack;        /* Pointer to lowest byte of stack */
//     u64 stack_size;   /* Size of stack */
//     u64 tls;          /* Location of new TLS */
//     u64 set_tid;      /* Pointer to a pid_t array
//                         (since Linux 5.5) */
//     u64 set_tid_size; /* Number of elements in set_tid
//                         (since Linux 5.5) */
//     u64 cgroup;       /* File descriptor for target cgroup
//                         of child (since Linux 5.7) */
// };
const c_clone3_args_t = struct {
  flags: u64,
  pidfd: u64,
  child_tid: u64,
  parent_tid: u64,
  exit_signal: u64,
  stack: u64,
  stack_size: u64,
  tls: u64,
  set_tid: u64,
  set_tid_size: u64,
  cgroup: u64,
};

pub const NewNS = enum(comptime_int) {
  CGROUP = linux.CLONE.NEWCGROUP,
  IPC = linux.CLONE.NEWIPC,
  NET = linux.CLONE.NEWNET,
  MNT = linux.CLONE.NEWNS,
  PID = linux.CLONE.NEWPID,
  TIME = linux.CLONE.NEWTIME,
  USER = linux.CLONE.NEWUSER,
  UTS = linux.CLONE.NEWUTS,
};

pub fn cloneProcess(
  ns: NewNS,
) OSError!pid_t {
  const clone3_args = _: {
    var clone3_args = c_clone3_args_t { .zeros }; // Zero out the Struct
    clone3_args.exit_signal = @intCast(linux.SIG.CHLD); // This is the default expected; don't change this.
    clone3_args.flags = @intCast(ns); // Let the user choose what Namespaces to create, if any.
    // TODO: Future Fields
    break :_ clone3_args; // Yield the now assembled struct
  };
  const rc = linux.syscall2(
    linux.SYS.clone3,
    @intFromPtr(&clone3_args),
    c_clone3_args_t.size(),
  );
  switch ( linux.E.init(rc) ){
    linux.E.SUCCESS => return @intCast(rc),
    // TODO: Make errors preceise
    else => return error.Generic,
  }
}
