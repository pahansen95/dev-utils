# Service Manager

The Service Manager is a software that manages the lifecycle of other software services; a software service is any executing piece of software that provides unique functionality or supports other software.

Sets of software services packaged as Software Bundles & deployed in System Partitions; a System Partition is a piece of the Operating System logically isolated from the rest of the system. Software Bundles & System Partitions are akin to OCI Containers. A Single deployment of a Software Service is called a Service Instance.

The Service Manager is similar in functionality to Docker or Kubelet (Kubernetes per Node Daemon); it is a Daemon & CLI providing multiple common functionalities such as bundle management (fetch, upload, etc...), app deployments (up, down, etc...), runtime lifecycle (start, stop, reload, etc...) & process interactions (attach, exec, logs, etc...).

**Table of Contents**

[TOC]

## System Partition

In this section we will describe the general design & operation of a System Partition. To preface, we assume System Partitions maps to a Single Service Set which, at runtime, is composed of multiple Service Instances.

A Service Set may deploy exclusively into a single named system partition. Each named system partition has a fake-init process: the root of the process hierarchy in that partition's PID namespace. This init process is part of the Service Manager & is the operator of the contained service instances. The Service Manager triggers creation of a new system partition. First, the service manager spawns a child process, herein the partition operator, & establishes an IPC channel with it. At this point, the partition operator is considered to be in an init state. As part of this state, it initially creates all the expected Namespaces: On Linux these namespaces are Cgroup, IPC, Network, Mount, PID, Time, User & UTS. The Operator then drops into "standby" mode & informs the Service Manager.

Concurrently, at some point in time, the user will request a named Service Set instance be brought up. Upon receiving this request, the Service Manager will select an unclaimed system partition & wait for it to reach "standby" mode, at which point it will begin to instruct that system partition's operator. At this point, the system operator is in the "setup" state. The Service Set, containing one or more service bundles, is extracted & mounted into the System Partition's Mount Namespace. The service operator then finalizes this root filesystem such as injecting runtime mounts (ie. proc, dev, etc...). Once the root filesystem is ready, the service operator then remounts the root filesystem as the root partition of the System Partition.

The system operator then spawns the processes for each service instance as defined by the Service Set. At this point, the system operator enters the "running" state.

Concurrently, at some point in time, the user will request a named Service Set Instance be brought down. The Service Manager will then begin to instruct the Service Operator what to do; at this point, the system operator enters the "teardown" state. Teardown consists of terminating all running services. Once all service processes are dead, the operator unmounts the root filesystem of the mount namespace & unwinds any other runtime configurations it mades. At this point the operator enters the "finalized" state. At this point, the Service Manager may instruct the operator to terminate, thereby releasing & cleaning up any system resources, or it may instruct the operator to redeploy the same named service instance. If reused, the operator will transition back to its "standby" state & the cycle repeats. It is noted that system partitions are resources with single, non-transferable ownership; once a system partition is claimed, it can only be re-used by that owner or destroyed. Ownership is further tied to a unique combination of requesting user & service set. Simply put, re-deployment a different service set or a new user triggers creation of a new system partition. Reuse of a system partition only occurs in a "redeploy" scenario.

## Service Sets

In this section we'll describe the general design & lifecycle of a Service Set.

A Service Set describes a collection of service instances to be deployed into a System Partition. A Service Set is paired with a Service Bundle which contains all the necessary filesystem data to run instances of the declared services.

The most basic form of a service set is a single instance deployed from a Single Service wherein the bundle contains 2 layers: the base root filesystem data & the service data. Service Sets may describe multiple service instances & bundle many seperate layers as required by the user's needs. In both scenarios however, the design of the Service Set is intended to be simple & straightforward. Highly complex scenarios (ex. inter-service set dependencies) are not handled; instead the user should fork & extend the reference implementation of the Service Manager and/or Service Operator to accommodate exceptions to the standard case.

As discussed in [System Partition](#system-partition), each system partition has an Operator that serves as the fake-init for that partition. This operator is what orchestrates the deployment & lifecycle of Service Set instance. The process starts with the Operator in "standby" mode, having received instructions to commence deployment of a service set.

The Operator expects certain preconditions to exist at this stage:

- The current root filesystem is minimal; including only a temporary filesystem & a runtime directory for the operator's use only.
- The runtime directory already contains:
  - The Extracted Service Bundle's data, including all layers as subfolders & the Bundle Metadata as a JSON File. The operator assumes the Service Manager has already verified data integrity of the extracted bundle.

The Operator then begins listening for deployment instructions:

- RootFS assembly instructions. This would include rootfs overlay assembly, system runtime mounts (ie. proc, dev, etc...), internal bind mounts or verification of externally injected mounts.
- Service Process Runtime Specifications which can define processes to spawn, cgroup configurations, runtime folders to create or ordering of services (ie. A Dependency DAG). No service processes are spawned at this time.
- Networking Runtime Configurations. This could include creation of network interfaces, hostname settings, IPAM commands, verification of runtime states, packet filtering rules (if supported by the bundle) and many other things.

Throughout this process, the Operator & Service Manager cooperatively communicate & react to the results of deployment efforts. Eventually, the Service Manager will inform the Operator once deployment instructions have concluded. At this point, the Service Operator commences spawning the service processes as previously agreed upon.

The Service Instance is responsible for the runtime state of each process. This includes:

- Start & Stop operations of individual processes.
- Restarting a process on exit (if configured to do so.)
- Aggregating Process Logs & Health Metrics.

The Service Instance also listens for lifecycle operational commands from the Service Manager such as service start/stop or exec commands. It likewise provides a supplementary server to the processes to assist with A) metadata requests, B) cooperative coordination between processes, C) logging, health and metrics reporting & D) other miscellaneous runtime requests such as privilege escalation or networking configurations.

Each spawned process is, by default, assigned its own Process Group & Process Session; this can be overridden during the deployment phase. Each Service Process may spawn their own children. Any children that would be orphaned are captured by the Service Operator; they may be immediately terminated if the original service was not configured to allow such behavior. No service process is allowed to terminate the fake-init process, which is the system operator; at most the operator receives a SIGHUP upon the death of a service process. All other coordination between Operator & Service must occur through the provided supplementary server. Service Logs can be directly sent to the Service Operator via the supplementary server. Alternatively, the stderr of the Service Process is captured by the service operator. The service operator splits the output of stderr by newline & treats each line as a message. These messages may either be serialized metric & log objects (ie. JSON), or they may be plain text. If plain text, lines are gathered & then regrouped into messages delimited by the Logging Prefix `LOG::[LEVEL]::`.
