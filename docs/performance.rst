.. Copyright (c) Advanced Micro Devices, Inc. All rights reserved.
.. SPDX-License-Identifier: MIT

NVMe-oF fabrics performance notes
=================================

This document captures fio throughput and latency from **2026-04-22** on the lab
**kernel nvmet** export (**g04u19**, **10.245.130.135**, RoCE ``traddr``).
Initiators **g04u07** (**10.245.130.131**) and **g04u13** (**10.245.130.133**)
share one subsystem namespace. Each host exposed ``/dev/nvme9n1``; ``nvme
list-subsys`` showed **RDMA** to the target. Enumerator indices can change after
reconnect; discover devices with ``nvme list-subsys -n <dev>`` before scripting.

**Caution:** concurrent raw I/O from several hosts on one namespace is unsafe
unless you coordinate with cluster file systems, reservations, or read-only
policy.

These tests were **read-only** random reads plus earlier **controlled** writes
to the first sector for connectivity smoke tests.

Common fio profile
------------------

Unless noted, all fio jobs used the following options (``fio`` 3.36) on the
hosts:

.. code-block:: text

   rw=randread
   bs=1M
   ioengine=io_uring
   direct=1
   runtime=60
   time_based=1
   numjobs=16
   thread
   group_reporting
   filename=/dev/nvme9n1

Each scenario below states ``iodepth`` and whether one initiator or both lab
hosts (**g04u07**, **g04u13**) ran fio concurrently for that summary row.

Throughput summary
------------------

.. list-table::
   :widths: 28 12 18 22
   :header-rows: 1

   * - Scenario
     - Host(s)
     - ``iodepth``
     - Read bandwidth (per host)
   * - Single initiator
     - g04u07 only
     - 1
     - **14.8 GiB/s** (15.9 GB/s), ~15.2k IOPS
   * - Dual initiators (parallel start)
     - g04u07 + g04u13
     - 1
     - **10.4 GiB/s** each (~20.8 GiB/s combined)
   * - Dual initiators (parallel start)
     - g04u07 + g04u13
     - 64
     - **13.2 GiB/s** + **13.5 GiB/s** (~26.6 GiB/s combined)

**Observations:**

* ``iodepth`` **64** raised aggregate reads to ~**26.6 GiB/s** versus ~**20.8
  GiB/s** at depth **1** while the target pool stayed busy (see *Target-side
  evidence* below).
* Dual-host **``iodepth=1``** left bandwidth on the table versus the single-host
  run because each host could not keep enough read I/O in-flight to fill the
  path.

Latency distributions (fio ``clat`` percentiles)
------------------------------------------------

fio reports **completion latency** (``clat``). For **``iodepth>1``**, ``clat``
includes **queueing** behind other outstanding I/O on the same job, so means and
tails grow even on a healthy fabric; compare scenarios on **shape**, not raw
microseconds alone.

Single initiator (g04u07), ``iodepth=1``, 16 threads
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

* Mean **``clat``**: ~**986 µs**; mean total **``lat``**: ~**1053 µs**.
* **``nvme9n1``** util ~**99.9%** for the run window.

.. code-block:: text

   clat percentiles (usec):
    |  1.00th=[  529],  5.00th=[  594], 10.00th=[  635], 20.00th=[  709],
    | 30.00th=[  775], 40.00th=[  848], 50.00th=[  930], 60.00th=[ 1012],
    | 70.00th=[ 1106], 80.00th=[ 1237], 90.00th=[ 1418], 95.00th=[ 1582],
    | 99.00th=[ 1893], 99.50th=[ 2008], 99.90th=[ 2311], 99.95th=[ 2638],
    | 99.99th=[ 4146]

Dual initiators, ``iodepth=1``, 16 threads each (started together)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

**g04u07** -- mean **``clat``** ~**1429 µs**, mean **``lat``** ~**1499 µs**:

.. code-block:: text

   clat percentiles (usec):
    |  1.00th=[  562],  5.00th=[  676], 10.00th=[  766], 20.00th=[  922],
    | 30.00th=[ 1057], 40.00th=[ 1188], 50.00th=[ 1336], 60.00th=[ 1483],
    | 70.00th=[ 1647], 80.00th=[ 1860], 90.00th=[ 2212], 95.00th=[ 2540],
    | 99.00th=[ 3261], 99.50th=[ 3523], 99.90th=[ 4113], 99.95th=[ 4359],
    | 99.99th=[ 4948]

**g04u13** -- mean **``clat``** ~**1421 µs**, mean **``lat``** ~**1502 µs**:

.. code-block:: text

   clat percentiles (usec):
    |  1.00th=[  578],  5.00th=[  685], 10.00th=[  775], 20.00th=[  922],
    | 30.00th=[ 1057], 40.00th=[ 1188], 50.00th=[ 1319], 60.00th=[ 1467],
    | 70.00th=[ 1631], 80.00th=[ 1860], 90.00th=[ 2212], 95.00th=[ 2507],
    | 99.00th=[ 3195], 99.50th=[ 3490], 99.90th=[ 4113], 99.95th=[ 4359],
    | 99.99th=[ 4948]

Dual initiators, ``iodepth=64``, 16 threads each (started together)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Depth **64**: fio **``clat``** in milliseconds; means ~**75 ms** on g04u07
and ~**74 ms** on g04u13 while thread queues kept target disks saturated.

**g04u07** (``clat`` percentiles in **msec**):

.. code-block:: text

   clat percentiles (msec):
    |  1.00th=[    6],  5.00th=[   24], 10.00th=[   33], 20.00th=[   45],
    | 30.00th=[   56], 40.00th=[   66], 50.00th=[   75], 60.00th=[   79],
    | 70.00th=[   83], 80.00th=[   89], 90.00th=[  108], 95.00th=[  148],
    | 99.00th=[  279], 99.50th=[  330], 99.90th=[  439], 99.95th=[  472],
    | 99.99th=[  514]

**g04u13** (``clat`` percentiles in **msec**):

.. code-block:: text

   clat percentiles (msec):
    |  1.00th=[    5],  5.00th=[   24], 10.00th=[   33], 20.00th=[   45],
    | 30.00th=[   56], 40.00th=[   66], 50.00th=[   75], 60.00th=[   79],
    | 70.00th=[   83], 80.00th=[   89], 90.00th=[  104], 95.00th=[  138],
    | 99.00th=[  259], 99.50th=[  309], 99.90th=[  418], 99.95th=[  451],
    | 99.99th=[  498]

**``lat`` buckets** (fio histogram summary) for the **``iodepth=64``** dual run
placed most mass between **2 ms** and **100 ms**, with a tail to **hundreds of
ms** at high percentiles -- consistent with **queueing** on a saturated backing
store, not a misconfigured link.

Target-side evidence (bottleneck)
---------------------------------

During **dual-host ``iodepth=64``** tests, **``iostat -xm 1``** on **g04u19**
showed **``dm-0``** (the **LVM** volume) plus **``nvme1n1``**, **``nvme2n1``**,
**``nvme4n1``**, and **``nvme7n1``** near **100% utilization** while sustaining
about **~27k MiB/s** combined read through **``dm-0``**. Four drives each sat
near **~6.8k MiB/s**, matching the **~26.6 GiB/s** aggregate from the two
initiators.

**Summary:** **1 MiB reads** exhausted **target-side NVMe** bandwidth on the
striped LVM stack before initiator CPU limits (modest user + system time).

Suggestions to increase performance
-------------------------------------

**Back-end media and layout**

* Add **more NVMe devices** or faster drives if four saturated Gen4 parts cannot
  reach your aggregate GB/s goals.
* Revisit **LVM layout** (segment size, stripe width, ``raid0`` vs ``linear``,
  alignment) so **1 MiB** random I/O maps cleanly onto physical devices.
* Align **nvmet** namespace **block size** with host expectations; check ``nvme
  id-ns``.

**Network path**

* Validate RoCE **MTU** end-to-end.
  **Jumbo frames** help large payloads when the path supports them.
* If telemetry shows **discards or pause storms**, tune **PFC/ECN** and buffers.
* Add **paths or target ports** when fabric counters show spare headroom versus
  disk.

**Software and tuning**

* Tune **``iodepth``**, **job count**, and **bs** to the real workload; replay
  production traces where possible.
* On the target, review **irq affinity**, **nvmet**, and kernel generation.
  Watch **softirq** when disks are not saturated but throughput is flat.
* For **multi-tenant** read-heavy designs, **separate namespaces** can cut
  head-of-line blocking versus one shared volume.

**Measurement hygiene**

* Collect initiator **fio**, target **iostat** / **blktrace**, and **NIC**
  counters in one aligned time window when claiming bottlenecks.
* Keep **``iodepth``** / **``numjobs``** fixed across latency comparisons.
* Treat **``iodepth>1``** ``clat`` as **queue-inclusive** (completion time
  includes queue wait).

Reproducing the dual-host run locally
---------------------------------------

From a workstation with SSH access, start both jobs together (``iodepth=64`` in
this example):

.. code-block:: bash

   FIO='sudo fio --name=nvmeof_randread --filename=/dev/nvme9n1 --rw=randread \
     --bs=1M --ioengine=io_uring --direct=1 --runtime=60 --time_based=1 \
     --numjobs=16 --thread --group_reporting --iodepth=64 --randrepeat=0 \
     --output-format=normal'
   ssh stebates@10.245.130.131 "$FIO" > /tmp/fio-g04u07.txt 2>&1 &
   ssh stebates@10.245.130.133 "$FIO" > /tmp/fio-g04u13.txt 2>&1 &
   wait

Adjust ``filename=`` after ``nvme list-subsys`` on each host.
