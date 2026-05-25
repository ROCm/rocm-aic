<!--
Copyright (c) Advanced Micro Devices, Inc. All rights reserved.
SPDX-License-Identifier: MIT
-->

# WEKA cluster notes (internal)

Notes from a Weka call (March 18, 2026). Not operator documentation.

Stephen to add more compute containers to the cluster. Use the same steps as
per the drive based container:

```
$ creation --cores 2 --flag
$ weka local setup container --name default6 --net $NIC --cores 2 \
  --cores-ids 12,13 --drives-dedicated-cores 1 \
  --compute-dedicated-cores 1 --no-frontends --base-port 17000 \
  --memory 16GB --failure-domain fd7
```

Only the first container needs a frontend to access the filesystem. Add to the
existing containers. Add a frontend to the first container. Be sure to update
the `--core-ids`.

- `weka cluster process` — CPU is the core id
- `weka status`
- `weka cloud enable` (sends heuristics to Weka)
