# amdgpu-dkms-repack.py

## Overview

A simple python tool that can pull either a URL or file-based debian package
file (.deb) extract it, apply patches and repackage it with a new version tag
and name. This new package can then be installed on a system using normal tools
like `dpkg`.

## Patches

We provide a series of patches in the [patches](./patches) directory. These 
include patches to improve the dkms install itself and also add features to
things like [hipfile][ref-hipfile].

Note that these patches may not work with newer versions of the amdgpu-dkms
package due to changes in the source code. The example given below does work.

## Example Usage

Note that this example only works inside the AMD VPN as we pull from an internal
package hosting site.
```
./amdgpu-dkms-repack.py \
   https://mkmartifactory.amd.com:8443/artifactory/amdgpu-deb-local-new/pool/2295296/noble/a/amdgpu-dkms_6.18.8.31200000-2295296.24.04_all.deb \
  --patch patches \
  --version 6.18.8.31200000-2295296.24.05 \
  --output debs/amdgpu-dkms_6.18.8.31200000-2295296.24.05_all.deb
```

[ref-hipfile]: https://github.com/ROCm/hipFile
