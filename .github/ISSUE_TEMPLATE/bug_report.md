---
name: Bug report
about: Create a report to help us improve
title: ''
labels: ''
assignees: ''

---

**Describe the bug**
A clear and concise description of what the bug is.

**Minimal Working Example**
Please provide a piece of code that leads to the bug you encounter.

If the code is **runnable**, it will help us identify the problem faster.

**Package versions**

Please provide the versions of packages you use. To do this, run the code:
```julia
import Pkg
Pkg.status([
    "Package1", "Package2"]; # etc.
    mode = PKGMODE_MANIFEST
)
```

Please also include the Julia version you run.