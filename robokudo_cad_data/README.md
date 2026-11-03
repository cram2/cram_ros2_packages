# robokudo_cad_data

Textured 3D/CAD mesh models following the [CRAM conventions](https://ai.uni-bremen.de/wiki/3dmodeling/items): x-axis
left, y-axis backwards, z-axis up.

Each object has its own directory under `meshes/` with a `textured.obj`, its `textured.mtl` and the texture images.
They are installed to the package's share directory:

```python
from pathlib import Path

from ament_index_python.packages import get_package_share_directory

mustard = Path(get_package_share_directory("robokudo_cad_data")) / "meshes" / "mustard_frenchs_bottle" / "textured.obj"
```

The `FoundationPoseAnnotator` of RoboKudo and the tests of `robokudo_foundation_pose_tensorrt` use them.
