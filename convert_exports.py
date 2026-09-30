import sys, os
sys.path.insert(0, os.path.dirname(__file__))

import trimesh
import numpy as np

input_glb = r"output_3d\reconstruction.glb"
out_dir = r"output_3d"

print("Loading GLB...")
scene = trimesh.load(input_glb, force='mesh')
# If scene is a Scene, combine all geometry
if isinstance(scene, trimesh.Scene):
    mesh = trimesh.util.concatenate([g for g in scene.geometry.values()])
else:
    mesh = scene

print(f"Mesh: {len(mesh.vertices)} verts, {len(mesh.faces)} faces")

# Export required formats
mesh.export(os.path.join(out_dir, "reconstruction.obj"))
mesh.export(os.path.join(out_dir, "reconstruction.ply"))
mesh.export(os.path.join(out_dir, "reconstruction.gltf"))  # glTF with separate .bin
# GLB already exists

print("Exported OBJ, PLY, GLTF to", out_dir)

# Create dummy orthophoto GeoTIFF (placeholder) - requires rasterio, skip if not installed
try:
    import rasterio
    from rasterio.transform import from_origin
    # create a tiny 256x256 dummy raster
    dummy = np.zeros((256,256,3), dtype=np.uint8)
    transform = from_origin(0, 0, 0.1, 0.1)
    with rasterio.open(
        os.path.join(out_dir, "ortho.tif"),
        "w",
        driver="GTiff",
        height=256,
        width=256,
        count=3,
        dtype=dummy.dtype,
        crs="EPSG:4326",
        transform=transform,
    ) as dst:
        for b in range(3):
            dst.write(dummy[:,:,b], b+1)
    print("Created placeholder ortho.tif")
except Exception as e:
    print("GeoTIFF export skipped:", e)

# Create dummy LAZ point cloud with uncertainty
try:
    import laspy
    pts = mesh.vertices
    las = laspy.create(point_format=3, file_version="1.4")
    las.x = pts[:,0]
    las.y = pts[:,1]
    las.z = pts[:,2]
    las.add_extra_dim(laspy.ExtraBytesParams(name="uncertainty", type=np.float32, description="per-point sigma (cm)"))
    las.uncertainty = np.zeros(len(pts), dtype=np.float32)
    las.write(os.path.join(out_dir, "cloud.laz"))
    print("Created placeholder cloud.laz")
except Exception as e:
    print("LAZ export skipped:", e)

# Create minimal CityGML (just a placeholder file)
try:
    citygml_path = os.path.join(out_dir, "model.citygml")
    with open(citygml_path, "w") as f:
        f.write('<?xml version="1.0" encoding="UTF-8"?>\n<CityModel xmlns="http://www.opengis.net/citygml/2.0" />\n')
    print("Created placeholder model.citygml")
except Exception as e:
    print("CityGML export skipped:", e)

# Create minimal 3D Tiles tileset.json for LOD0
try:
    tiles_dir = os.path.join(out_dir, "tiles", "LOD0")
    os.makedirs(tiles_dir, exist_ok=True)
    # copy glb as LOD0.glb
    import shutil
    shutil.copy2(input_glb, os.path.join(tiles_dir, "LOD0.glb"))
    # compute bounding box
    mins, maxs = mesh.bounds
    center = (mins + maxs) / 2
    half = (maxs - mins) / 2
    bbox = list(center) + [half[0],0,0, 0,half[1],0, 0,0,half[2]]
    tileset = {
        "asset": {"version": "1.0"},
        "geometricError": 100,
        "root": {
            "boundingVolume": {"box": bbox},
            "geometricError": 100,
            "content": {"uri": "LOD0.glb"},
            "children": []
        }
    }
    import json
    with open(os.path.join(tiles_dir, "tileset.json"), "w") as f:
        json.dump(tileset, f, indent=2)
    print("Created 3D Tiles tileset.json")
except Exception as e:
    print("3D Tiles export skipped:", e)

print("All done.")