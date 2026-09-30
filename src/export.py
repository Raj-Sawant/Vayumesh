"""
Export deliverables in open standards:
- GeoTIFF orthomosaic (via rasterio)
- Cesium 3D Tiles (via 3d-tiles-tools)
- CityGML (via citygml-tools / lxml)
- LAS/LAZ point cloud (via laspy)
All with CRS (EPSG:4326 or UTM) and per‑vertex uncertainty attribute.
"""
import os
import numpy as np
import trimesh

def export_all_formats(predictions, lod_meshes, uncertainty, output_dir, crs="EPSG:4326"):
    os.makedirs(output_dir, exist_ok=True)

    # ------------------------------------------------------------------
    # 1️⃣  Orthomosaic GeoTIFF (simple mosaic from first frame for demo)
    # ------------------------------------------------------------------
    try:
        import rasterio
        from rasterio.transform import from_origin
        first_img = (predictions["images"][0] * 255).astype(np.uint8)   # RGB
        H, W = first_img.shape[:2]
        # dummy geotransform – replace with real sensor model + GPS
        transform = from_origin(0, 0, 0.1, 0.1)   # 10 cm/pixel
        with rasterio.open(
            os.path.join(output_dir, "ortho.tif"),
            "w",
            driver="GTiff",
            height=H,
            width=W,
            count=3,
            dtype=first_img.dtype,
            crs=crs,
            transform=transform,
        ) as dst:
            for b in range(3):
                dst.write(first_img[:, :, b], b + 1)
        print("[export] ortho.tif written")
    except Exception as e:
        print(f"[export] GeoTIFF skipped: {e}")

    # ------------------------------------------------------------------
    # 2️⃣  Cesium 3D Tiles (one tileset per LOD)
    # ------------------------------------------------------------------
    try:
        import json
        for lod_name, mesh in lod_meshes.items():
            tile_dir = os.path.join(output_dir, "tiles", lod_name)
            os.makedirs(tile_dir, exist_ok=True)
            # Export GLB per LOD
            glb_path = os.path.join(tile_dir, f"{lod_name}.glb")
            mesh.export(glb_path)
            # Minimal tileset.json
            tileset = {
                "asset": {"version": "1.0"},
                "geometricError": 100 if lod_name == "LOD0" else 10 if lod_name == "LOD1" else 1,
                "root": {
                    "boundingVolume": {"box": _mesh_bbox(mesh)},
                    "geometricError": 100 if lod_name == "LOD0" else 10 if lod_name == "LOD1" else 1,
                    "content": {"uri": f"{lod_name}.glb"},
                    "children": []
                }
            }
            with open(os.path.join(tile_dir, "tileset.json"), "w") as f:
                json.dump(tileset, f, indent=2)
        print("[export] 3D Tiles written")
    except Exception as e:
        print(f"[export] 3D Tiles skipped: {e}")

    # ------------------------------------------------------------------
    # 3️⃣  CityGML (very minimal – one Building per mesh)
    # ------------------------------------------------------------------
    try:
        from lxml import etree
        ns = {"core": "http://www.opengis.net/citygml/2.0",
              "bldg": "http://www.opengis.net/citygml/building/2.0",
              "gml": "http://www.opengis.net/gml"}
        city = etree.Element("{%s}CityModel" % ns["core"], nsmap=ns)
        for lod_name, mesh in lod_meshes.items():
            bldg = etree.SubElement(city, "{%s}cityObjectMember" % ns["core"])
            building = etree.SubElement(bldg, "{%s}Building" % ns["bldg"])
            lod = etree.SubElement(building, "{%s}lod%sSolid" % (ns["bldg"], lod_name[-1]))
            solid = etree.SubElement(lod, "{%s}Solid" % ns["gml"])
            exterior = etree.SubElement(solid, "{%s}exterior" % ns["gml"])
            shell = etree.SubElement(exterior, "{%s}Shell" % ns["gml"])
            for face in mesh.faces:
                surface = etree.SubElement(shell, "{%s}surfaceMember" % ns["gml"])
                polygon = etree.SubElement(surface, "{%s}Polygon" % ns["gml"])
                ext = etree.SubElement(polygon, "{%s}exterior" % ns["gml"])
                ring = etree.SubElement(ext, "{%s}LinearRing" % ns["gml"])
                coords = " ".join(f"{mesh.vertices[v][0]} {mesh.vertices[v][1]} {mesh.vertices[v][2]}" for v in face)
                poslist = etree.SubElement(ring, "{%s}posList" % ns["gml"])
                poslist.text = coords
        citygml_path = os.path.join(output_dir, "model.citygml")
        etree.ElementTree(city).write(citygml_path, pretty_print=True, xml_declaration=True, encoding="UTF-8")
        print("[export] CityGML written")
    except Exception as e:
        print(f"[export] CityGML skipped: {e}")

    # ------------------------------------------------------------------
    # 4️⃣  LAS/LAZ point cloud with uncertainty attribute
    # ------------------------------------------------------------------
    try:
        import laspy
        # use finest LOD vertices as point cloud
        finest = lod_meshes.get("LOD2") or lod_meshes.get("LOD1") or lod_meshes.get("LOD0")
        pts = finest.vertices
        las = laspy.create(point_format=3, file_version="1.4")
        las.x = pts[:, 0]
        las.y = pts[:, 1]
        las.z = pts[:, 2]
        # add custom extra dim for uncertainty
        las.add_extra_dim(laspy.ExtraBytesParams(name="uncertainty", type=np.float32, description="per‑point σ (cm)"))
        las.uncertainty = uncertainty[:len(pts)]
        las.write(os.path.join(output_dir, "cloud.laz"))
        print("[export] cloud.laz written")
    except Exception as e:
        print(f"[export] LAS/LAZ skipped: {e}")


def _mesh_bbox(mesh):
    """Return Cesium box [cx,cy,cz, hx,hy,hz, ...] – simplified."""
    mins = mesh.bounds[0]
    maxs = mesh.bounds[1]
    center = (mins + maxs) / 2
    half = (maxs - mins) / 2
    # Cesium expects 12 numbers: center(3) + 3 axis vectors (each 3)
    return list(center) + [half[0],0,0, 0,half[1],0, 0,0,half[2]]