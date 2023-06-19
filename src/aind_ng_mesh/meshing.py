from zmesh import Mesher


# Mesh generation
def labels_to_meshes(labels, anistropy=(1.0, 1.0, 1.0)):
    # Initial mesh
    mesher = Mesher(anistropy)
    mesher.mesh(labels, close=True)

    # Extract and simplify
    meshes = dict()
    for obj_id in mesher.ids():
        mesh = extract_mesh(mesher, obj_id)
        mesh = simplify_mesh(mesher, mesh)
        mesh.triangles()
        meshes[obj_id] = mesh
    mesher.clear()
    return meshes

def extract_mesh(mesher, obj_id):
    mesh = mesher.get_mesh(
        obj_id,
        normals=False,
        simplification_factor=100,
        voxel_centered=False,
    )
    mesher.erase(obj_id)
    return mesh


def simplify_mesh(mesher, mesh):
    mesh = mesher.simplify(
        mesh,
        reduction_factor=100,
        max_error=40,
        compute_normals=True,
    )
    mesh.triangles()
    return mesh


# Save mesh
def save_mesh(meshes, mesh_dir):
    write_mesh_info(mesh_dir)
    for obj_id in meshes.keys():
        filename = f"{obj_id}:0:0000000000000000"
        write_mesh_filenames(mesh_dir, obj_id)
        with open(os.path.join(mesh_dir, filename), "wb") as f:
            f.write(meshes[obj_id].to_precomputed())


def write_mesh_info(mesh_dir):
    info = {"@type": "neuroglancer_legacy_mesh"}
    with open(f"{mesh_dir}/info", "w") as f:
        json.dump(info, f)
    print("/info")


def write_mesh_filenames(mesh_dir, obj_id):
    data = {"fragments": [f"{obj_id}:0:0000000000000000"]}
    with open(f"{mesh_dir}/{obj_id}:{0}", "w") as f:
        json.dump(data, f)