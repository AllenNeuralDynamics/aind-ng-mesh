"""
Created on Mon June 19 12:00:00 2023

@author: Anna Grim
@email: anna.grim@alleninstitute.org

"""
import aind_ng_mesh.meshing as meshing
import boto3
import json
import os
import shutil
import tensorstore as ts
from tifffile import imread
import zarr


def mkdir(path_to_dir):
    """
    Make a new directory at "path_to_dir".

    Parameters
    ----------
    path_to_dir : str
        Path to directory that will be created.

    Returns
    -------
    None

    """
    if not os.path.exists(path_to_dir):
        os.mkdir(path_to_dir)


def read_block(path):
    """
    Uploads image volume stored at "path" on local machine.

    Parameters
    ----------
    path : str
        Path where block is stored.

    Returns
    -------
    numpy.array

    """
    if "tif" in path:
        return imread(path)
    elif ".n5" in path:
        return zarr.open(zarr.N5FSStore(path), "r").volume[:]


def write_to_s3(
    labels,
    meshes,
    root_dir,
    bucket,
    s3_prefix,
    access_id=None,
    access_key=None,
):
    """
    Writes to "labels" and "meshes" to an s3 bucket.

    Parameters
    ----------
    labels : numpy.array
        Segmentation.
    meshes : dict
        Dictionary of meshes where the keys are object ids and values are meshes.
    root_dir : str
        Directory where files will be written to inorder to write to s3. This
        directory is deleted after the files are uploaded.
    bucket : str
        Name of s3 bucket.
    s3_prefix : str
        Path where data will be stored in "bucket".

    Returns
    -------
    None

    """
    # Create temp directory for uploading
    upload_dir = os.path.join(root_dir, "upload_dir")
    mesh_dir = os.path.join(upload_dir, "mesh")
    mkdir(upload_dir)

    # Store labels and meshes
    print("Converting to precomputed format...")
    to_precomputed(upload_dir, labels)
    obj_ids = meshing.save_mesh(meshes, mesh_dir)
    write_segment_properties(upload_dir, obj_ids)

    # Write to s3
    print("Writing to s3 bucket...")
    to_s3(
        upload_dir,
        bucket,
        s3_prefix,
        access_id=access_id,
        secret_access_key=access_key,
    )
    shutil.rmtree(upload_dir)


def to_s3(
    directory_path, bucket, s3_prefix, access_id=None, secret_access_key=None
):
    """
    Uploads a directory to an s3 bucket.

    """
    # Create session
    session = boto3.Session()
    s3_client = session.client("s3")

    # Upload files
    for root, dirs, files in os.walk(directory_path):
        for file_name in files:
            local_path = os.path.join(root, file_name)
            s3_key = os.path.join(
                s3_prefix, os.path.relpath(local_path, directory_path)
            )

            s3_client.upload_file(local_path, bucket, s3_key)

        for dir_name in dirs:
            local_dir_path = os.path.join(root, dir_name)
            s3_dir_key = (
                os.path.join(
                    s3_prefix, os.path.relpath(local_dir_path, directory_path)
                )
                + "/"
            )
            s3_client.put_object(Body="", Bucket=bucket, Key=s3_dir_key)


def to_precomputed(path, block):
    spec = {
        "dtype": "uint16",
        "driver": "neuroglancer_precomputed",
        "kvstore": {
            "driver": "file",
            "path": path,
        },
        "create": True,
        "delete_existing": True,
    }
    shape = block.shape + (1,)
    dataset = (
        ts.open(
            spec,
            dtype=ts.uint16,
            shape=shape,
        )
        .result()
        .T
    )
    write_future = dataset.write(block)
    edit_info(path)


def edit_info(precomputed_dir):
    """
    Edits info file in "precomputed_dir" so that it points to the mesh
    directory.

    Parameters
    ----------
    precomputed_dir : str
        Path to parent directory that contains precomputed labels.

    Returns
    -------
    None

    """
    path = f"{precomputed_dir}/info"
    info = read_json(path)
    info["type"] = "segmentation"
    info["mesh"] = "mesh"
    info["segment_properties"] = "segment_properties"
    write_json(path, info)


def write_segment_properties(root_dir, seg_ids):
    """ """
    info = {
        "@type": "neuroglancer_segment_properties",
        "inline": {
            "ids": list(map(str, seg_ids)),
            "properties": [{
                "id": "label",
                "type": "label",
                "values": list(map(str, seg_ids))
            }]
        },
    }
    print(info)
    property_dir = os.path.join(root_dir, "segment_properties")
    mkdir(property_dir)
    write_json(os.path.join(property_dir, "info"), info)


def read_json(path):
    """
    Reads json file stored at "path".

    Parameters
    ----------
    path : str
        Path where json file is stored.

    Returns
    -------
    dict

    """
    with open(path, "r") as f:
        data = json.load(f)
    return data


def write_json(path, data):
    """
    Writes "data" as json file stored at "path".

    Parameters
    ----------
    path : str
        Path where json file is stored.
    data : dict or list
        Data to written to json file.

    Returns
    -------
    None

    """
    with open(path, "w") as f:
        json.dump(data, f)
