"""
Created on Mon June 19 12:00:00 2023

@author: Anna Grim
@email: anna.grim@alleninstitute.org

"""

import zarr
from tifffile import imread


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


def upload_block(path):
    """
    Uploads image volume stored at "path"

    Parameters
    ----------
    path : str
        Path where block is stored
    """
    if "tif" in path:
        return imread(path)
    if ".n5" in path:
        return zarr.open(zarr.N5FSStore(path), "r").volume[:]
        