"""
Created on Wed June 5 16:00:00 2023

@author: Anna Grim
@email: anna.grim@alleninstitute.org

Routines for working with SWC files. An SWC file is a text-based file format
used to represent the directed graphical structure of a neuron. It contains a
series of nodes such that each has the following attributes:
    "id" (int): node ID
    "type" (int): node type (e.g. soma, axon, dendrite)
    "x" (float): x coordinate
    "y" (float): y coordinate
    "z" (float): z coordinate
    "pid" (int): node ID of parent

Note: Each line in an SWC file corresponds to a node and contains these
      attributes in the same order.
"""

import boto3
from botocore import UNSIGNED as _UNSIGNED
from botocore.config import Config as _BotocoreConfig
from collections import deque
from concurrent.futures import (
    ProcessPoolExecutor,
    ThreadPoolExecutor,
    as_completed,
)
from google.auth.exceptions import TransportError
from google.cloud import storage as gcs_storage
from io import BytesIO, StringIO
from tqdm import tqdm
from zipfile import ZipFile

import ast
import networkx as nx
import numpy as np
import os

from arborist.utils import util


class Reader:
    """
    Class that reads SWC files stored in a (1) local directory, (2) local ZIP
    archive, and (3) local directory of ZIP archives.
    """

    gcs_client = None

    def __init__(self, anisotropy=(1.0, 1.0, 1.0), verbose=True):
        """
        Initializes a Reader object that reads SWC files.

        Parameters
        ----------
        swc_names : Set[str], optional
            Only SWC files with names in this set are loaded if provided.
            Otherwise, all SWC files are loaded. Default is an empty set.
        verbose : bool, optional
            Indication of whether to display a progress bar. Default is True.
        """
        self.anisotropy = anisotropy
        self.verbose = verbose

    @classmethod
    def _get_gcs_client(cls):
        if cls.gcs_client is None:
            cls.gcs_client = gcs_storage.Client()
        return cls.gcs_client

    @classmethod
    def _get_client(cls, path):
        if util.is_gcs_path(path):
            return cls._get_gcs_client()
        return None

    # --- Main ---

    def __call__(self, swc_pointer):
        """
        Loads SWC files based on the type pointer provided.

        Parameters
        ----------
        swc_pointer : str
            Object that points to SWC files to be read, must be one of:
                - file_path: Path to single SWC file
                - dir_path: Path to local directory with SWC files
                - zip_path: Path to local ZIP with SWC files
                - zip_dir_path: Path to local directory of ZIPs with SWC files
                - s3_dir_path: Path to S3 prefix with SWC files
                - gcs_dir_path: Path to GCS prefix with SWC files
                - gcs_zip_dir_path: Path to GCS prefix with ZIPs of SWC files

        Returns
        -------
        Deque[dict]
            Dictionaries whose keys and values are the attribute names and
            values from the SWC files. Each dictionary contains the following:
            items:
                - "id": unique identifier of each node in an SWC file.
                - "pid": parent ID of each node.
                - "radius": radius value corresponding to each node.
                - "xyz": coordinate corresponding to each node.
                - "filename": filename of SWC file
                - "swc_id": name of SWC file, minus the ".swc".
        """
        # Case 1: List containing...
        if isinstance(swc_pointer, list):
            return self.read_swcs(swc_pointer)

        # Case 2: Directory containing...
        if os.path.isdir(swc_pointer):
            # Case 2.1: Local ZIP archives with SWC files
            zip_paths = util.list_paths(swc_pointer, extension=".zip")
            if zip_paths:
                return self.read_zips(zip_paths, self.read_zip)

            # Case 2.2: Local SWC files
            swc_paths = util.list_paths(swc_pointer, extension=".swc")
            if swc_paths:
                return self.read_swcs(swc_paths)
            raise ValueError(f"No SWC or ZIP files found in {swc_pointer}")

        # Case 3: Path to...
        if isinstance(swc_pointer, str):
            # Case 3.1: GCS/S3 prefix
            if util.is_gcs_path(swc_pointer) or util.is_s3_path(swc_pointer):
                return self._read_from_cloud(swc_pointer)

            # Case 3.2: ZIP archive
            if swc_pointer.endswith(".zip"):
                return self.read_zip(swc_pointer)

            # Case 3.3: SWC file
            if swc_pointer.endswith(".swc"):
                return deque([self.read_swc(swc_pointer)])

        raise ValueError(f"Unrecognised swc_pointer: {swc_pointer!r}")

    def read_swc(self, path):
        """
        Reads a single SWC file.

        Paramters
        ---------
        path : str
            Path to SWC file.

        Returns
        -------
        dict
            Dictionary whose keys and values are the attribute names and
            values from an SWC file.
        """
        client = self._get_client(path)
        content = util.read_txt(path, client=client).splitlines()
        filename = os.path.basename(path)
        return self.parse(content, filename)

    def read_swcs(self, swc_paths):
        """
        Reads SWC files stored in a GCS or S3 bucket.

        Parameters
        ----------
        swc_paths : List[str]
            List of paths to SWC files to be read.

        Returns
        -------
        swc_dicts : Deque[dict]
            Dictionaries whose keys and values are the attribute names and
            values from an SWC file.
        """
        with ThreadPoolExecutor() as executor:
            threads = {executor.submit(self.read_swc, p) for p in swc_paths}
            pbar = self._pbar(len(threads), "Read SWCs")
            swc_dicts = deque()
            for thread in as_completed(threads):
                result = thread.result()
                if result:
                    swc_dicts.append(result)
                if self.verbose and pbar:
                    pbar.update(1)
        return swc_dicts

    def read_zips(self, zip_paths, read_fn):
        """
        Reads SWC files stored in ZIP archives.

        Parameters
        ----------
        zip_paths : List[str]
            Paths to ZIP archives containing SWC files to be read.
        read_fn : callable
            Function used to read ZIP archives.

        Returns
        -------
        swc_dicts : Deque[dict]
            Dictionaries whose keys and values are the attribute names and
            values from an SWC file.
        """
        pbar = self._pbar(len(zip_paths), "Read ZIPs")
        with ProcessPoolExecutor() as executor:
            futures = {executor.submit(read_fn, p) for p in zip_paths}
            swc_dicts = deque()
            for process in as_completed(futures):
                try:
                    swc_dicts.extend(process.result())
                except Exception:
                    pass
                if self.verbose and pbar:
                    pbar.update(1)
        return swc_dicts

    def read_zip(self, zip_path):
        """
        Reads SWC files from a ZIP archive.

        Paramters
        ---------
        zip_path : str
            Path to ZIP archive.

        Returns
        -------
        swc_dicts : Deque[dict]
            Dictionaries whose keys and values are the attribute names and
            values from an SWC file.
        """
        with ThreadPoolExecutor() as executor:
            zf = ZipFile(zip_path, "r")
            threads = {
                executor.submit(self._read_zipped_swc, zf, name)
                for name in zf.namelist()
                if name.endswith(".swc")
            }
            swc_dicts = deque()
            for thread in as_completed(threads):
                result = thread.result()
                if result:
                    swc_dicts.append(result)
        return swc_dicts

    def _read_zipped_swc(self, zipfile, path):
        """
        Reads an SWC file stored in a ZIP archive.

        Parameters
        ----------
        zipfile : ZipFile
            ZIP archive containing SWC files.
        path : str
            Path to SWC file.

        Returns
        -------
        dict
            Dictionary whose keys and values are the attribute names and
            values from an SWC file.
        """
        content = util.read_zip_entry(zipfile, path).splitlines()
        filename = os.path.basename(path)
        return self.parse(content, filename)

    def _read_from_cloud(self, path):
        """
        Reads SWC files stored in a GCS or S3 bucket.

        Parameters
        ----------
        path : str
            Path to location in a GCS or S3 bucket containing SWC files,
            must be in the format "{scheme}://{bucket_name}/{prefix}".

        Returns
        -------
        Deque[dict]
            Dictionaries whose keys and values are the attribute names and
            values from an SWC file.
        """
        swc_paths = util.list_paths(path, extension=".swc")
        zip_paths = util.list_paths(path, extension=".zip")
        use_s3 = util.is_s3_path(path)
        if swc_paths:
            return self.read_swcs(swc_paths)
        elif zip_paths:
            read_fn = self._read_s3_zip if use_s3 else self._read_gcs_zip
            return self.read_zips(zip_paths, read_fn)
        return deque()

    def _read_gcs_zip(self, path):
        """
        Reads SWC files stored in a ZIP archive downloaded from a GCS
        bucket.

        Parameters
        ----------
        path : str
            Path to ZIP archive containing SWC files to be read.

        Returns
        -------
        swc_dicts : Deque[dict]
            Dictionaries whose keys and values are the attribute names and
            values from an SWC file.
        """
        bucket_name, key = util.parse_cloud_path(path)
        bucket = gcs_storage.Client().bucket(bucket_name)
        try:
            zip_content = bucket.blob(key).download_as_bytes()
        except TransportError:
            print(f"Failed to read {path}!")
            return deque()
        return self._parse_zip_bytes(zip_content)

    def _read_s3_zip(self, path):
        """
        Reads SWC files stored in a ZIP archive downloaded from an S3
        bucket.

        Parameters
        ----------
        path : str
            Path to ZIP archive containing SWC files to be read.

        Returns
        -------
        swc_dicts : Deque[dict]
            Dictionaries whose keys and values are the attribute names and
            values from an SWC file.
        """
        bucket, key = util.parse_cloud_path(path)
        s3 = boto3.client(
            "s3", config=_BotocoreConfig(signature_version=_UNSIGNED)
        )
        zip_content = s3.get_object(Bucket=bucket, Key=key)["Body"].read()
        return self._parse_zip_bytes(zip_content)

    def _parse_zip_bytes(self, zip_content):
        """
        Parse all SWC files contained in a ZIP archive.

        Parameters
        ----------
        zip_content : bytes
            Contents of a ZIP archive stored in memory.

        Returns
        -------
        Deque[dict]
            Dictionaries whose keys and values are the attribute names and
            values from an SWC file.
        """
        with ZipFile(BytesIO(zip_content), "r") as zf:
            names = [f for f in zf.namelist() if f.endswith(".swc")]
            with ThreadPoolExecutor() as executor:
                threads = {
                    executor.submit(self._read_zipped_swc, zf, name)
                    for name in names
                }
                return deque(
                    t.result() for t in as_completed(threads) if t.result()
                )

    def parse(self, content, filename):
        """
        Parses an SWC file to extract the content which is stored in a dict.

        Parameters
        ----------
        content : List[str]
            List of strings such that each is a line from an SWC file.

        Returns
        -------
        dict
            Dictionary whose keys and values are the attribute names and
            values from an SWC file.
        """
        swc_name, _ = os.path.splitext(filename)
        content, offset = self._process_content(content)
        n = len(content)
        swc_dict = {
            "id": np.zeros(n, dtype=int),
            "pid": np.zeros(n, dtype=int),
            "radius": np.zeros(n, dtype=float),
            "xyz": np.zeros((n, 3), dtype=np.int32),
            "soma_nodes": set(),
            "swc_name": swc_name,
        }
        for i, line in enumerate(content):
            parts = line.split()
            swc_dict["id"][i] = parts[0]
            swc_dict["pid"][i] = parts[-1]
            swc_dict["radius"][i] = float(parts[-2])
            swc_dict["xyz"][i] = self._read_coord(parts[2:5], offset)
            if int(parts[1]) == 1:
                swc_dict["soma_nodes"].add(parts[0])
        if swc_dict["radius"][0] > 100:
            swc_dict["radius"] /= 1000
        return swc_dict

    # --- Helpers ---

    def _pbar(self, total, desc):
        """
        Gets progress bar that needs to be updated manually.

        Parameters
        ----------
        total : int
            Size of progress bar.

        Returns
        -------
        tqdm.tqdm
            Iterator that is optionally wrapped in a progress bar.
        """
        return tqdm(total=total, desc=desc) if self.verbose else None

    def _process_content(self, content):
        """
        Processes lines of text from an SWC file, extracting an offset
        value and returning the remaining content starting from the line
        immediately after the last commented line.

        Parameters
        ----------
        content : List[str]
            List of strings such that each is a line from an SWC file.

        Returns
        -------
        content : List[str]
            Lines from an SWC file after comments.
        offset : Tuple[int]
            Offset used to shift coordinate.
        """
        offset = (0, 0, 0)
        for i, line in enumerate(content):
            if line.startswith("# OFFSET"):
                parts = line.split()
                offset = self._read_coord(parts[2:5])
            if not line.startswith("#") and len(line.strip()) > 0:
                return content[i:], offset
        return [], offset

    def _read_coord(self, xyz_str, offset=(0, 0, 0)):
        """
        Reads a coordinate from a string and converts it to voxel coordinates.

        Parameters
        ----------
        coord_str : str
            Coordinate stored as a string.
        offset : Tuple[int]
            Offset of coordinates in SWC file. Default is (0, 0, 0).

        Returns
        -------
        Tuple[int]
            xyz coordinates of an entry from an SWC file.
        """
        return [
            a * (float(s) + o)
            for a, s, o in zip(self.anisotropy, xyz_str, offset)
        ]


# --- Graph Conversion ---

def to_graph(swc_dict):
    """
    Converts a parsed SWC dictionary to a NetworkX graph with reindexed nodes.

    Parameters
    ----------
    swc_dict : dict
        Parsed SWC file contents.

    Returns
    -------
    networkx.Graph
    """
    id_map = {old: new for new, old in enumerate(swc_dict["id"])}
    edges = [
        (id_map[child], id_map[parent])
        for child, parent in zip(swc_dict["id"][1:], swc_dict["pid"][1:])
    ]
    graph = nx.Graph(
        swc_name=swc_dict["swc_name"],
        radius=swc_dict["radius"],
        xyz=swc_dict["xyz"],
    )
    graph.add_edges_from(edges)
    return graph


# --- Utilities ---

def get_swc_name(path):
    """Returns the SWC filename without its extension."""
    return os.path.splitext(os.path.basename(path))[0]


def get_segment_id(swc_name):
    """Extracts the segment ID from an SWC filename."""
    try:
        return ast.literal_eval(swc_name.split(".")[0])
    except Exception:
        return swc_name


# --- Write ---

def to_zipped_points(
    zip_path, points, color=None, prefix="", radius=10, write_mode="w"
):
    """
    Writes a list of 3D points as individual SWC files inside a ZIP archive.
    """
    zf = ZipFile(zip_path, write_mode)
    for i, xyz in enumerate(points):
        filename = prefix + str(i + 1) + ".swc"
        to_zipped_point(zf, filename, xyz, color=color, radius=radius)


def to_zipped_point(zf, filename, xyz, color=None, radius=5):
    """
    Writes a single 3D point as an SWC entry into an open ZIP archive.
    """
    with StringIO() as buf:
        if color:
            buf.write("# COLOR " + color)
        buf.write("\n# id, type, z, y, x, r, pid")
        x, y, z = tuple(xyz)
        buf.write(f"\n1 5 {x} {y} {z} {radius} -1")
        zf.writestr(filename, buf.getvalue())
