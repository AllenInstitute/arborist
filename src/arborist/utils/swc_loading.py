"""
Created on Wed June 5 16:00:00 2023

@author: Anna Grim
@email: anna.grim@alleninstitute.org

Routines for reading and writing SWC files.

SWC format: each non-comment line is a node with fields
    id  type  x  y  z  radius  parent_id

"""

from collections import deque
from concurrent.futures import (
    ProcessPoolExecutor,
    ThreadPoolExecutor,
    as_completed,
)
from io import BytesIO, StringIO
from tqdm import tqdm
from zipfile import ZipFile

import ast
import networkx as nx
import numpy as np
import os

try:
    from google.auth.exceptions import TransportError
    from google.cloud import storage as gcs_storage
    _GCS_AVAILABLE = True
except ImportError:
    _GCS_AVAILABLE = False

try:
    import boto3 as _boto3
    from botocore import UNSIGNED as _UNSIGNED
    from botocore.config import Config as _BotocoreConfig
    _S3_AVAILABLE = True
except ImportError:
    _S3_AVAILABLE = False


# --- Path helpers ---

def is_gcs_path(path):
    return isinstance(path, str) and path.startswith("gs://")


def is_s3_path(path):
    return isinstance(path, str) and path.startswith("s3://")


def parse_cloud_path(path):
    if path.startswith("s3://") or path.startswith("gs://"):
        path = path[5:]
    parts = path.split("/", 1)
    bucket = parts[0]
    key = parts[1] if len(parts) > 1 else ""
    return bucket, key


def listdir(path, extension=None):
    filenames = [f for f in os.listdir(path) if not f.startswith(".")]
    if extension:
        return [f for f in filenames if f.endswith(extension)]
    return filenames


def list_paths(dir_path, extension=""):
    if is_gcs_path(dir_path):
        return _list_gcs_paths(dir_path, extension=extension)
    elif is_s3_path(dir_path):
        return _list_s3_paths(dir_path, extension)
    else:
        filenames = listdir(dir_path, extension=extension or None)
        return [os.path.join(dir_path, f) for f in filenames]


def read_txt(path, client=None):
    if is_s3_path(path):
        return _read_s3_txt(path, client=client)
    elif is_gcs_path(path):
        return _read_gcs_txt(path, client=client)
    else:
        with open(path, "r") as f:
            return f.read()


def read_zip_entry(zip_file, path):
    with zip_file.open(path) as f:
        return f.read().decode("utf-8")


# --- GCS helpers ---

def _list_gcs_paths(path, extension=""):
    assert _GCS_AVAILABLE, "google-cloud-storage not installed"
    bucket_name, prefix = parse_cloud_path(path)
    bucket = gcs_storage.Client().bucket(bucket_name)
    paths = []
    for name in [b.name for b in bucket.list_blobs(prefix=prefix)]:
        if extension in name:
            paths.append(os.path.join(f"gs://{bucket_name}", name))
    return sorted(paths)


def _read_gcs_txt(path, client=None):
    assert _GCS_AVAILABLE, "google-cloud-storage not installed"
    bucket_name, subprefix = parse_cloud_path(path)
    client = client or gcs_storage.Client()
    bucket = client.bucket(bucket_name)
    return bucket.blob(subprefix).download_as_text()


# --- S3 helpers ---

def _list_s3_paths(path, extension=""):
    assert _S3_AVAILABLE, "boto3 not installed"
    bucket_name, prefix = parse_cloud_path(path)
    s3 = _boto3.client(
        "s3", config=_BotocoreConfig(signature_version=_UNSIGNED)
    )
    response = s3.list_objects_v2(Bucket=bucket_name, Prefix=prefix)
    paths = []
    if "Contents" in response:
        for obj in response["Contents"]:
            filename = obj["Key"]
            if filename.endswith(extension):
                paths.append(f"s3://{bucket_name}/{filename}")
    return paths


def _read_s3_txt(path, client=None):
    assert _S3_AVAILABLE, "boto3 not installed"
    bucket_name, key = parse_cloud_path(path)
    s3 = client or _boto3.client(
        "s3", config=_BotocoreConfig(signature_version=_UNSIGNED)
    )
    return s3.get_object(Bucket=bucket_name, Key=key)["Body"].read().decode()


# --- Reader ---

class Reader:
    """
    Reads SWC files from local directories, ZIP archives, or cloud storage
    (GCS / S3).
    """

    gcs_client = None

    def __init__(self, anisotropy=(1.0, 1.0, 1.0), min_swc_pts=1, verbose=True):
        self.anisotropy = anisotropy
        self.min_swc_pts = min_swc_pts
        self.verbose = verbose

    @classmethod
    def _get_gcs_client(cls):
        if cls.gcs_client is None:
            assert _GCS_AVAILABLE, "google-cloud-storage not installed"
            cls.gcs_client = gcs_storage.Client()
        return cls.gcs_client

    @classmethod
    def _get_client(cls, path):
        if is_gcs_path(path):
            return cls._get_gcs_client()
        return None

    def __call__(self, swc_pointer):
        """
        Reads SWC files. ``swc_pointer`` may be a local file, local directory,
        local ZIP, list of paths, or a cloud prefix (gs:// / s3://).
        """
        if isinstance(swc_pointer, list):
            return self.read_swcs(swc_pointer)

        if os.path.isdir(swc_pointer):
            zip_paths = list_paths(swc_pointer, extension=".zip")
            if zip_paths:
                return self.read_zips(zip_paths, self.read_zip)
            swc_paths = list_paths(swc_pointer, extension=".swc")
            if swc_paths:
                return self.read_swcs(swc_paths)
            raise ValueError(f"No SWC or ZIP files found in {swc_pointer}")

        if isinstance(swc_pointer, str):
            if is_gcs_path(swc_pointer) or is_s3_path(swc_pointer):
                return self._read_from_cloud(swc_pointer)
            if swc_pointer.endswith(".zip"):
                return self.read_zip(swc_pointer)
            if swc_pointer.endswith(".swc"):
                return deque([self.read_swc(swc_pointer)])

        raise ValueError(f"Unrecognised swc_pointer: {swc_pointer!r}")

    def read_swc(self, path):
        client = self._get_client(path)
        content = read_txt(path, client=client).splitlines()
        filename = os.path.basename(path)
        return self.parse(content, filename)

    def read_swcs(self, swc_paths):
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
        content = read_zip_entry(zipfile, path).splitlines()
        filename = os.path.basename(path)
        return self.parse(content, filename)

    def _read_from_cloud(self, path):
        swc_paths = list_paths(path, extension=".swc")
        zip_paths = list_paths(path, extension=".zip")
        use_s3 = is_s3_path(path)
        if swc_paths:
            return self.read_swcs(swc_paths)
        elif zip_paths:
            read_fn = self._read_s3_zip if use_s3 else self._read_gcs_zip
            return self.read_zips(zip_paths, read_fn)
        return deque()

    def _read_gcs_zip(self, path):
        assert _GCS_AVAILABLE, "google-cloud-storage not installed"
        bucket_name, key = parse_cloud_path(path)
        bucket = gcs_storage.Client().bucket(bucket_name)
        try:
            zip_content = bucket.blob(key).download_as_bytes()
        except TransportError:
            print(f"Failed to read {path}!")
            return deque()
        return self._parse_zip_bytes(zip_content)

    def _read_s3_zip(self, path):
        assert _S3_AVAILABLE, "boto3 not installed"
        bucket, key = parse_cloud_path(path)
        s3 = _boto3.client(
            "s3", config=_BotocoreConfig(signature_version=_UNSIGNED)
        )
        zip_content = s3.get_object(Bucket=bucket, Key=key)["Body"].read()
        return self._parse_zip_bytes(zip_content)

    def _parse_zip_bytes(self, zip_content):
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

    def _pbar(self, total, desc):
        return tqdm(total=total, desc=desc) if self.verbose else None

    def parse(self, content, filename):
        swc_name, _ = os.path.splitext(filename)
        content, offset = self._process_content(content)
        if len(content) < self.min_swc_pts:
            return None
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

    def _process_content(self, content):
        offset = (0, 0, 0)
        for i, line in enumerate(content):
            if line.startswith("# OFFSET"):
                parts = line.split()
                offset = self._read_coord(parts[2:5])
            if not line.startswith("#") and len(line.strip()) > 0:
                return content[i:], offset
        return [], offset

    def _read_coord(self, xyz_str, offset=(0, 0, 0)):
        return [
            a * (float(s) + o)
            for a, s, o in zip(self.anisotropy, xyz_str, offset)
        ]


# --- Graph conversion ---

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

def write_points(
    zip_path, points, color=None, prefix="", radius=10, write_mode="w"
):
    """Writes a list of 3D points as individual SWC files inside a ZIP archive."""
    zf = ZipFile(zip_path, write_mode)
    for i, xyz in enumerate(points):
        filename = prefix + str(i + 1) + ".swc"
        to_zipped_point(zf, filename, xyz, color=color, radius=radius)


def to_zipped_point(zf, filename, xyz, color=None, radius=5):
    """Writes a single 3D point as an SWC entry into an open ZIP archive."""
    with StringIO() as buf:
        if color:
            buf.write("# COLOR " + color)
        buf.write("\n# id, type, z, y, x, r, pid")
        x, y, z = tuple(xyz)
        buf.write(f"\n1 5 {x} {y} {z} {radius} -1")
        zf.writestr(filename, buf.getvalue())
