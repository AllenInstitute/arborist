"""
Created on Wed June 5 16:00:00 2023

@author: Anna Grim
@email: anna.grim@alleninstitute.org

Routines for reading and writing SWC files.

SWC format: each non-comment line is a node with fields
    id  type  x  y  z  radius  parent_id

"""

from botocore import UNSIGNED as _UNSIGNED
from botocore.config import Config as _BotocoreConfig
from google.cloud import storage as gcs_storage

import boto3
import json
import os


# --- IO Utils ---

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


def write_json(path, contents):
    """
    Writes contents to a JSON file.

    Parameters
    ----------
    path : str
        Destination file path.
    contents : dict
        Data to serialize.
    """
    with open(path, "w") as f:
        json.dump(contents, f, indent=4)


# --- Path Utils ---

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


# --- GCS Utils ---

def _list_gcs_paths(path, extension=""):
    bucket_name, prefix = parse_cloud_path(path)
    bucket = gcs_storage.Client().bucket(bucket_name)
    paths = []
    for name in [b.name for b in bucket.list_blobs(prefix=prefix)]:
        if extension in name:
            paths.append(os.path.join(f"gs://{bucket_name}", name))
    return sorted(paths)


def _read_gcs_txt(path, client=None):
    client = client or gcs_storage.Client()
    bucket_name, subprefix = parse_cloud_path(path)
    bucket = client.bucket(bucket_name)
    return bucket.blob(subprefix).download_as_text()


# --- S3 Utils ---

def _list_s3_paths(path, extension=""):
    bucket_name, prefix = parse_cloud_path(path)
    s3 = boto3.client(
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
    bucket_name, key = parse_cloud_path(path)
    s3 = client or boto3.client(
        "s3", config=_BotocoreConfig(signature_version=_UNSIGNED)
    )
    s3_obj = s3.get_object(Bucket=bucket_name, Key=key)["Body"]
    return s3_obj.read().decode()
