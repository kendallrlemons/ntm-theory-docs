"""Download public Fah-XChem analysis JSON files with resumable verification."""

import argparse
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import re
import subprocess
import urllib.parse
import urllib.request
import xml.etree.ElementTree as ET


BUCKET_URL = (
    "https://fah-public-data-covid19-moonshot-sprints.s3.us-east-2.amazonaws.com"
)
DEFAULT_SPRINTS = [
    "sprint-4", "sprint-5", "sprint-5-dimer", "sprint-5-retrospective",
    "sprint-7", "sprint-8", "sprint-9", "sprint-10", "sprint-11",
    "sprint-11A", "sprint-12",
]
MANIFEST_SCHEMA_VERSION = 1
S3_NAMESPACE = {"s3": "http://s3.amazonaws.com/doc/2006-03-01/"}
MD5_ETAG = re.compile(r"^[0-9a-fA-F]{32}$")


def list_prefix(prefix):
    """List all analysis.json objects below one bucket prefix."""
    objects = []
    token = None
    while True:
        params = {"list-type": "2", "prefix": prefix}
        if token is not None:
            params["continuation-token"] = token
        url = BUCKET_URL + "/?" + urllib.parse.urlencode(params)
        with urllib.request.urlopen(url, timeout=60) as response:
            root = ET.fromstring(response.read())
        for content in root.findall(".//s3:Contents", S3_NAMESPACE):
            key = content.findtext("s3:Key", namespaces=S3_NAMESPACE)
            if not key or not key.endswith("/analysis.json"):
                continue
            etag = content.findtext("s3:ETag", namespaces=S3_NAMESPACE)
            objects.append({
                "key": key,
                "size": int(content.findtext("s3:Size", namespaces=S3_NAMESPACE)),
                "etag": etag.strip('"') if etag else "",
                "url": BUCKET_URL + "/" + urllib.parse.quote(key, safe="/"),
            })
        truncated = root.findtext(
            "s3:IsTruncated", default="false", namespaces=S3_NAMESPACE,
        ).lower() == "true"
        if not truncated:
            break
        token = root.findtext("s3:NextContinuationToken", namespaces=S3_NAMESPACE)
        if not token:
            raise ValueError("Truncated S3 listing is missing its continuation token.")
    return objects


def list_analyses(sprints):
    by_key = {}
    for sprint in sprints:
        for item in list_prefix(f"dashboards/{sprint}/"):
            if item["key"] in by_key and by_key[item["key"]] != item:
                raise ValueError(f"Conflicting S3 metadata for {item['key']}")
            by_key[item["key"]] = item
    return [by_key[key] for key in sorted(by_key)]


def sha256_file(path):
    digest = hashlib.sha256()
    with open(path, "rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def md5_file(path):
    digest = hashlib.md5()
    with open(path, "rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _target_path(root, key):
    relative = PurePosixPath(key)
    if relative.is_absolute() or ".." in relative.parts:
        raise ValueError(f"Unsafe object key path: {key}")
    target = root.joinpath(*relative.parts).resolve()
    if os.path.commonpath([str(root.resolve()), str(target)]) != str(root.resolve()):
        raise ValueError(f"Object path escapes output directory: {key}")
    return target


def _load_manifest(path):
    if not path.exists():
        return {}
    with open(path) as stream:
        manifest = json.load(stream)
    if manifest.get("schema_version") != MANIFEST_SCHEMA_VERSION:
        raise ValueError(f"Unsupported download manifest schema: {path}")
    records = manifest.get("files")
    if not isinstance(records, list):
        raise ValueError(f"Invalid download manifest: {path}")
    by_key = {}
    for record in records:
        key = record.get("key")
        if not isinstance(key, str) or key in by_key:
            raise ValueError(f"Invalid or duplicate key in download manifest: {path}")
        by_key[key] = record
    return by_key


def _save_manifest(path, records):
    temporary = path.with_name(path.name + ".tmp")
    payload = {
        "schema_version": MANIFEST_SCHEMA_VERSION,
        "files": [records[key] for key in sorted(records)],
    }
    with open(temporary, "w") as stream:
        json.dump(payload, stream, indent=2, sort_keys=True)
        stream.write("\n")
    os.replace(temporary, path)


def _check_listing_matches(record, item, relative_path):
    expected = {
        "key": item["key"], "url": item["url"], "path": relative_path,
        "size": item["size"], "etag": item["etag"],
    }
    if any(record.get(key) != value for key, value in expected.items()):
        raise ValueError(
            f"Live S3 metadata differs from completed manifest for {item['key']}"
        )


def _verify_object(path, item):
    if path.stat().st_size != item["size"]:
        raise ValueError(f"Downloaded size mismatch for {item['key']}")
    etag = item["etag"]
    if MD5_ETAG.fullmatch(etag) and md5_file(path) != etag.lower():
        raise ValueError(f"Downloaded ETag/MD5 mismatch for {item['key']}")
    return sha256_file(path)


def download_analyses(items, output_dir):
    root = Path(output_dir)
    root.mkdir(parents=True, exist_ok=True)
    manifest_path = root / "download_manifest.json"
    records = _load_manifest(manifest_path)

    for item in items:
        key = item["key"]
        path = _target_path(root, key)
        relative_path = PurePosixPath(key).as_posix()
        record = records.get(key)
        if record is not None:
            _check_listing_matches(record, item, relative_path)
            if path.exists():
                if not path.is_file():
                    raise ValueError(f"Expected a file at {path}")
                if path.stat().st_size != record["size"] or sha256_file(path) != record.get("sha256"):
                    raise ValueError(f"Completed file failed manifest verification: {path}")
                print(f"Verified, skipping {key}")
                continue
        elif path.exists():
            raise FileExistsError(
                f"Refusing to overwrite unmanaged file {path}; use a fresh output directory."
            )

        path.parent.mkdir(parents=True, exist_ok=True)
        part = path.with_name(path.name + ".part")
        subprocess.run([
            "curl", "--fail", "--location", "--retry", "3",
            "--continue-at", "-", "--output", str(part), item["url"],
        ], check=True)
        try:
            sha256 = _verify_object(part, item)
        except ValueError:
            if item["etag"] and MD5_ETAG.fullmatch(item["etag"]):
                try:
                    part.unlink()
                except FileNotFoundError:
                    pass
            raise
        os.replace(part, path)
        records[key] = {
            "key": key, "url": item["url"], "path": relative_path,
            "size": item["size"], "etag": item["etag"], "sha256": sha256,
        }
        _save_manifest(manifest_path, records)
        print(f"Downloaded {key} ({item['size']} bytes)")


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--list-only", action="store_true")
    parser.add_argument("--sprints", nargs="+", default=DEFAULT_SPRINTS)
    args = parser.parse_args(argv)
    items = list_analyses(args.sprints)
    if args.list_only:
        for item in items:
            print(f"{item['key']}\t{item['size']}")
        return
    download_analyses(items, args.output_dir)


if __name__ == "__main__":
    main()
