import hashlib
import io
import json
from pathlib import Path
import unittest
from contextlib import redirect_stdout
from tempfile import TemporaryDirectory
from unittest.mock import patch
from urllib.parse import parse_qs, urlsplit

import download_fah_analyses as downloader


S3_XML = "http://s3.amazonaws.com/doc/2006-03-01/"


def listing_page(contents, truncated=False, token=None):
    rows = []
    for key, size, etag in contents:
        rows.append(
            "<Contents><Key>{}</Key><Size>{}</Size><ETag>&quot;{}&quot;</ETag></Contents>".format(
                key, size, etag,
            )
        )
    next_token = f"<NextContinuationToken>{token}</NextContinuationToken>" if token else ""
    return (
        f'<ListBucketResult xmlns="{S3_XML}">'
        f"<IsTruncated>{str(truncated).lower()}</IsTruncated>"
        f"{next_token}{''.join(rows)}</ListBucketResult>"
    ).encode()


class DownloadListingTests(unittest.TestCase):
    def test_list_prefix_follows_s3_v2_pages_and_filters_keys(self):
        pages = [
            listing_page([
                ("dashboards/sprint-x/a/analysis.json", 12, "a" * 32),
                ("dashboards/sprint-x/a/other.json", 8, "b" * 32),
            ], truncated=True, token="next/token"),
            listing_page([
                ("dashboards/sprint-x/b/analysis.json", 34, "c" * 32),
            ]),
        ]
        requested = []

        def fake_urlopen(url, timeout):
            requested.append((url, timeout))
            return io.BytesIO(pages[len(requested) - 1])

        with patch.object(downloader.urllib.request, "urlopen", side_effect=fake_urlopen):
            items = downloader.list_prefix("dashboards/sprint-x/")

        self.assertEqual(
            [item["key"] for item in items],
            [
                "dashboards/sprint-x/a/analysis.json",
                "dashboards/sprint-x/b/analysis.json",
            ],
        )
        self.assertEqual([item["size"] for item in items], [12, 34])
        self.assertEqual(items[0]["etag"], "a" * 32)
        self.assertEqual(len(requested), 2)
        first_query = parse_qs(urlsplit(requested[0][0]).query)
        second_query = parse_qs(urlsplit(requested[1][0]).query)
        self.assertEqual(first_query["prefix"], ["dashboards/sprint-x/"])
        self.assertEqual(second_query["continuation-token"], ["next/token"])
        self.assertEqual([timeout for _, timeout in requested], [60, 60])

    def test_list_only_prints_key_and_size_without_writing(self):
        item = {
            "key": "dashboards/sprint-x/a/analysis.json",
            "size": 17, "etag": "d" * 32, "url": "https://example/file",
        }
        with TemporaryDirectory() as directory:
            output = Path(directory) / "not-created"
            stdout = io.StringIO()
            with (
                patch.object(downloader, "list_analyses", return_value=[item]),
                patch.object(downloader.subprocess, "run") as run,
                redirect_stdout(stdout),
            ):
                downloader.main([
                    "--output_dir", str(output), "--list-only",
                    "--sprints", "sprint-x",
                ])
            self.assertEqual(
                stdout.getvalue(),
                "dashboards/sprint-x/a/analysis.json\t17\n",
            )
            self.assertFalse(output.exists())
            run.assert_not_called()


class DownloadResumeTests(unittest.TestCase):
    def test_resume_download_skip_verified_file_and_reject_corruption(self):
        key = "dashboards/sprint-x/a/analysis.json"
        content = b'{"analysis": "synthetic"}'
        etag = hashlib.md5(content).hexdigest()
        item = {
            "key": key, "size": len(content), "etag": etag,
            "url": "https://bucket.example/" + key,
        }
        with TemporaryDirectory() as directory:
            root = Path(directory)
            target = root / key
            target.parent.mkdir(parents=True)
            part = target.with_name(target.name + ".part")
            part.write_bytes(content[:8])
            calls = []

            def fake_curl(command, check):
                calls.append((command, check))
                self.assertTrue(check)
                self.assertIn("--continue-at", command)
                self.assertEqual(command[command.index("--continue-at") + 1], "-")
                part_path = Path(command[command.index("--output") + 1])
                current = part_path.read_bytes() if part_path.exists() else b""
                with open(part_path, "ab") as stream:
                    stream.write(content[len(current):])

            with patch.object(downloader.subprocess, "run", side_effect=fake_curl):
                downloader.download_analyses([item], root)
            self.assertEqual(target.read_bytes(), content)
            self.assertFalse(part.exists())
            manifest_path = root / "download_manifest.json"
            with open(manifest_path) as stream:
                manifest = json.load(stream)
            self.assertEqual(manifest["schema_version"], 1)
            self.assertEqual(len(manifest["files"]), 1)
            self.assertEqual(manifest["files"][0]["path"], key)
            self.assertEqual(
                manifest["files"][0]["sha256"], hashlib.sha256(content).hexdigest(),
            )
            self.assertEqual(len(calls), 1)

            with patch.object(downloader.subprocess, "run") as run:
                downloader.download_analyses([item], root)
            run.assert_not_called()

            changed_listing = dict(item, etag="0" * 32)
            with patch.object(downloader.subprocess, "run") as run:
                with self.assertRaisesRegex(ValueError, "Live S3 metadata differs"):
                    downloader.download_analyses([changed_listing], root)
            run.assert_not_called()

            target.write_bytes(b"corrupted")
            with patch.object(downloader.subprocess, "run") as run:
                with self.assertRaisesRegex(ValueError, "manifest verification"):
                    downloader.download_analyses([item], root)
            run.assert_not_called()

    def test_existing_unmanaged_file_is_never_overwritten(self):
        key = "dashboards/sprint-y/x/analysis.json"
        item = {
            "key": key, "size": 3, "etag": hashlib.md5(b"new").hexdigest(),
            "url": "https://bucket.example/" + key,
        }
        with TemporaryDirectory() as directory:
            root = Path(directory)
            existing = root / key
            existing.parent.mkdir(parents=True)
            existing.write_bytes(b"old")
            with patch.object(downloader.subprocess, "run") as run:
                with self.assertRaisesRegex(FileExistsError, "unmanaged"):
                    downloader.download_analyses([item], root)
            self.assertEqual(existing.read_bytes(), b"old")
            run.assert_not_called()

    def test_download_rejects_singlepart_etag_corruption(self):
        key = "dashboards/sprint-z/a/analysis.json"
        content = b"expected"
        wrong = b"bad-data"
        item = {
            "key": key, "size": len(content),
            "etag": hashlib.md5(content).hexdigest(),
            "url": "https://bucket.example/" + key,
        }
        with TemporaryDirectory() as directory:
            root = Path(directory)

            def fake_curl(command, check):
                self.assertTrue(check)
                Path(command[command.index("--output") + 1]).write_bytes(wrong)

            with patch.object(downloader.subprocess, "run", side_effect=fake_curl):
                with self.assertRaisesRegex(ValueError, "ETag/MD5 mismatch"):
                    downloader.download_analyses([item], root)
            self.assertFalse((root / key).exists())
            self.assertFalse((root / "download_manifest.json").exists())
            self.assertFalse((root / (key + ".part")).exists())


if __name__ == "__main__":
    unittest.main()
