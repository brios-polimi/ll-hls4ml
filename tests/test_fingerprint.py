import os
from pathlib import Path
import tempfile
import unittest

from ll_hls4ml.data.fingerprint import (
    assert_manifest_covers,
    build_content_manifest,
    load_content_manifest,
    write_content_manifest,
)


class ContentFingerprintTests(unittest.TestCase):
    def test_fingerprint_changes_when_content_changes_with_same_size_and_mtime(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            tensor = root / "family" / "archive_1" / "graph.pt"
            tensor.parent.mkdir(parents=True)
            tensor.write_bytes(b"first-content")
            stat = tensor.stat()
            first = build_content_manifest([tensor], root)

            tensor.write_bytes(b"other-content")
            os.utime(tensor, ns=(stat.st_atime_ns, stat.st_mtime_ns))
            second = build_content_manifest([tensor], root)

            self.assertNotEqual(
                first["snapshot_sha256"],
                second["snapshot_sha256"],
            )

            manifest_path = root / "manifest.json"
            write_content_manifest(manifest_path, second)
            loaded = load_content_manifest(manifest_path)
            self.assertEqual(loaded, second)
            assert_manifest_covers(loaded, [tensor], root)


if __name__ == "__main__":
    unittest.main()
