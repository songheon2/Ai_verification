"""Report publishing must tolerate Windows reader locks without hanging flush."""
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from visualization.AtomicFile import write_atomic_text
from visualization.ReluSearchTree import RealtimeSearchTreeVisualizer


class ReportFileLockTests(unittest.TestCase):
    def test_transient_lock_retries_and_publishes_complete_file(self):
        import os
        replace = os.replace
        calls = []
        def locked_once(source, destination):
            calls.append(source)
            if len(calls) < 3:
                raise PermissionError("reader lock")
            replace(source, destination)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "report.html"
            path.write_text("old", encoding="utf-8")
            with patch("visualization.AtomicFile.os.replace", side_effect=locked_once), \
                 patch("visualization.AtomicFile.sleep") as delay:
                write_atomic_text(path, "새 보고서")
            self.assertEqual(path.read_text(encoding="utf-8"), "새 보고서")
            self.assertEqual(delay.call_count, 2)
            self.assertEqual(list(Path(directory).iterdir()), [path])

    def test_persistent_lock_preserves_previous_report_and_cleans_temporary(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "report.html"
            path.write_text("old", encoding="utf-8")
            with patch("visualization.AtomicFile.os.replace", side_effect=PermissionError("locked")) as replace, \
                 patch("visualization.AtomicFile.sleep"):
                with self.assertRaises(PermissionError):
                    write_atomic_text(path, "new")
            self.assertEqual(replace.call_count, 4)
            self.assertEqual(path.read_text(encoding="utf-8"), "old")
            self.assertEqual(list(Path(directory).iterdir()), [path])

    def test_writer_recovers_on_next_update_and_flush_does_not_hang(self):
        writer = RealtimeSearchTreeVisualizer("unused.jsonl", "unused.html", update_interval_seconds=0)
        with patch.object(writer, "_write_once", side_effect=[PermissionError("locked"), None]) as publish:
            with self.assertLogs("visualization.ReluSearchTree", level="WARNING"):
                writer.request_update()
                worker = writer._worker
                if worker is not None:
                    worker.join(timeout=2)
                    self.assertFalse(worker.is_alive())
            self.assertIsNone(writer._worker)
            self.assertIsInstance(writer.last_write_error, PermissionError)
            writer.flush()
            writer.request_update()
            worker = writer._worker
            if worker is not None:
                worker.join(timeout=2)
                self.assertFalse(worker.is_alive())
            writer.flush()
            self.assertIsNone(writer.last_write_error)
            self.assertEqual(publish.call_count, 2)


if __name__ == "__main__":
    unittest.main()
