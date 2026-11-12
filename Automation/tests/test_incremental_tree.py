"""Correctness and work-count checks for incremental realtime reports."""
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from visualization.LiveSnapshot import snapshot_path
from visualization.ReluSearchTree import BranchLogReader, RealtimeSearchTreeVisualizer, trace_from_branch_log
from visualization.SolverMechanisms import mechanism_report_path


def start_record(name="b", variable="z1_0"):
    return {"event": "branch_start", "branch_id": name, "variable": variable,
            "time_ns": 1000000000, "depth": 1, "phase": "+"}


def append(path, record):
    with path.open("a", encoding="utf-8") as stream:
        stream.write(json.dumps(record, ensure_ascii=False) + "\n")


class IncrementalTreeTests(unittest.TestCase):
    def test_only_new_lines_are_decoded_and_open_branch_is_updated_in_place(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "branches.jsonl"
            append(path, start_record())
            reader = BranchLogReader(path)
            original_loads = json.loads
            with patch("visualization.ReluSearchTree.json.loads", wraps=original_loads) as decode:
                reader.read()
                event = reader.trace.events[0]
                reader.read()
                self.assertEqual(decode.call_count, 1)
                append(path, {"event": "branch_end", "branch_id": "b", "time_ns": 3000000000,
                              "result": "UNSAT", "reason": "LP", "unsat_origin": "LP"})
                reader.read()
                self.assertEqual(decode.call_count, 2)
                self.assertIs(reader.trace.events[0], event)
                self.assertEqual(event.duration, 2)
                self.assertEqual(event.meta["result"], "UNSAT")
            self.assertEqual(reader.trace, trace_from_branch_log(path))

    def test_partial_utf8_line_waits_for_completion(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "branches.jsonl"
            data = json.dumps(start_record(variable="뉴런"), ensure_ascii=False).encode("utf-8")
            cut = data.index("뉴".encode("utf-8")) + 1
            path.write_bytes(data[:cut])
            reader = BranchLogReader(path)
            reader.read()
            self.assertEqual(reader.offset, 0)
            self.assertFalse(reader.trace.events)
            with path.open("ab") as stream:
                stream.write(data[cut:] + b"\n")
            reader.read()
            self.assertEqual(reader.trace.events[0].branch_x, "뉴런")

    def test_rewritten_log_resets_even_when_new_file_is_longer(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "branches.jsonl"
            append(path, start_record())
            reader = BranchLogReader(path)
            reader.read()
            path.write_text(json.dumps(start_record("new", "z2_1234567")) + "\n", encoding="utf-8")
            reader.read()
            self.assertEqual(len(reader.trace.events), 1)
            self.assertEqual(reader.trace.events[0].branch_x, "z2_1234567")
            self.assertEqual(set(reader.open_events), {"new"})

    def test_unchanged_data_skips_rebuild_and_live_updates_write_only_snapshots(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            log, page = root / "branches.jsonl", root / "tree.html"
            append(log, start_record())
            writer = RealtimeSearchTreeVisualizer(log, page, update_interval_seconds=0)
            writer._write_once()
            original = page.read_bytes()
            mechanisms = mechanism_report_path(page)
            original_mechanisms = mechanisms.read_bytes()
            with patch("visualization.ReluSearchTree.write_relu_search_tree_html") as publish:
                writer._write_once()
                publish.assert_not_called()
            append(log, {"event": "branch_end", "branch_id": "b", "time_ns": 3000000000, "result": "SAT"})
            writer._write_once()
            self.assertEqual(page.read_bytes(), original)
            self.assertEqual(mechanisms.read_bytes(), original_mechanisms)
            snapshot = snapshot_path(page).read_text(encoding="utf-8")
            self.assertIn('"branchResult": "SAT"', snapshot)
            self.assertNotIn("function renderTree", snapshot)
            self.assertLess(len(snapshot), len(original))
            self.assertNotIn("function refresh", snapshot_path(mechanisms).read_text(encoding="utf-8"))
            writer.set_overall("SAT", "THEORY_SAT")
            writer._write_once()
            self.assertIn('"status": "SAT"', page.read_text(encoding="utf-8"))
            self.assertIn('"status": "SAT"', mechanisms.read_text(encoding="utf-8"))

    def test_metrics_change_is_not_skipped_without_new_log_lines(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            stats = {"DERIVE_TOTAL_RELUS": 12}
            writer = RealtimeSearchTreeVisualizer(root / "missing.jsonl", root / "tree.html", theory_stats=stats)
            writer._write_once()
            stats["DERIVE_FIXED_RELUS"] = 3
            writer._write_once()
            self.assertIn('"fixed": 3', snapshot_path(writer.output_path).read_text(encoding="utf-8"))

    def test_worker_throttles_updates_but_flush_can_bypass_delay(self):
        writer = RealtimeSearchTreeVisualizer("unused", "unused", update_interval_seconds=3)
        writer._last_write_started = 100
        with patch("visualization.ReluSearchTree.monotonic", return_value=101), \
             patch.object(writer._wake, "wait") as wait, patch.object(writer, "_write_once"):
            writer.request_update()
            worker = writer._worker
            if worker is not None:
                worker.join(timeout=2)
                self.assertFalse(worker.is_alive())
            wait.assert_called_once_with(2)
        writer._force = True
        with patch.object(writer._wake, "wait") as wait, patch.object(writer, "_write_once"):
            writer.request_update()
            worker = writer._worker
            if worker is not None:
                worker.join(timeout=2)
            wait.assert_called_once_with(0)


if __name__ == "__main__":
    unittest.main()
