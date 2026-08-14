from __future__ import annotations

from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from visualization.RealtimeSplitVisualization import (
    RealtimeSplitVisualizer,
    SplitEventLogger,
    active_split_events,
    draw_realtime_split_dashboard,
    draw_realtime_split_history,
    read_split_events,
    split_neuron_counts,
)
from visualization.VisualizationMode import VisualizationMode, combined_mode
from visualization.realtime_split_demo import run_demo
from DPLL import AndProp, ReLUProp
from GenericNNEncoding import NNModel, encode_nn
from Reluplex import reluplex


def _relu_atoms(prop):
    if isinstance(prop, ReLUProp):
        return [prop]
    if isinstance(prop, AndProp):
        return _relu_atoms(prop.p) + _relu_atoms(prop.q)
    return []


class RealtimeSplitModeTests(unittest.TestCase):
    def test_plus_minus_are_paired_and_each_requests_update(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "log.txt"
            updates = []
            logger = SplitEventLogger(path, update_callback=lambda: updates.append(True))

            split_id = logger.begin("z2_7_run", 2, 7)
            self.assertEqual(logger.global_split_count, 1)
            self.assertEqual(len(active_split_events(read_split_events(path))), 1)
            logger.end(split_id, "z2_7_run", 2, 7)

            events = read_split_events(path)
            self.assertEqual([event.event for event in events], ["+", "-"])
            self.assertEqual(events[0].split_id, events[1].split_id)
            self.assertEqual((events[0].layer, events[0].index), (2, 7))
            self.assertEqual(logger.global_split_count, 0)
            self.assertEqual(active_split_events(events), [])
            self.assertEqual(len(updates), 2)

    def test_reset_log_starts_a_new_realtime_session(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "log.txt"
            path.write_text("old run\n", encoding="utf-8")

            SplitEventLogger(path, reset_log=True)

            self.assertEqual(path.read_text(encoding="utf-8"), "")

    def test_reluplex_records_balanced_events_around_two_branches(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "log.txt"
            logger = SplitEventLogger(path)
            _, sat = reluplex(
                [],
                {"z1_3": (-0.5, 0.5), "h1_3": (1.0, 1.0)},
                [("z1_3", "h1_3")],
                local_repair_max_iter=1,
                branch_tau=1,
                split_logger=logger,
                relu_metadata={("z1_3", "h1_3"): (1, 3)},
            )
            self.assertFalse(sat)
            events = read_split_events(path)
            self.assertEqual([event.event for event in events], ["+", "-"])
            self.assertEqual(logger.global_split_count, 0)

    def test_encoder_stores_layer_and_index_in_relu_ast(self):
        model = NNModel(
            num_layers=2,
            layer_sizes=[1, 2, 1],
            weights=[[[1.0], [-1.0]], [[1.0, 1.0]]],
            biases=[[0.0, 0.0], [0.0]],
        )
        prop, _, _ = encode_nn(model, ["x"])
        relus = _relu_atoms(prop)
        self.assertEqual([(atom.layer, atom.index) for atom in relus], [(1, 0), (1, 1)])

    def test_visualization_modes_can_be_combined_independently(self):
        self.assertEqual(
            combined_mode(realtime=False, feedback=False),
            VisualizationMode.OFF,
        )
        self.assertEqual(
            combined_mode(realtime=True, feedback=False),
            VisualizationMode.REALTIME,
        )
        self.assertEqual(
            combined_mode(realtime=False, feedback=True),
            VisualizationMode.FEEDBACK,
        )
        self.assertEqual(
            combined_mode(realtime=True, feedback=True),
            VisualizationMode.BOTH,
        )

    def test_live_view_is_created_and_auto_refreshes_png(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            log_path = root / "events.jsonl"
            output_path = root / "realtime.png"
            visualizer = RealtimeSplitVisualizer(
                log_path,
                output_path,
                live_view=True,
                refresh_interval_ms=250,
            )
            logger = SplitEventLogger(log_path, visualizer.request_update)

            logger.request_update()
            visualizer.flush()

            self.assertTrue(output_path.exists())
            self.assertTrue(visualizer.live_view_path.exists())
            html = visualizer.live_view_path.read_text(encoding="utf-8")
            self.assertIn('setInterval(refresh, 250)', html)
            self.assertIn('realtime.png', html)

    def test_live_view_browser_is_opened_only_once(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            visualizer = RealtimeSplitVisualizer(
                root / "events.jsonl",
                root / "realtime.png",
                open_live_view=True,
            )

            with patch(
                "visualization.RealtimeSplitVisualization.webbrowser.open"
            ) as open_browser:
                visualizer.request_update()
                visualizer.request_update()
                visualizer.flush()

            open_browser.assert_called_once_with(
                visualizer.live_view_path.resolve().as_uri()
            )

    def test_controlled_demo_shows_and_completes_multiple_splits(self):
        with tempfile.TemporaryDirectory() as directory:
            log_path, image_path, live_path = run_demo(
                output_dir=directory,
                split_count=2,
                interval_seconds=0,
                hold_seconds=0,
                open_live_view=False,
            )

            events = read_split_events(log_path)
            self.assertEqual([event.event for event in events], ["+", "+", "-", "-"])
            self.assertEqual(active_split_events(events), [])
            self.assertTrue(image_path.exists())
            self.assertTrue(live_path.exists())

    def test_realtime_history_is_drawn_as_a_connected_line(self):
        import matplotlib.pyplot as plt

        with tempfile.TemporaryDirectory() as directory:
            log_path = Path(directory) / "events.jsonl"
            logger = SplitEventLogger(log_path)
            first_id = logger.begin("z1_0", 1, 0)
            logger.begin("z1_1", 1, 1)
            logger.end(first_id, "z1_0", 1, 0)

            fig, ax = draw_realtime_split_history(log_path)
            try:
                self.assertEqual(len(ax.lines), 1)
                self.assertEqual(ax.get_ylabel(), "Active ReLU splits")
                self.assertIn("ACTIVE  1", [text.get_text() for text in ax.texts])
            finally:
                plt.close(fig)

    def test_layer_index_build_active_and_cumulative_neuron_state(self):
        with tempfile.TemporaryDirectory() as directory:
            log_path = Path(directory) / "events.jsonl"
            logger = SplitEventLogger(log_path)
            first_id = logger.begin("z1_2", 1, 2)
            logger.begin("z2_4", 2, 4)
            logger.end(first_id, "z1_2", 1, 2)

            active, totals, unknown_active = split_neuron_counts(
                read_split_events(log_path)
            )

            self.assertEqual(active, {(2, 4): 1})
            self.assertEqual(totals, {(1, 2): 1, (2, 4): 1})
            self.assertEqual(unknown_active, 0)

    def test_realtime_dashboard_places_neurons_by_layer_and_index(self):
        import matplotlib.pyplot as plt

        with tempfile.TemporaryDirectory() as directory:
            log_path = Path(directory) / "events.jsonl"
            logger = SplitEventLogger(log_path)
            logger.begin("z3_7", 3, 7)

            fig, axes = draw_realtime_split_dashboard(log_path)
            try:
                offsets = axes["neurons"].collections[0].get_offsets().tolist()
                self.assertIn([3.0, 7.0], offsets)
                self.assertIn("active=1", axes["neurons"].get_title(loc="left"))
            finally:
                plt.close(fig)

    def test_playback_renders_each_log_event_in_order(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            visualizer = RealtimeSplitVisualizer(
                root / "events.jsonl",
                root / "realtime.png",
                playback_interval_ms=1,
            )
            rendered_lengths = []
            logger = SplitEventLogger(
                visualizer.log_path,
                update_callback=visualizer.request_update,
                reset_log=True,
            )

            with patch.object(
                visualizer,
                "_render_once",
                side_effect=lambda events=None: rendered_lengths.append(len(events or [])),
            ):
                split_id = logger.begin("z1_0", 1, 0)
                logger.end(split_id, "z1_0", 1, 0)
                visualizer.flush()

            self.assertEqual(rendered_lengths, [1, 2])


if __name__ == "__main__":
    unittest.main()
