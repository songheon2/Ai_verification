"""Property labels next to network names; no benchmark execution."""
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from visualization.RunLabels import network_display_name, property_display_name


class RunLabelTests(unittest.TestCase):
    def test_numbered_properties_are_compact_for_plain_and_compressed_paths(self):
        for path in ("prop_3.vnnlib.gz", "prop3.vnnlib", r"C:\bench\prop_03.vnnlib.gz", "prop-3.vnnlib"):
            with self.subTest(path=path):
                self.assertEqual(property_display_name(path), "prop3")

    def test_missing_generic_and_model_duplicate_properties_are_not_shown(self):
        for prop in (None, "", "property.vnnlib", "spec.vnnlib", "network.vnnlib.gz"):
            with self.subTest(prop=prop):
                self.assertIsNone(property_display_name(prop, "network.onnx"))
        for prop in (None, "", " "):
            self.assertEqual(network_display_name({"model_name": "network.onnx", "property_name": prop}), "network.onnx")

    def test_named_property_and_model_identifier_remain_separate(self):
        context = {"model_name": "network.onnx", "property_name": "prop3"}
        self.assertEqual(network_display_name(context), "network.onnx (prop3)")
        self.assertEqual(context["model_name"], "network.onnx")
        self.assertEqual(property_display_name("properties/safety_condition.vnnlib.gz"), "safety_condition")
        self.assertIsNone(network_display_name({"property_name": "prop3"}))

    def test_dpll_dashboard_appends_only_available_property(self):
        import matplotlib.pyplot as plt
        from visualization.SolverProgress import draw_dpll_theory_panel
        for prop in (None, "prop3"):
            with self.subTest(property=prop):
                context = {"model_name": "ACASXU_run2a_2_8_batch_2000.onnx", "model_layer_sizes": [5, 50, 5]}
                if prop:
                    context["property_name"] = prop
                figure, _ = draw_dpll_theory_panel([{"event": "run_context", "witness_context": context}])
                try:
                    labels = [item.get_text() for item in figure.texts if item.get_text().startswith("Network:")]
                    self.assertEqual(labels, ["Network: ACASXU_run2a_2_8_batch_2000.onnx" + (" (prop3)" if prop else "")])
                finally:
                    plt.close(figure)

    def test_property_context_is_available_before_encoding_and_during_solver(self):
        import Automation.AutoVerify as auto_verify
        from Automation.SolverStatus import SolverResult, SolverStatus
        model = Path(__file__).resolve().parents[2] / "xor_network.bin"
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            prop = root / "prop_3.vnnlib"
            prop.write_text('(declare-const X_0 Real) (declare-const X_1 Real) '
                            '(declare-const Y_0 Real) (assert (>= Y_0 0))', encoding="utf-8")
            run = root / "run"
            original_encode = auto_verify.encode_nn
            def encode(*args, **kwargs):
                initial = json.loads((run / "solver_progress.jsonl").read_text(encoding="utf-8").splitlines()[0])
                self.assertEqual(initial["witness_context"]["property_name"], "prop3")
                return original_encode(*args, **kwargs)
            def solver(*args, **kwargs):
                self.assertEqual(kwargs["progress"].witness_context["property_name"], "prop3")
                self.assertEqual(kwargs["progress"].witness_context["model_name"], "xor_network.bin")
                self.assertIn("(prop3)", kwargs["realtime_tree_title"])
                return SolverResult(status=SolverStatus.UNSAT, reason="BOOLEAN_UNSAT", rounds=1, elapsed_seconds=0)
            with patch("Automation.AutoVerify.encode_nn", side_effect=encode), \
                    patch("Automation.AutoVerify.dpll_t_detailed", side_effect=solver):
                result = auto_verify.run_vnnlib_verification(str(model), str(prop),
                    visualization_mode="realtime", realtime_panels=["dpll-theory"], visualization_output_dir=str(run))
            self.assertEqual(result["vnnlib"]["property_name"], "prop3")
            self.assertEqual(Path(result["vnnlib"]["path"]).name, "property.vnnlib")


if __name__ == "__main__":
    unittest.main()
