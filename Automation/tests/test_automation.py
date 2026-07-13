from __future__ import annotations

import importlib.util
from pathlib import Path
import sys
import unittest
from unittest.mock import patch


AUTOMATION_DIR = Path(__file__).resolve().parents[1]
PROJECT_DIR = AUTOMATION_DIR.parent
REPO_DIR = PROJECT_DIR.parent
if str(PROJECT_DIR) not in sys.path:
    sys.path.insert(0, str(PROJECT_DIR))

from Automation.AutoVerify import forward_model, run_verification, run_vnnlib_verification
from DPLL import AndProp, InequProp, Prop, parse_prop
from DPLL_T import dpll_t_detailed
from GenericNNEncoding import load_nn_model
from Automation.ModelInspector import _interpret_input_preprocessing, inspect_custom, inspect_model
from Automation.PropertyBuilder import (
    build_postcondition,
    build_precondition,
    infer_expected_from_outputs,
)
from Reluplex import reluplex
from Simplex import build_tableau, simplex
from Automation.SolverStatus import SolverLimitReached, SolverResult, SolverStatus
from Automation.VnnlibParser import (
    VnnlibParseError,
    parse_vnnlib_file,
    parse_vnnlib_text,
)


def inequalities(prop: Prop):
    if isinstance(prop, InequProp):
        return [prop]
    if isinstance(prop, AndProp):
        return inequalities(prop.p) + inequalities(prop.q)
    return []


class PropertyBuilderTests(unittest.TestCase):
    def test_standard_onnx_preprocessing_is_extracted(self):
        steps = [
            {
                "op": "Max",
                "data_input_index": 0,
                "constants": [{"values": [0.0, -10.0]}],
            },
            {
                "op": "Min",
                "data_input_index": 0,
                "constants": [{"values": [20.0, 30.0]}],
            },
            {
                "op": "Sub",
                "data_input_index": 0,
                "constants": [{"values": [10.0, 5.0]}],
            },
            {
                "op": "Div",
                "data_input_index": 0,
                "constants": [{"values": [2.0, 5.0]}],
            },
        ]
        preprocessing = _interpret_input_preprocessing(steps, 2)
        self.assertEqual(preprocessing["domain"]["lower"], [0.0, -10.0])
        self.assertEqual(preprocessing["domain"]["upper"], [20.0, 30.0])
        self.assertEqual(preprocessing["normalization"]["mean"], [10.0, 5.0])
        self.assertEqual(preprocessing["normalization"]["scale"], [2.0, 5.0])

    def test_per_feature_epsilon_is_clipped_to_domain(self):
        input_spec = {
            "space": "model",
            "domain": {"lower": [0.0, -2.0], "upper": [1.0, 2.0]},
        }
        case = {"center": [0.1, 1.5], "epsilon": [0.2, 1.0]}
        prop, lower, upper = build_precondition(["x0", "x1"], input_spec, case)
        self.assertEqual(lower, [0.0, 0.5])
        self.assertEqual(upper, [0.30000000000000004, 2.0])
        self.assertEqual(len(inequalities(prop)), 4)

    def test_raw_bounds_are_normalized(self):
        input_spec = {
            "space": "raw",
            "normalization": {"mean": [10.0, 100.0], "scale": [2.0, 20.0]},
        }
        case = {"lower": [8.0, 80.0], "upper": [12.0, 140.0]}
        _, lower, upper = build_precondition(["x0", "x1"], input_spec, case)
        self.assertEqual(lower, [-1.0, -1.0])
        self.assertEqual(upper, [1.0, 2.0])

    def test_binary_margin(self):
        positive = build_postcondition(
            ["y"], {"type": "binary", "threshold": 0.5, "margin": 0.2}, 1
        )
        negative = build_postcondition(
            ["y"], {"type": "binary", "threshold": 0.5, "margin": 0.2}, 0
        )
        self.assertEqual(positive.b, 0.7)
        self.assertEqual(dict(negative.coeffs), {"y": -1.0})
        self.assertAlmostEqual(negative.b, -0.3)

    def test_multiclass_argmax_and_argmin(self):
        argmax = build_postcondition(
            ["y0", "y1", "y2"],
            {"type": "multiclass", "decision": "argmax", "margin": 0.1},
            1,
        )
        argmin = build_postcondition(
            ["y0", "y1", "y2"],
            {"type": "multiclass", "decision": "argmin", "margin": 0.1},
            1,
        )
        self.assertEqual(len(inequalities(argmax)), 2)
        self.assertEqual(dict(inequalities(argmax)[0].coeffs)["y1"], 1.0)
        self.assertEqual(dict(inequalities(argmin)[0].coeffs)["y1"], -1.0)

    def test_multilabel_supports_per_output_threshold_and_margin(self):
        post = build_postcondition(
            ["y0", "y1"],
            {
                "type": "multilabel",
                "threshold": [0.0, 1.0],
                "margin": [0.1, 0.2],
            },
            [1, 0],
        )
        atoms = inequalities(post)
        self.assertEqual([atom.b for atom in atoms], [0.1, -0.8])

    def test_regression_target_tolerance(self):
        post = build_postcondition(
            ["y0", "y1"],
            {"type": "regression", "tolerance": [0.5, 1.0]},
            [2.0, 4.0],
        )
        self.assertEqual(len(inequalities(post)), 4)

    def test_center_prediction_is_explicitly_inferred(self):
        self.assertEqual(
            infer_expected_from_outputs(
                {"type": "multiclass", "decision": "argmax"}, [0.1, 0.8, 0.2]
            ),
            1,
        )


class SolverStatusTests(unittest.TestCase):
    def test_sat_is_distinguished(self):
        result = dpll_t_detailed(
            parse_prop("ineq(1,x,0)"), timeout_seconds=1.0
        )
        self.assertEqual(result.status, SolverStatus.SAT)

    def test_unsat_is_distinguished(self):
        result = dpll_t_detailed(
            parse_prop("ineq(1,x,1) and ineq(-1,x,0)"),
            timeout_seconds=1.0,
        )
        self.assertEqual(result.status, SolverStatus.UNSAT)

    def test_theory_conflict_blocks_signed_literals(self):
        result = dpll_t_detailed(
            parse_prop(
                "(ineq(1,x,1) and not ineq(1,x,0)) "
                "or (ineq(1,x,1) and ineq(1,x,0))"
            ),
            max_rounds=10,
            timeout_seconds=1.0,
        )
        self.assertEqual(result.status, SolverStatus.SAT)

    def test_false_theory_literals_do_not_short_circuit_unsat(self):
        result = dpll_t_detailed(
            parse_prop(
                "(not ineq(1,x,0) and not ineq(-1,x,0)) "
                "or (not ineq(1,x,0) and ineq(-1,x,0))"
            ),
            max_rounds=10,
            timeout_seconds=1.0,
        )
        self.assertEqual(result.status, SolverStatus.SAT)

    def test_dpll_t_round_limit_is_unknown(self):
        result = dpll_t_detailed(
            parse_prop("ineq(1,x,0)"),
            max_rounds=0,
            timeout_seconds=1.0,
        )
        self.assertEqual(result.status, SolverStatus.UNKNOWN)
        self.assertEqual(result.reason, "DPLL_T_ROUND_LIMIT")

    def test_timeout_is_unknown(self):
        result = dpll_t_detailed(
            parse_prop("ineq(1,x,0)"), timeout_seconds=0.0
        )
        self.assertEqual(result.status, SolverStatus.UNKNOWN)
        self.assertEqual(result.reason, "TIMEOUT")

    def test_legacy_dpll_t_does_not_silently_map_unknown_to_unsat(self):
        from DPLL_T import dpll_t

        with self.assertRaises(SolverLimitReached) as context:
            dpll_t(parse_prop("ineq(1,x,0)"), max_rounds=0)
        self.assertEqual(context.exception.reason, "DPLL_T_ROUND_LIMIT")

    def test_simplex_iteration_limit_is_unknown(self):
        tableau = build_tableau(
            [("s", {"x": 1.0})],
            {"s": (1.0, float("inf")), "x": (float("-inf"), float("inf"))},
        )
        with self.assertRaises(SolverLimitReached) as context:
            simplex(tableau, max_iter=0, report_unknown=True)
        self.assertEqual(context.exception.reason, "SIMPLEX_ITERATION_LIMIT")

    def test_reluplex_recursion_limit_is_unknown(self):
        with self.assertRaises(SolverLimitReached) as context:
            reluplex([], {}, [], max_recursion=-1, report_unknown=True)
        self.assertEqual(context.exception.reason, "RELUPLEX_RECURSION_LIMIT")


class VnnlibParserTests(unittest.TestCase):
    def setUp(self):
        self.substitutions = {
            "X_0": "x0",
            "X_1": "x1",
            "Y_0": "y0",
            "Y_1": "y1",
        }

    def test_linear_arithmetic_is_converted_to_inequality(self):
        text = """
        (declare-const X_0 Real)
        (declare-const X_1 Real)
        (assert (<= (+ (* 2 X_0) (- X_1) 3) 10))
        """
        document = parse_vnnlib_text(text, self.substitutions)
        atoms = inequalities(document.formula)
        self.assertEqual(len(atoms), 1)
        self.assertEqual(dict(atoms[0].coeffs), {"x0": -2.0, "x1": 1.0})
        self.assertEqual(atoms[0].b, -7.0)

    def test_boolean_output_property_and_declarations(self):
        text = """
        (declare-const X_0 Real)
        (declare-const X_1 Real)
        (declare-const Y_0 Real)
        (declare-const Y_1 Real)
        (assert (and (>= X_0 0) (<= X_0 1)))
        (assert (or (<= Y_0 Y_1) (>= Y_0 2)))
        """
        document = parse_vnnlib_text(text, self.substitutions)
        self.assertEqual(document.assertion_count, 2)
        self.assertEqual(document.input_variables, ("X_0", "X_1"))
        self.assertEqual(document.output_variables, ("Y_0", "Y_1"))

    def test_gzip_vnnlib_is_supported(self):
        import gzip
        import tempfile

        text = """
        (declare-const X_0 Real)
        (declare-const X_1 Real)
        (declare-const Y_0 Real)
        (declare-const Y_1 Real)
        (assert (>= X_0 0))
        """
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "property.vnnlib.gz"
            with gzip.open(path, "wt", encoding="utf-8") as handle:
                handle.write(text)
            document = parse_vnnlib_file(str(path), self.substitutions)
        self.assertEqual(document.assertion_count, 1)

    def test_nonlinear_expression_is_rejected(self):
        text = """
        (declare-const X_0 Real)
        (declare-const X_1 Real)
        (assert (>= (* X_0 X_1) 0))
        """
        with self.assertRaises(VnnlibParseError):
            parse_vnnlib_text(text, self.substitutions)


class AutomationIntegrationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.model_path = PROJECT_DIR / "xor_network.txt"

    @staticmethod
    def xor_spec():
        return {
            "model_contract": {"input_size": 2, "output_size": 1},
            "input": {
                "space": "model",
                "domain": {"lower": [0.0, 0.0], "upper": [1.0, 1.0]},
                "epsilon": 0.02,
            },
            "output": {
                "type": "binary",
                "threshold": 0.0,
                "margin": 0.0001,
            },
            "property": {"reference": "expected"},
            "cases": [
                {"name": "xor_00", "center": [0.0, 0.0], "expected": 0},
                {"name": "xor_01", "center": [0.0, 1.0], "expected": 1},
                {"name": "xor_10", "center": [1.0, 0.0], "expected": 1},
                {"name": "xor_11", "center": [1.0, 1.0], "expected": 0},
            ],
        }

    def test_custom_model_inspection(self):
        info = inspect_custom(str(self.model_path))
        self.assertEqual(info.layer_sizes, [2, 2, 1])
        self.assertEqual(info.relu_count, 2)

    @unittest.skipUnless(importlib.util.find_spec("onnx"), "onnx package is not installed")
    def test_acas_onnx_preprocessing_is_detected(self):
        path = PROJECT_DIR / "Onnx" / "ACASXU_experimental_v2a_1_1.onnx"
        info = inspect_model(str(path))
        self.assertEqual(info.layer_sizes, [5, 50, 50, 50, 50, 50, 50, 5])
        self.assertEqual(info.input_preprocessing["domain"]["lower"][0], 0.0)
        self.assertAlmostEqual(
            info.input_preprocessing["normalization"]["mean"][0],
            19791.091,
            places=2,
        )

    @unittest.skipUnless(importlib.util.find_spec("onnx"), "onnx package is not installed")
    def test_legacy_onnx_initializer_inputs_are_not_data_inputs(self):
        import tempfile
        import onnx
        from onnx import TensorProto, helper, numpy_helper
        import numpy as np

        data_input = helper.make_tensor_value_info("X", TensorProto.FLOAT, [1, 2])
        # 구형 ONNX 형식을 재현하기 위해 initializer W를 graph.input에도 등록한다.
        weight_input = helper.make_tensor_value_info("W", TensorProto.FLOAT, [2, 1])
        output = helper.make_tensor_value_info("Y", TensorProto.FLOAT, [1, 1])
        weight = numpy_helper.from_array(
            np.array([[1.0], [2.0]], dtype=np.float32), "W"
        )
        graph = helper.make_graph(
            [helper.make_node("MatMul", ["X", "W"], ["Y"])],
            "legacy_initializer_input",
            [data_input, weight_input],
            [output],
            [weight],
        )
        onnx_model = helper.make_model(graph)

        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "legacy.onnx"
            onnx.save(onnx_model, path)
            info = inspect_model(str(path))

        self.assertEqual(info.layer_sizes, [2, 1])

    def test_forward_model(self):
        model = load_nn_model(str(self.model_path))
        self.assertLess(forward_model(model, [0.0, 0.0])[0], 0.0)
        self.assertGreater(forward_model(model, [0.0, 1.0])[0], 0.0)

    def test_xor_spec_dry_run(self):
        result = run_verification(
            str(self.model_path), self.xor_spec(), dry_run=True
        )
        self.assertEqual(len(result["cases"]), 4)
        self.assertTrue(all(case["status"] == "DRY_RUN" for case in result["cases"]))

    def test_auto_verify_maps_unsat_to_verified(self):
        solver_result = SolverResult(
            status=SolverStatus.UNSAT,
            reason="BOOLEAN_UNSAT",
            rounds=3,
            elapsed_seconds=0.1,
        )
        with patch("Automation.AutoVerify.dpll_t_detailed", return_value=solver_result):
            result = run_verification(str(self.model_path), self.xor_spec())
        self.assertTrue(all(case["status"] == "VERIFIED" for case in result["cases"]))
        self.assertEqual(result["cases"][0]["solver"]["status"], "UNSAT")

    def test_auto_verify_maps_limit_to_unknown(self):
        solver_result = SolverResult(
            status=SolverStatus.UNKNOWN,
            reason="TIMEOUT",
            rounds=7,
            elapsed_seconds=1.0,
        )
        with patch("Automation.AutoVerify.dpll_t_detailed", return_value=solver_result):
            result = run_verification(str(self.model_path), self.xor_spec())
        self.assertTrue(all(case["status"] == "UNKNOWN" for case in result["cases"]))
        self.assertEqual(result["cases"][0]["solver"]["reason"], "TIMEOUT")

    def test_xor_wrong_expected_label_returns_counterexample(self):
        spec = {
            "model_contract": {"input_size": 2, "output_size": 1},
            "input": {
                "space": "model",
                "domain": {"lower": [0.0, 0.0], "upper": [1.0, 1.0]},
            },
            "output": {"type": "binary", "threshold": 0.0, "margin": 0.0},
            "property": {"reference": "expected"},
            "cases": [
                {
                    "name": "known_wrong_label",
                    "center": [0.0, 0.0],
                    "epsilon": 0.0,
                    "expected": 1,
                }
            ],
        }
        result = run_verification(str(self.model_path), spec)
        case = result["cases"][0]
        self.assertEqual(case["status"], "COUNTEREXAMPLE")
        self.assertEqual(case["counterexample"]["input_declared_space"], [0.0, 0.0])

    def test_xor_vnnlib_unsafe_region_returns_counterexample(self):
        import tempfile

        text = """
        (declare-const X_0 Real)
        (declare-const X_1 Real)
        (declare-const Y_0 Real)
        (assert (= X_0 0))
        (assert (= X_1 0))
        (assert (<= Y_0 0))
        """
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "xor_unsafe.vnnlib"
            path.write_text(text, encoding="utf-8")
            result = run_vnnlib_verification(str(self.model_path), str(path))
        self.assertEqual(result["status"], "COUNTEREXAMPLE")
        self.assertEqual(
            result["counterexample"]["input_declared_space"], [0.0, 0.0]
        )


if __name__ == "__main__":
    unittest.main()
