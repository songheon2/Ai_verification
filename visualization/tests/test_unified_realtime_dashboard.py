from __future__ import annotations

from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from visualization.UnifiedRealtimeDashboard import (
    DedicatedRealtimeView,
    UnifiedRealtimeDashboard,
)


class UnifiedRealtimeDashboardTests(unittest.TestCase):
    def test_dedicated_view_refreshes_only_its_own_image(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            image = root / "relu.png"
            other_image = root / "simplex.png"
            live_view = root / "relu.html"
            view = DedicatedRealtimeView(
                image,
                live_view,
                title="Realtime ReLU Split History",
                description="active split history",
                refresh_interval_ms=200,
            )

            with patch(
                "visualization.UnifiedRealtimeDashboard.webbrowser.open"
            ) as open_browser:
                result = view.write(open_browser=True)

            self.assertEqual(result, live_view)
            html = live_view.read_text(encoding="utf-8")
            self.assertIn(image.resolve().as_uri(), html)
            self.assertNotIn(other_image.resolve().as_uri(), html)
            self.assertIn("setInterval(refresh, 200)", html)
            self.assertIn("Realtime ReLU Split History", html)
            open_browser.assert_called_once_with(live_view.resolve().as_uri())

    def test_one_html_refreshes_solver_and_relu_images(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            split_image = root / "split.png"
            dpll_theory_image = root / "dpll_theory.png"
            simplex_image = root / "simplex.png"
            live_view = root / "unified.html"
            dashboard = UnifiedRealtimeDashboard(
                split_image,
                dpll_theory_image,
                simplex_image,
                live_view,
                refresh_interval_ms=250,
            )

            with patch(
                "visualization.UnifiedRealtimeDashboard.webbrowser.open"
            ) as open_browser:
                result = dashboard.write(open_browser=True)

            self.assertEqual(result, live_view)
            html = live_view.read_text(encoding="utf-8")
            self.assertIn(split_image.resolve().as_uri(), html)
            self.assertIn(dpll_theory_image.resolve().as_uri(), html)
            self.assertIn(simplex_image.resolve().as_uri(), html)
            self.assertIn("setInterval(refresh, 250)", html)
            self.assertIn("DPLL Rounds &amp; Theory Flow", html)
            self.assertIn("Simplex Internals", html)
            self.assertIn("Realtime ReLU Split History", html)
            open_browser.assert_called_once_with(live_view.resolve().as_uri())


if __name__ == "__main__":
    unittest.main()
