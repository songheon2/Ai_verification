import unittest

import matplotlib.pyplot as plt

from GenericNNEncoding import NNModel
from visualization.SplitHeatmap import draw_split_heatmap


class SplitHeatmapTests(unittest.TestCase):
    def test_every_model_node_gets_a_cell(self):
        model = NNModel(
            num_layers=2,
            layer_sizes=[3, 25, 2],
            weights=[],
            biases=[],
        )

        # 옛 선택 기준대로라면 cap=1 때문에 대부분 사라져야 하지만, 현재
        # 히트맵에서는 0회 노드까지 실제 셀로 모두 남아야 한다.
        fig, ax = draw_split_heatmap(
            model,
            {(1, 24): 7},
            threshold=99,
            cap=1,
        )
        self.addCleanup(plt.close, fig)

        cells = ax.images[0].get_array()
        self.assertEqual(cells.shape, (25, 3))
        self.assertEqual(cells.mask[:, 0].sum(), 22)
        self.assertEqual(cells.mask[:, 1].sum(), 0)
        self.assertEqual(cells.mask[:, 2].sum(), 23)
        self.assertEqual(cells[24, 1], 7)
        self.assertEqual(cells[0, 1], 0)

        image = ax.images[0]
        zero_color = image.cmap(image.norm(0))
        hotspot_color = image.cmap(image.norm(7))
        self.assertNotEqual(zero_color, hotspot_color)
        self.assertLess(sum(hotspot_color[:3]), sum(zero_color[:3]))


if __name__ == "__main__":
    unittest.main()
