import json
import unittest
from pathlib import Path


class VariantConfigTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        path = Path(__file__).parents[1] / "variants.json"
        cls.variants = json.loads(path.read_text(encoding="utf-8"))

    def test_expected_retrieval_candidates_exist(self) -> None:
        geometries = {
            (variant["index_dim"], variant["index_bits"], variant["retrieval_block_size"])
            for variant in self.variants.values()
        }
        self.assertEqual(geometries, {(64, 4, 4), (128, 2, 4), (128, 4, 8)})

    def test_ffn_reallocation_preserves_average_width(self) -> None:
        for variant in self.variants.values():
            average = (3 * variant["recurrent_intermediate_size"]
                       + variant["global_intermediate_size"]) // 4
            self.assertEqual(average, 14_336)


if __name__ == "__main__":
    unittest.main()
