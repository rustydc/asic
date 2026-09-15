import json
import unittest
from pathlib import Path


GEOMETRIES = {"qwen3_5_9b", "qwen3_5_4b"}


class VariantConfigTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        path = Path(__file__).parents[1] / "variants.json"
        cls.variants = json.loads(path.read_text(encoding="utf-8"))

    def test_expected_retrieval_candidates_exist(self) -> None:
        geometries = {
            (variant["index_dim"], variant["index_bits"], variant["retrieval_block_size"])
            for name, variant in self.variants.items()
            if variant["geometry"] == "qwen3_5_9b"
        }
        self.assertEqual(geometries, {(64, 4, 4), (128, 2, 4), (128, 4, 8)})

    def test_every_variant_names_a_known_geometry(self) -> None:
        for name, variant in self.variants.items():
            self.assertIn(variant.get("geometry"), GEOMETRIES, name)

    def test_variants_do_not_override_teacher_ffn_widths(self) -> None:
        for name, variant in self.variants.items():
            self.assertNotIn("recurrent_intermediate_size", variant, name)
            self.assertNotIn("global_intermediate_size", variant, name)
            self.assertNotIn("intermediate_size", variant, name)


if __name__ == "__main__":
    unittest.main()
