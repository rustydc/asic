"""Model and importer tests; skipped when PyTorch is not installed."""

import unittest
from pathlib import Path

try:
    import torch
except ImportError:  # pragma: no cover - exercised only without torch
    torch = None

if torch is not None:
    from fixed_llm_poc import ASICLM, ASICLMConfig, DeltaState, geometry_report, tiny_config
    from training.layerwise_distill import load_variant
    from training.qwen35_import import (config_from_hf, convert_state_dict, init_index_projections,
                                        load_into_model, rename_hf_key)


@unittest.skipIf(torch is None, "PyTorch not installed")
class GeometryTest(unittest.TestCase):
    def test_presets_match_qwen35_parameter_budgets(self) -> None:
        nine = geometry_report(ASICLMConfig.from_preset("qwen3_5_9b"))
        four = geometry_report(ASICLMConfig.from_preset("qwen3_5_4b"))
        self.assertAlmostEqual(nine.total / 1e9, 8.96, delta=0.05)
        self.assertAlmostEqual(four.total / 1e9, 4.21, delta=0.05)
        self.assertAlmostEqual(nine.per_shard / 1e6, 866, delta=5)
        self.assertAlmostEqual(four.per_shard / 1e6, 447, delta=5)
        # The stages are balanced within a few percent for both geometries.
        for report in (nine, four):
            self.assertLess(abs(report.recurrent_layer - report.global_layer) / report.recurrent_layer, 0.05)

    def test_mixer_inner_width_is_independent_of_hidden_size(self) -> None:
        cfg = ASICLMConfig.from_preset("qwen3_5_4b")
        self.assertEqual(cfg.hidden_size, 2560)
        self.assertEqual(cfg.linear_value_dim, 4096)
        self.assertEqual(cfg.num_attention_heads * cfg.head_dim, 4096)

    def test_variants_resolve_to_presets(self) -> None:
        path = Path(__file__).parents[1] / "variants.json"
        cfg = load_variant(path, "index128_fp4_compress4_4b")
        self.assertEqual(cfg.hidden_size, 2560)
        self.assertEqual(cfg.retrieval_block_size, 4)
        self.assertEqual(cfg.intermediate_size, 9216)

    def test_config_from_hf_reads_text_subconfig_and_layer_types(self) -> None:
        hf = {
            "tie_word_embeddings": True,
            "text_config": {
                "vocab_size": 248320, "hidden_size": 2560, "num_hidden_layers": 32,
                "intermediate_size": 9216, "linear_num_key_heads": 16, "linear_num_value_heads": 32,
                "linear_key_head_dim": 128, "linear_value_head_dim": 128, "linear_conv_kernel_dim": 4,
                "num_attention_heads": 16, "num_key_value_heads": 4, "head_dim": 256,
                "rope_parameters": {"partial_rotary_factor": 0.25, "rope_theta": 10_000_000},
                "layer_types": (["linear_attention"] * 3 + ["full_attention"]) * 8,
            },
        }
        cfg = config_from_hf(hf)
        self.assertEqual((cfg.recurrent_every, cfg.global_layer_offset), (4, 3))
        self.assertEqual(cfg.rotary_dim, 64)
        self.assertTrue(cfg.tie_embeddings)


@unittest.skipIf(torch is None, "PyTorch not installed")
class ModelTest(unittest.TestCase):
    def setUp(self) -> None:
        torch.manual_seed(0)
        self.cfg = tiny_config()
        self.model = ASICLM(self.cfg).eval()

    def test_forward_shapes(self) -> None:
        ids = torch.randint(0, self.cfg.vocab_size, (2, 40))
        out = self.model(ids, return_hidden_states=True)
        self.assertEqual(out["logits"].shape, (2, 40, self.cfg.vocab_size))
        self.assertEqual(len(out["hidden_states"]), self.cfg.num_layers)
        for idx, state in enumerate(out["recurrent_states"]):
            if self.cfg.is_global_layer(idx):
                self.assertIsNone(state)
            else:
                self.assertIsInstance(state, DeltaState)

    def test_recurrent_layer_state_carry_matches_single_pass(self) -> None:
        layer = self.model.layers[0]
        x = torch.randn(2, 24, self.cfg.hidden_size)
        with torch.no_grad():
            full, _ = layer(x)
            first, state = layer(x[:, :10])
            second, _ = layer(x[:, 10:], state)
        torch.testing.assert_close(torch.cat((first, second), dim=1), full, atol=1e-4, rtol=1e-4)

    def test_global_layer_is_causal(self) -> None:
        layer = self.model.layers[self.cfg.global_layer_offset]
        x = torch.randn(1, 40, self.cfg.hidden_size)
        with torch.no_grad():
            base, _ = layer(x)
            perturbed, _ = layer(torch.cat((x[:, :30], torch.randn(1, 10, self.cfg.hidden_size)), dim=1))
        torch.testing.assert_close(perturbed[:, :30], base[:, :30])

    def test_global_layer_uses_retrieved_blocks(self) -> None:
        # Positions beyond the local window can only reach early tokens via retrieval.
        layer = self.model.layers[self.cfg.global_layer_offset]
        x = torch.randn(1, 40, self.cfg.hidden_size)
        with torch.no_grad():
            base, _ = layer(x)
            changed, _ = layer(torch.cat((torch.randn(1, 4, self.cfg.hidden_size), x[:, 4:]), dim=1))
        self.assertFalse(torch.allclose(changed[:, -1], base[:, -1]))

    def test_hf_key_rename_and_round_trip_load(self) -> None:
        source = ASICLM(self.cfg)
        hf_state = {"model.language_model." + key: value for key, value in source.state_dict().items()
                    if not key.endswith(("index_q.weight", "index_k.weight", "lm_head.weight"))}
        hf_state["lm_head.weight"] = source.state_dict()["lm_head.weight"]
        hf_state["model.visual.blocks.0.weight"] = torch.zeros(1)
        self.assertIsNone(rename_hf_key("model.visual.blocks.0.weight"))
        self.assertEqual(rename_hf_key("model.language_model.layers.0.linear_attn.A_log"),
                         "layers.0.linear_attn.A_log")
        converted = convert_state_dict(hf_state)
        self.assertNotIn("visual.blocks.0.weight", converted)
        target = ASICLM(self.cfg)
        new_keys = load_into_model(target, hf_state)
        self.assertEqual(len(new_keys), 2 * (self.cfg.num_layers // self.cfg.recurrent_every))
        ids = torch.randint(0, self.cfg.vocab_size, (1, 12))
        with torch.no_grad():
            # Everything except the retrieval index was copied; compare a recurrent layer directly.
            a, _ = source.layers[0](source.embed_tokens(ids))
            b, _ = target.layers[0](target.embed_tokens(ids))
        torch.testing.assert_close(a, b)

    def test_index_init_tracks_mean_attention_logits(self) -> None:
        layer = self.model.layers[self.cfg.global_layer_offset]
        attn = layer.self_attn
        init_index_projections(layer)
        x = torch.randn(64, self.cfg.hidden_size)
        with torch.no_grad():
            nh, nkv, hd = attn.num_heads, attn.num_kv_heads, attn.head_dim
            q = attn.q_proj(x).view(64, nh, 2 * hd)[:, :, :hd].view(64, nkv, nh // nkv, hd).mean(2)
            k = attn.k_proj(x).view(64, nkv, hd)
            reference = torch.einsum("qnd,knd->qk", q, k)
            approx = attn.index_q(x) @ attn.index_k(x).T
        correlation = torch.corrcoef(torch.stack((reference.flatten(), approx.flatten())))[0, 1]
        self.assertGreater(correlation.item(), 0.9)


if __name__ == "__main__":
    unittest.main()
