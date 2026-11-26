import os
import sys
import unittest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from model.model_minimind import MiniMindConfig, MiniMindForCausalLM, estimate_model_footprint, torch


class TestMiniMindFootprint(unittest.TestCase):
    def test_dense_parameter_count_and_kv_cache_vram(self):
        cfg = MiniMindConfig(
            hidden_size=128,
            num_hidden_layers=2,
            num_attention_heads=4,
            num_key_value_heads=2,
            intermediate_size=256,
            vocab_size=512,
            max_position_embeddings=256,
            use_moe=False,
        )
        est = estimate_model_footprint(cfg, batch_size=2, seq_len=128, bytes_per_param=2)

        # Expected dense parameter count:
        # embed_tokens (tied with lm_head): 512 * 128 = 65536
        # final norm: 128
        # per layer (2 layers):
        #   attn: q(128*128) + k(128*64) + v(128*64) + o(128*128) + q_norm(32) + k_norm(32) = 49152 + 64 = 49216
        #   norms: 2 * 128 = 256
        #   ffn: 3 * 128 * 256 = 98304
        #   layer total = 147776 -> 2 layers = 295552
        # total = 65536 + 128 + 295552 = 361216
        self.assertEqual(est["total_params"], 361216)
        self.assertEqual(est["active_params"], 361216)
        self.assertFalse(est["use_moe"])

        if torch is not None:
            model = MiniMindForCausalLM(cfg)
            actual_params = sum(p.numel() for p in model.parameters())
            self.assertEqual(est["total_params"], actual_params)

        # KV cache: 2 (K+V) * 2 layers * 2 batch * 128 seq * 2 kv_heads * 32 head_dim * 2 bytes = 131072 bytes = 0.125 MB
        self.assertAlmostEqual(est["kv_cache_mb"], 0.125, places=4)
        self.assertEqual(est["clamped_seq_len"], 128)

    def test_moe_parameter_count_and_active_subset(self):
        cfg = MiniMindConfig(
            hidden_size=128,
            num_hidden_layers=2,
            num_attention_heads=4,
            num_key_value_heads=2,
            intermediate_size=256,
            vocab_size=512,
            max_position_embeddings=256,
            use_moe=True,
            num_experts=4,
            num_experts_per_tok=1,
        )
        est = estimate_model_footprint(cfg, batch_size=1, seq_len=64)

        # MoE per layer adds router gate (128 * 4 = 512) + 4 * 98304 = 393728 total FFN, active FFN = 512 + 1 * 98304 = 98816
        self.assertGreater(est["total_params"], est["active_params"])
        self.assertEqual(est["total_params"] - est["active_params"], 2 * 3 * (3 * 128 * 256))
        self.assertTrue(est["use_moe"])

        if torch is not None:
            model = MiniMindForCausalLM(cfg)
            actual_params = sum(p.numel() for p in model.parameters())
            self.assertEqual(est["total_params"], actual_params)

    def test_seq_len_and_token_generation_clamping(self):
        cfg = MiniMindConfig(
            hidden_size=64,
            num_hidden_layers=2,
            num_attention_heads=4,
            num_key_value_heads=2,
            vocab_size=128,
            max_position_embeddings=64,
            tie_word_embeddings=False,
        )
        est_high = estimate_model_footprint(cfg, batch_size=1, seq_len=99999)
        self.assertEqual(est_high["clamped_seq_len"], 64)

        est_low = estimate_model_footprint(cfg, batch_size=0, seq_len=-10)
        self.assertEqual(est_low["clamped_seq_len"], 1)
        self.assertEqual(est_low["batch_size"], 1)


if __name__ == "__main__":
    unittest.main()
