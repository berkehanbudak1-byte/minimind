import os
import sys
import unittest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from model.model_minimind import (
    MiniMindConfig,
    MiniMindForCausalLM,
    benchmark_context_scaling,
    estimate_model_footprint,
    recommend_training_hyperparams,
    torch,
)


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

    def test_recommend_training_hyperparams_scales_with_vram(self):
        cfg = MiniMindConfig(hidden_size=512, num_hidden_layers=8)
        rec_low = recommend_training_hyperparams(cfg, vram_gb=6.0)
        rec_high = recommend_training_hyperparams(cfg, vram_gb=24.0)

        for key in ("batch_size", "gradient_accumulation_steps", "dtype", "learning_rate"):
            self.assertIn(key, rec_low)
            self.assertIn(key, rec_high)

        self.assertEqual(rec_low["dtype"], "float16")
        self.assertEqual(rec_high["dtype"], "bfloat16")
        self.assertGreaterEqual(rec_high["batch_size"], rec_low["batch_size"])
        self.assertLessEqual(rec_high["gradient_accumulation_steps"], rec_low["gradient_accumulation_steps"])
        self.assertGreater(rec_high["learning_rate"], 0)

    def test_benchmark_context_scaling_across_512_to_32768(self):
        cfg = MiniMindConfig(hidden_size=768, num_hidden_layers=8, num_attention_heads=8, num_key_value_heads=4)
        contexts = [512, 1024, 2048, 4096, 8192, 16384, 32768]
        bench_24 = benchmark_context_scaling(cfg, context_lengths=contexts, vram_gb=24.0)
        bench_1 = benchmark_context_scaling(cfg, context_lengths=contexts, vram_gb=1.0)

        self.assertEqual([r["context_length"] for r in bench_24["results"]], contexts)
        self.assertEqual(bench_24["max_supported_context"], 32768)
        self.assertLess(bench_1["max_supported_context"], 32768)

        rows = bench_24["results"]
        for prev, curr in zip(rows, rows[1:]):
            self.assertGreater(curr["kv_cache_mb_per_seq"], prev["kv_cache_mb_per_seq"])
            self.assertGreater(curr["prefill_total_mb_batch1"], prev["prefill_total_mb_batch1"])
            self.assertGreater(curr["decode_total_mb_batch1"], prev["decode_total_mb_batch1"])
            self.assertLessEqual(curr["max_batch_size"], prev["max_batch_size"])
            self.assertGreaterEqual(curr["max_decode_batch_size"], curr["max_prefill_batch_size"])

        # KV cache at 32768 should be 64x KV cache at 512
        self.assertAlmostEqual(rows[-1]["kv_cache_mb_per_seq"] / rows[0]["kv_cache_mb_per_seq"], 64.0, places=3)
        self.assertFalse(bench_1["results"][-1]["fits"])
        self.assertEqual(bench_1["results"][-1]["max_batch_size"], 0)

        # Non-flash attention increases prefill activation memory quadratically with context
        cfg_no_flash = MiniMindConfig(hidden_size=768, num_hidden_layers=8, flash_attn=False)
        bench_no_flash = benchmark_context_scaling(cfg_no_flash, context_lengths=[512, 8192], vram_gb=24.0)
        self.assertGreater(
            bench_no_flash["results"][1]["prefill_activation_mb_per_seq"],
            bench_24["results"][4]["prefill_activation_mb_per_seq"],
        )


if __name__ == "__main__":
    unittest.main()
