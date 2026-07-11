import json
import os
from pathlib import Path
import sys
import tempfile
import unittest
from unittest import mock

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from fastgpro_based_pros.pros_config import ProsConfig


DEFAULT_LORA_MODULES = [
    "q_proj",
    "k_proj",
    "v_proj",
    "o_proj",
    "gate_proj",
    "up_proj",
    "down_proj",
]


class ProsConfigTest(unittest.TestCase):
    def _write_json(self, directory, data, name="config.json"):
        path = Path(directory) / name
        path.write_text(json.dumps(data), encoding="utf-8")
        return path

    def test_new_defaults(self):
        cfg = ProsConfig()

        self.assertEqual(cfg.verification_capacity, 160)
        self.assertEqual(cfg.max_draft_token_length, 5)
        self.assertEqual(cfg.min_draft_token_length, 3)
        self.assertEqual(cfg.max_draft_k, 8)
        self.assertEqual(cfg.max_verification_num, 160)
        self.assertEqual(cfg.draft_token_length_c, 0.75)
        self.assertEqual(cfg.lora_target_modules, DEFAULT_LORA_MODULES)
        self.assertEqual(cfg.lora_bias, "none")
        self.assertTrue(cfg.use_tensorboard)
        self.assertEqual(cfg.tensorboard_log_dir, "")
        self.assertEqual(cfg.reward_debug_sample_count, 5)
        self.assertEqual(cfg.reward_debug_sample_chars, 800)

    def test_lora_default_list_is_not_shared(self):
        first = ProsConfig()
        second = ProsConfig()
        first.lora_target_modules.append("custom_proj")
        self.assertEqual(second.lora_target_modules, DEFAULT_LORA_MODULES)

    def test_multitask_cli_fields_are_coerced(self):
        cfg = ProsConfig.from_args(
            [
                "--dry-run",
                "true",
                "--task-config",
                "FastGRPO/configs/multitask_rlvr.example.json",
                "--task-samples-per-epoch",
                "17",
                "--generation-backend",
                "target",
                "--train-draft",
                "false",
            ]
        )
        self.assertTrue(cfg.dry_run)
        self.assertEqual(cfg.task_config, "FastGRPO/configs/multitask_rlvr.example.json")
        self.assertEqual(cfg.task_samples_per_epoch, 17)
        self.assertEqual(cfg.generation_backend, "target")
        self.assertFalse(cfg.train_draft)

    def test_all_speculative_controls_are_available_from_cli(self):
        cfg = ProsConfig.from_args(
            [
                "--verification-capacity",
                "256",
                "--max-draft-token-length",
                "7",
                "--min-draft-token-length",
                "2",
                "--max-draft-k",
                "12",
                "--max-verification-num",
                "128",
                "--draft-token-length-c",
                "0.625",
            ]
        )
        self.assertEqual(cfg.verification_capacity, 256)
        self.assertEqual(cfg.max_draft_token_length, 7)
        self.assertEqual(cfg.min_draft_token_length, 2)
        self.assertEqual(cfg.max_draft_k, 12)
        self.assertEqual(cfg.max_verification_num, 128)
        self.assertEqual(cfg.draft_token_length_c, 0.625)

    def test_boolean_cli_values_are_strict_and_case_insensitive(self):
        truthy = ("1", "true", "TRUE", "yes", "y", "on")
        falsy = ("0", "false", "FALSE", "no", "n", "off")
        for raw in truthy:
            with self.subTest(raw=raw):
                self.assertTrue(ProsConfig.from_args(["--dry-run", raw]).dry_run)
        for raw in falsy:
            with self.subTest(raw=raw):
                self.assertFalse(ProsConfig.from_args(["--dry-run", raw]).dry_run)

        with self.assertRaisesRegex(ValueError, "invalid boolean value"):
            ProsConfig.from_args(["--dry-run", "truthy-ish"])
        with self.assertRaisesRegex(ValueError, "invalid boolean value"):
            ProsConfig(use_tensorboard="sometimes")

    def test_json_boolean_strings_are_coerced_without_truthiness_bug(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = self._write_json(
                tmp,
                {"dry_run": "false", "train_draft": "YES", "use_tensorboard": 0},
            )
            cfg = ProsConfig.from_json(path)
        self.assertFalse(cfg.dry_run)
        self.assertTrue(cfg.train_draft)
        self.assertFalse(cfg.use_tensorboard)

    def test_cli_lora_modules_are_comma_separated_and_trimmed(self):
        cfg = ProsConfig.from_args(
            [
                "--lora-target-modules",
                " q_proj, k_proj, , v_proj ",
                "--lora-bias",
                "lora_only",
            ]
        )
        self.assertEqual(cfg.lora_target_modules, ["q_proj", "k_proj", "v_proj"])
        self.assertEqual(cfg.lora_bias, "lora_only")

    def test_json_lora_modules_require_and_preserve_an_array(self):
        with tempfile.TemporaryDirectory() as tmp:
            valid_path = self._write_json(
                tmp,
                {"lora_target_modules": ["q_proj", "custom_proj"], "lora_bias": "all"},
                "valid.json",
            )
            cfg = ProsConfig.from_json(valid_path)
            self.assertEqual(cfg.lora_target_modules, ["q_proj", "custom_proj"])
            self.assertEqual(cfg.lora_bias, "all")

            invalid_path = self._write_json(
                tmp,
                {"lora_target_modules": "q_proj,v_proj"},
                "invalid.json",
            )
            with self.assertRaisesRegex(TypeError, "must use a JSON array"):
                ProsConfig.from_json(invalid_path)

    def test_json_and_cli_override_precedence_with_environment_expansion(self):
        with tempfile.TemporaryDirectory() as tmp, mock.patch.dict(
            os.environ,
            {"PROS_TEST_MODEL": "/models/base", "PROS_TEST_CONFIG_DIR": tmp},
        ):
            path = self._write_json(
                tmp,
                {
                    "model_dir": "${PROS_TEST_MODEL}/target",
                    "batch_size": 2,
                    "repeated_generate_nums": 3,
                    "verification_capacity": 12,
                    "lora_target_modules": ["json_proj"],
                    "use_tensorboard": False,
                },
            )
            cfg = ProsConfig.from_args(
                [
                    "--config",
                    "${PROS_TEST_CONFIG_DIR}/config.json",
                    "--verification-capacity",
                    "24",
                    "--lora-target-modules",
                    "cli_q,cli_v",
                    "--use-tensorboard",
                    "true",
                ]
            )

        self.assertEqual(cfg.model_dir, "/models/base/target")
        self.assertEqual(cfg.batch_size, 2)
        self.assertEqual(cfg.repeated_generate_nums, 3)
        self.assertEqual(cfg.verification_capacity, 24)
        self.assertEqual(cfg.lora_target_modules, ["cli_q", "cli_v"])
        self.assertTrue(cfg.use_tensorboard)

    def test_resolved_config_round_trip_serializes_lora_modules_as_list(self):
        cfg = ProsConfig(
            generation_backend="target",
            verification_capacity=1,
            lora_target_modules=["q_proj", "v_proj"],
            tensorboard_log_dir="logs/tensorboard",
            reward_debug_sample_count=3,
            reward_debug_sample_chars=123,
        )
        with tempfile.TemporaryDirectory() as tmp:
            resolved_path = cfg.save(tmp)
            raw = json.loads(resolved_path.read_text(encoding="utf-8"))
            restored = ProsConfig.from_json(resolved_path)

        self.assertIsInstance(raw["lora_target_modules"], list)
        self.assertEqual(raw["lora_target_modules"], ["q_proj", "v_proj"])
        self.assertEqual(restored.to_dict(), cfg.to_dict())

    def test_capacity_exactly_at_speculative_threshold_passes(self):
        cfg = ProsConfig(
            batch_size=3,
            repeated_generate_nums=5,
            verification_capacity=30,
        )
        self.assertEqual(cfg.verification_capacity, 30)
        self.assertIs(cfg.validate(), cfg)

    def test_capacity_one_below_speculative_threshold_fails(self):
        with self.assertRaisesRegex(ValueError, "requires at least 30"):
            ProsConfig(
                batch_size=3,
                repeated_generate_nums=5,
                verification_capacity=29,
            )

    def test_target_backend_bypasses_unused_minimum_capacity(self):
        cfg = ProsConfig(
            generation_backend="target",
            batch_size=100,
            repeated_generate_nums=100,
            verification_capacity=1,
        )
        self.assertEqual(cfg.verification_capacity, 1)

    def test_invalid_scalar_values_are_rejected_centrally(self):
        invalid_cases = (
            ({"batch_size": 0}, "batch_size"),
            ({"repeated_generate_nums": 0}, "repeated_generate_nums"),
            ({"verification_capacity": 0}, "verification_capacity"),
            ({"max_draft_k": 0}, "max_draft_k"),
            ({"min_draft_token_length": 0}, "min_draft_token_length"),
            ({"max_draft_token_length": 0}, "max_draft_token_length"),
            (
                {"min_draft_token_length": 6, "max_draft_token_length": 5},
                "min_draft_token_length",
            ),
            ({"max_verification_num": 1}, "max_verification_num"),
            ({"draft_token_length_c": 0}, "draft_token_length_c"),
            ({"generation_backend": "mystery"}, "generation_backend"),
            ({"reward_debug_sample_count": -1}, "reward_debug_sample_count"),
            ({"reward_debug_sample_chars": 0}, "reward_debug_sample_chars"),
        )
        for updates, message in invalid_cases:
            with self.subTest(updates=updates):
                with self.assertRaisesRegex(ValueError, message):
                    ProsConfig(**updates)

    def test_invalid_lora_values_are_rejected_centrally(self):
        invalid_cases = (
            ({"lora_r": 0}, "lora_r"),
            ({"lora_alpha": 0}, "lora_alpha"),
            ({"lora_dropout": -0.01}, "lora_dropout"),
            ({"lora_dropout": 1.0}, "lora_dropout"),
            ({"lora_target_modules": []}, "lora_target_modules"),
            ({"lora_target_modules": ["  "]}, "lora_target_modules"),
            ({"lora_bias": "invalid"}, "lora_bias"),
            (
                {"use_lora": False, "load_lora_path": "adapter/checkpoint"},
                "load_lora_path requires use_lora=true",
            ),
            (
                {"use_lora": False, "beta": 0.1},
                "beta > 0 requires use_lora=true",
            ),
            (
                {"lora_bias": "all", "beta": 0.1},
                "lora_bias='none'",
            ),
        )
        for updates, message in invalid_cases:
            with self.subTest(updates=updates):
                with self.assertRaisesRegex(ValueError, message):
                    ProsConfig(**updates)

        with self.assertRaisesRegex(TypeError, "entries must be strings"):
            ProsConfig(lora_target_modules=["q_proj", 123])

        self.assertEqual(ProsConfig(beta=0.1, use_lora=True, lora_bias="none").beta, 0.1)
        self.assertEqual(
            ProsConfig(beta=0.1, objective="clipped_grpo", use_lora=False).beta,
            0.1,
        )

    def test_replace_runs_the_same_validation(self):
        cfg = ProsConfig()
        with self.assertRaisesRegex(ValueError, "lora_bias"):
            cfg.replace(lora_bias="bad-bias")

    def test_json_top_level_must_be_an_object(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = self._write_json(tmp, ["not", "an", "object"])
            with self.assertRaisesRegex(TypeError, "top level"):
                ProsConfig.from_json(path)


if __name__ == "__main__":
    unittest.main()
