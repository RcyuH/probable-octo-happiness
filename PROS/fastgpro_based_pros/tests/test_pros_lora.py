import sys
import types
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock


sys.path.insert(0, str(Path(__file__).resolve().parents[2]))


class _Parameter:
    def __init__(self, requires_grad=True):
        self.requires_grad = requires_grad


class _TargetModel:
    def __init__(self):
        self.params = [_Parameter(), _Parameter()]
        self.eval_calls = 0
        self.loaded_adapters = []

    def to(self, device):
        del device
        return self

    def eval(self):
        self.eval_calls += 1
        return self

    def parameters(self):
        return iter(self.params)

    def load_adapter(self, path, adapter_name):
        self.loaded_adapters.append((path, adapter_name))

    @staticmethod
    def print_trainable_parameters():
        return None


class _DraftModel:
    def __init__(self):
        self.params = [_Parameter()]

    def parameters(self):
        return iter(self.params)


class _ModelWrapper:
    def __init__(self, draft_config, target_model):
        self.draft_config = draft_config
        self.target_model = target_model
        self.draft_model = _DraftModel()

    def to(self, device):
        del device
        return self

    @staticmethod
    def load_model(path):
        del path


class _Tokenizer:
    pad_token_id = 0
    eos_token_id = 2
    pad_token = "<pad>"
    eos_token = "</s>"


class ProsLoraIntegrationTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        try:
            import numpy  # noqa: F401
        except ModuleNotFoundError:
            raise unittest.SkipTest("numpy is required to import pros_trainer")

    @staticmethod
    def _imports():
        from fastgpro_based_pros.pros_config import ProsConfig
        from fastgpro_based_pros.pros_trainer import ProsTrainer

        return ProsConfig, ProsTrainer

    @staticmethod
    def _transformers_module(target_model):
        module = types.ModuleType("transformers")

        class AutoConfig:
            @staticmethod
            def from_pretrained(path):
                del path
                return SimpleNamespace(
                    model_type="qwen2",
                    rope_scaling={"type": "unused"},
                    num_hidden_layers=12,
                )

        class AutoModelForCausalLM:
            @staticmethod
            def from_pretrained(path, **kwargs):
                del path, kwargs
                return target_model

        class AutoTokenizer:
            @staticmethod
            def from_pretrained(path, **kwargs):
                del path, kwargs
                return _Tokenizer()

        module.AutoConfig = AutoConfig
        module.AutoModelForCausalLM = AutoModelForCausalLM
        module.AutoTokenizer = AutoTokenizer
        return module

    def _bare_trainer(self, cfg):
        _, ProsTrainer = self._imports()
        trainer = object.__new__(ProsTrainer)
        trainer.cfg = cfg
        trainer.device = "cpu"
        trainer.Model = _ModelWrapper
        return trainer

    def test_lora_config_receives_custom_target_modules_bias_and_adapter_path(self):
        ProsConfig, _ = self._imports()
        cfg = ProsConfig(
            model_dir="fake-model",
            use_lora=True,
            train_draft=False,
            lora_r=17,
            lora_alpha=29,
            lora_dropout=0.125,
            lora_target_modules=["q_proj", "custom_proj"],
            lora_bias="lora_only",
            load_lora_path="fake-adapter",
        )
        target = _TargetModel()
        trainer = self._bare_trainer(cfg)
        transformers = self._transformers_module(target)
        peft = types.ModuleType("peft")
        lora_constructor = mock.Mock(side_effect=lambda **kwargs: SimpleNamespace(**kwargs))
        get_peft_model = mock.Mock(side_effect=lambda model, config: model)
        peft.LoraConfig = lora_constructor
        peft.TaskType = SimpleNamespace(CAUSAL_LM="causal-lm")
        peft.get_peft_model = get_peft_model

        with mock.patch.dict(sys.modules, {"transformers": transformers, "peft": peft}):
            trainer._load_models()

        lora_constructor.assert_called_once_with(
            task_type="causal-lm",
            r=17,
            lora_alpha=29,
            lora_dropout=0.125,
            target_modules=["q_proj", "custom_proj"],
            bias="lora_only",
        )
        get_peft_model.assert_called_once()
        self.assertEqual(target.loaded_adapters, [("fake-adapter", "default")])
        self.assertEqual(target.eval_calls, 1)

    def test_disabled_lora_does_not_touch_peft_and_keeps_target_trainable(self):
        ProsConfig, _ = self._imports()
        cfg = ProsConfig(
            model_dir="fake-model",
            generation_backend="target",
            verification_capacity=1,
            use_lora=False,
            train_draft=False,
        )
        target = _TargetModel()
        trainer = self._bare_trainer(cfg)
        transformers = self._transformers_module(target)
        peft = types.ModuleType("peft")
        peft.LoraConfig = mock.Mock(side_effect=AssertionError("PEFT must not be used when LoRA is disabled"))
        peft.TaskType = SimpleNamespace(CAUSAL_LM="causal-lm")
        peft.get_peft_model = mock.Mock(side_effect=AssertionError("PEFT must not be used when LoRA is disabled"))

        with mock.patch.dict(sys.modules, {"transformers": transformers, "peft": peft}):
            trainer._load_models()

        peft.LoraConfig.assert_not_called()
        peft.get_peft_model.assert_not_called()
        self.assertTrue(all(parameter.requires_grad for parameter in target.params))
        self.assertTrue(all(not parameter.requires_grad for parameter in trainer.model.draft_model.params))

    def test_target_optimizer_parameter_filter_excludes_frozen_parameters(self):
        ProsConfig, _ = self._imports()
        trainer = self._bare_trainer(ProsConfig(generation_backend="target", verification_capacity=1))
        trainable = _Parameter(requires_grad=True)
        frozen = _Parameter(requires_grad=False)
        trainer.model = SimpleNamespace(
            target_model=SimpleNamespace(parameters=lambda: iter([frozen, trainable]))
        )

        self.assertEqual(trainer._trainable_target_parameters(), [trainable])


if __name__ == "__main__":
    unittest.main()
