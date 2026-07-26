import os

from baselines.defenses import (
    PerplexityFilter, SmoothLLM, SelfReminder,
    ParaphraseDefense, AttentionSharpening,
    HijackingSuppression, BaseDefense,
)
from utils.config_utils import AttackConfig


def _defense_type_key(defense_type: str) -> str:
    return defense_type.lower().replace("-", "").replace("_", "")


def _needs_eager_attention(config: AttackConfig) -> bool:
    if not config.defenses:
        return False
    eager_types = {"attentionsharpening", "hijackingsuppression"}
    return any(_defense_type_key(dc.type) in eager_types for dc in config.defenses)


def _load_pipeline_model(model_path: str, device: str, config: AttackConfig):
    import torch
    from transformers import AutoConfig, AutoModelForCausalLM

    model_config = AutoConfig.from_pretrained(model_path, trust_remote_code=True)
    model_cls = AutoModelForCausalLM
    if str(getattr(model_config, "model_type", "")).startswith("qwen3_5"):
        from transformers import AutoModelForMultimodalLM

        model_cls = AutoModelForMultimodalLM

    kwargs = dict(trust_remote_code=True, torch_dtype=torch.bfloat16)
    if _needs_eager_attention(config):
        kwargs["attn_implementation"] = "eager"
        print(
            "[grading_attack] Using attn_implementation=eager for attention hook defense",
            flush=True,
        )
    model = model_cls.from_pretrained(model_path, **kwargs)
    return model.to(device)


def _build_defenses(config: AttackConfig, model=None, tokenizer=None):
    """根据配置构建防御模块列表"""
    if not config.defenses:
        return []
    defenses = []
    device = os.environ.get("GRADING_ATTACK_DEVICE", config.params.get("device", "cuda"))
    for dc in config.defenses:
        t = _defense_type_key(dc.type)
        if t == "perplexityfilter":
            defenses.append(PerplexityFilter(
                model=model, tokenizer=tokenizer,
                threshold=dc.params.get("threshold", 1000.0),
                device=device,
            ))
        elif t == "smoothllm":
            defenses.append(SmoothLLM(
                num_copies=dc.params.get("num_copies", 5),
                perturb_rate=dc.params.get("perturb_rate", None),
            ))
        elif t == "selfreminder":
            defenses.append(SelfReminder(
                reminder=dc.params.get("reminder", None),
            ))
        elif t == "paraphrasedefense" or t == "paraphrase":
            defenses.append(ParaphraseDefense())
        elif t == "attentionsharpening":
            defenses.append(AttentionSharpening(
                temperature=dc.params.get("temperature", 0.5),
                layers=dc.params.get("layers", "all"),
            ))
        elif t == "hijackingsuppression":
            defenses.append(HijackingSuppression(
                beta=dc.params.get("beta", 0.1),
                top_fraction=dc.params.get("top_fraction", 0.01),
                layers=dc.params.get("layers", "all"),
            ))
        else:
            raise ValueError(f"Unknown defense type: {dc.type}")
    return defenses


class GradingAttack:
    def __init__(self, config: AttackConfig):
        self.config = config
        if config.pipeline_mode:
            # pipeline 模式: 在 pipeline.py 中统一处理
            self.attack = None
        elif config.attack_method.lower() == "gcg":
            from baselines.gcg.gcg import GCG

            self.attack = GCG(config)
        elif config.attack_method.lower() == "roleplay":
            from baselines.roleplay.roleplay import RolePlay

            self.attack = RolePlay(config)
        else:
            raise ValueError(f"Not supported attack method {config.attack_method}")

    def run(self):
        if self.config.pipeline_mode:
            from pipeline import GradingDefensePipeline
            from transformers import AutoTokenizer

            device = os.environ.get(
                "GRADING_ATTACK_DEVICE",
                self.config.params.get("device", "cuda"),
            )
            model = _load_pipeline_model(
                self.config.model_config.path,
                device,
                self.config,
            )
            tokenizer = AutoTokenizer.from_pretrained(
                self.config.model_config.path,
                trust_remote_code=True,
            )
            defenses = _build_defenses(self.config, model, tokenizer)
            pipeline = GradingDefensePipeline(
                config=self.config,
                model=model,
                tokenizer=tokenizer,
                defenses=defenses,
            )
            pipeline.run()
        elif self.attack is not None:
            self.attack.run()
        else:
            raise RuntimeError(
                "pipeline_mode=True requires defenses configured, "
                "or set pipeline_mode=False for pure attack."
            )
