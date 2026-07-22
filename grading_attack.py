from baselines.defenses import (
    PerplexityFilter, SmoothLLM, SelfReminder,
    ParaphraseDefense, BaseDefense,
)
from utils.config_utils import AttackConfig


def _build_defenses(config: AttackConfig, model=None, tokenizer=None):
    """根据配置构建防御模块列表"""
    if not config.defenses:
        return []
    defenses = []
    device = config.params.get("device", "cuda")
    for dc in config.defenses:
        t = dc.type.lower().replace("-", "").replace("_", "")
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
            import torch
            from transformers import AutoModelForCausalLM, AutoTokenizer

            device = self.config.params.get("device") or (
                "cuda" if torch.cuda.is_available() else "cpu"
            )
            print(f"[grading_attack] Using device: {device}, dtype: bfloat16", flush=True)
            model = AutoModelForCausalLM.from_pretrained(
                self.config.model_config.path,
                trust_remote_code=True,
                torch_dtype=torch.bfloat16,
            ).to(device)
            model.generation_config.temperature = None
            model.generation_config.top_p = None
            model.generation_config.top_k = None
            model.generation_config.do_sample = False
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
