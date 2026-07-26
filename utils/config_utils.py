import os
import random
import yaml
from typing import Optional, Dict, List
from dataclasses import dataclass, asdict, field
from collections.abc import Iterable


@dataclass
class ModelConfig:
    name: Optional[str] = None
    path: Optional[str] = None


@dataclass
class DataConfig:
    name: Optional[str] = None
    path: Optional[str] = None
    max_samples: Optional[int] = None
    random_seed: Optional[int] = None
    sample_offset: int = 0


def apply_data_sampling(data_list: list, data_config: DataConfig) -> list:
    indexed = list(enumerate(data_list))
    offset = max(data_config.sample_offset, 0)
    if data_config.max_samples is None:
        sampled = indexed[offset:]
    elif data_config.random_seed is not None:
        sampled = list(indexed)
        random.Random(data_config.random_seed).shuffle(sampled)
        sampled = sampled[offset:offset + data_config.max_samples]
    else:
        sampled = indexed[offset:offset + data_config.max_samples]

    for source_index, item in sampled:
        setattr(item, "_source_index", source_index)
    return [item for _, item in sampled]


@dataclass
class GenerationConfig:
    temperature: float = 0.01
    max_tokens: int = 1024

    def as_dict(self):
        return asdict(self)


@dataclass
class LogConfig:
    log_dir: str = "./logs"
    result_dir: str = "./result"


@dataclass
class DefenseConfig:
    type: str = ""
    params: Dict = None

    def __post_init__(self):
        if self.params is None:
            self.params = {}


@dataclass
class AttackConfig:
    name: Optional[str] = None
    model_config: Optional[ModelConfig] = None
    data_config: Optional[List[DataConfig]] = None
    generation_config: GenerationConfig = field(default_factory=GenerationConfig)
    log_config: LogConfig = field(default_factory=LogConfig)
    attack_method: str = "GCG"
    params: Optional[Dict] = None
    grading_template: Optional[str] = None
    defenses: Optional[List[DefenseConfig]] = None
    pipeline_mode: bool = False


def parse_config(path: str) -> AttackConfig:
    with open(path, "r", encoding="utf-8") as config_file:
        config_dict = yaml.safe_load(config_file)
    
    if "name" in config_dict:
        name = config_dict["name"]
    else:
        name = os.path.split(os.path.basename(path))[0]
    
    model_config = ModelConfig(**config_dict["model"])
    
    if isinstance(config_dict["data"], Iterable):
        data_config = [DataConfig(**d) for d in config_dict["data"]]
    else:
        data_config = [DataConfig(**config_dict["data"])]

    if "generation" in config_dict:
        generation_config = GenerationConfig(**config_dict["generation"])
    else:
        generation_config = GenerationConfig()

    if "log" in config_dict:
        log_config = LogConfig(**config_dict["log"])
    else:
        log_config = LogConfig()

    if "params" in config_dict:
        params = config_dict["params"]
    else:
        params = {}
    
    attack_method = config_dict["method"]
    grading_template = config_dict["grading_template"]


    defenses = None
    if "defenses" in config_dict and config_dict["defenses"]:
        defenses = [DefenseConfig(**d) for d in config_dict["defenses"]]

    pipeline_mode = config_dict.get("pipeline_mode", False)

    return AttackConfig(
        name=name,
        model_config=model_config,
        data_config=data_config,
        generation_config=generation_config,
        log_config=log_config,
        attack_method=attack_method,
        params=params,
        grading_template=grading_template,
        defenses=defenses,
        pipeline_mode=pipeline_mode,
    )
