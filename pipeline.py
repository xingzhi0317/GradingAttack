"""
Attack-Defense 联合评估 Pipeline。

控制流:
    data → attack (GCG/RolePlay) → defenses → evaluate → metrics

参考论文:
    - SmoothLLM (Robey et al., 2023)
    - Baseline Defenses (Jain et al., 2023)
    - GradingAttack (ICLR 2026)
"""

import json
import torch
from typing import List, Optional
from transformers import AutoModelForCausalLM, AutoTokenizer

from utils.config_utils import AttackConfig, apply_data_sampling
from utils.log_utils import GradingAttackLogger
from utils.data_utils import read_student_qa_data_from_jsonl, AttackResult, extract_grade
from eval.metrics import compute_metrics, EvalMetrics
from baselines.defenses.base import BaseDefense, DefenseRejectException


class GradingDefensePipeline:
    def __init__(self, config: AttackConfig,
                 model: AutoModelForCausalLM = None,
                 tokenizer: AutoTokenizer = None,
                 defenses: List[BaseDefense] = None):
        """
        config: AttackConfig (含 defense 配置)
        model / tokenizer: 可选外部注入，仅 GCG 方式需要
        defenses: 防御模块列表
        """
        self.config = config
        self.model = model
        self.tokenizer = tokenizer
        self.defenses = defenses or []
        self.device = config.params.get("device", "cuda")

        # 检查是否有 SmoothLLM 类需要多次推理的防御
        self.multi_gen_defenses = [d for d in self.defenses
                                   if d.requires_multiple_generations()]
        self.context_aware_defenses = any(
            d.requires_inference_context() for d in self.defenses
        )
        self._suffix_bank: Optional[List[dict]] = None

    @staticmethod
    def _insert_attack_suffix(prompt: str, suffix: str) -> str:
        end_tag = "</student_answer>"
        insert_at = prompt.rfind(end_tag)
        if insert_at == -1:
            return prompt + suffix
        return prompt[:insert_at] + suffix + prompt[insert_at:]

    @staticmethod
    def _parse_response_grade(response: str) -> Optional[str]:
        grade = extract_grade(response)
        return grade.strip().lower() if grade else None

    def _load_suffix_bank(self) -> List[dict]:
        if self._suffix_bank is not None:
            return self._suffix_bank

        bank_path = self.config.params.get("bank_path")
        if not bank_path:
            raise ValueError("gcg_suffix_bank requires params.bank_path")

        bank_limit = self.config.params.get("bank_limit")
        suffix_bank = []
        with open(bank_path, "r", encoding="utf-8") as bank_file:
            for line in bank_file:
                if not line.strip():
                    continue
                item = json.loads(line)
                if isinstance(item, str):
                    suffix = item
                    meta = {}
                else:
                    suffix = (
                        item.get("best_string")
                        or item.get("suffix")
                        or item.get("attack_suffix")
                        or item.get("gcg_suffix")
                    )
                    meta = item
                if suffix:
                    suffix_bank.append({"suffix": suffix, "meta": meta})
                if bank_limit is not None and len(suffix_bank) >= bank_limit:
                    break

        if not suffix_bank:
            raise ValueError(f"No suffixes loaded from bank: {bank_path}")
        self._suffix_bank = suffix_bank
        return suffix_bank

    def _run_suffix_bank_attack(self, prompt: str, original_resp: str):
        suffix_bank = self._load_suffix_bank()
        original_grade = self._parse_response_grade(original_resp)
        target_grade = self.config.params.get("target_grade", "correct")

        fallback_suffix = suffix_bank[0]["suffix"]
        fallback_content = self._insert_attack_suffix(prompt, fallback_suffix)
        fallback_resp = ""
        fallback_meta = suffix_bank[0]["meta"]

        for bank_index, item in enumerate(suffix_bank):
            attack_suffix = item["suffix"]
            attacked_content = self._insert_attack_suffix(prompt, attack_suffix)
            attacked_resp = self._generate([{"role": "user", "content": attacked_content}])
            attacked_grade = self._parse_response_grade(attacked_resp)

            if bank_index == 0:
                fallback_resp = attacked_resp

            if original_grade != target_grade and attacked_grade == target_grade:
                return attack_suffix, attacked_content, attacked_resp, {
                    "bank_index": bank_index,
                    "bank_meta": item["meta"],
                    "bank_success": True,
                    "original_grade": original_grade,
                    "attacked_grade": attacked_grade,
                }

        return fallback_suffix, fallback_content, fallback_resp, {
            "bank_index": 0,
            "bank_meta": fallback_meta,
            "bank_success": False,
            "original_grade": original_grade,
            "attacked_grade": self._parse_response_grade(fallback_resp),
        }

    def _install_defense_hooks(self, prompt_content: str, attack_suffix: str) -> list:
        removers = []
        for defense in self.defenses:
            if defense.requires_inference_context():
                defense.set_inference_context(
                    self.tokenizer,
                    prompt_content,
                    attack_suffix,
                )
            if defense.requires_model_hooks():
                removers.extend(defense.install_model_hooks(self.model))
        return removers

    @staticmethod
    def _remove_defense_hooks(removers: list) -> None:
        for remove in removers:
            remove()

    def _run_with_defense_hooks(
        self,
        prompt_content: str,
        attack_suffix: str,
        fn,
    ):
        removers = self._install_defense_hooks(prompt_content, attack_suffix)
        try:
            return fn()
        finally:
            self._remove_defense_hooks(removers)

    def run(self):
        config = self.config
        logger = GradingAttackLogger(config)

        # 如果未传入 model，在此加载 (RolePlay 用 vLLM 则不需)
        if self.model is None:
            self.model = AutoModelForCausalLM.from_pretrained(
                config.model_config.path,
                trust_remote_code=True,
                torch_dtype=torch.bfloat16,
            ).to(self.device)
        if self.tokenizer is None:
            self.tokenizer = AutoTokenizer.from_pretrained(
                config.model_config.path,
                trust_remote_code=True,
            )

        for data_config in config.data_config:
            data_list = read_student_qa_data_from_jsonl(data_config.path)
            data_list = apply_data_sampling(data_list, data_config)
            all_results = []

            for data in data_list:
                prompt = config.grading_template.format(
                    question=data.question,
                    solution=data.question_answer,
                    student_answer=data.student_answer,
                )

                # ── Step 1: 原始推理 ──
                messages = [{"role": "user", "content": prompt}]
                original_resp = self._generate(messages)

                # ── Step 2: 攻击推理 ──
                attacked_messages = [{"role": "user", "content": prompt}]
                attack_suffix = ""
                attack_meta = {}
                if config.attack_method.lower() == "gcg":
                    import nanogcg
                    target = config.params["target"]
                    gcg_config = nanogcg.GCGConfig(**config.params["gcg_config"])
                    gcg_result = nanogcg.run(self.model, self.tokenizer,
                                             [{"role": "user", "content": prompt}],
                                             target, gcg_config)
                    attack_suffix = gcg_result.best_string
                    attacked_messages[0]["content"] = self._insert_attack_suffix(
                        prompt, attack_suffix
                    )
                elif config.attack_method.lower() == "roleplay":
                    attack_suffix = config.params["adv_prompt"]
                    attacked_messages[0]["content"] = self._insert_attack_suffix(
                        prompt, attack_suffix
                    )
                elif config.attack_method.lower() == "gcg_suffix_bank":
                    (
                        attack_suffix,
                        attacked_messages[0]["content"],
                        attacked_resp,
                        attack_meta,
                    ) = self._run_suffix_bank_attack(prompt, original_resp)
                else:
                    raise ValueError(f"Unsupported attack method: {config.attack_method}")

                if config.attack_method.lower() != "gcg_suffix_bank":
                    attacked_resp = self._generate(attacked_messages)

                # ── Step 3: 防御推理 ──
                defended_original_resp = None
                defended_attacked_resp = None
                rejected = False

                if self.defenses:
                    try:
                        # 对有多次推理需求的 defense 做批量扰动+投票
                        if self.multi_gen_defenses:
                            defended_original_resp = self._run_with_defense_hooks(
                                prompt,
                                "",
                                lambda: self._generate_with_voting(
                                    [{"role": "user", "content": prompt}]
                                ),
                            )
                            defended_attacked_resp = self._run_with_defense_hooks(
                                attacked_messages[0]["content"],
                                attack_suffix,
                                lambda: self._generate_with_voting(attacked_messages),
                            )
                        else:
                            # pre_process 单次
                            defended_prompt = prompt
                            for d in self.defenses:
                                defended_prompt = d.pre_process(defended_prompt)

                            defended_original_resp = self._run_with_defense_hooks(
                                defended_prompt,
                                "",
                                lambda: self._generate(
                                    [{"role": "user", "content": defended_prompt}]
                                ),
                            )
                            defended_attacked_content = self._insert_attack_suffix(
                                defended_prompt, attack_suffix
                            )
                            defended_attacked_resp = self._run_with_defense_hooks(
                                defended_attacked_content,
                                attack_suffix,
                                lambda: self._generate(
                                    [{"role": "user", "content": defended_attacked_content}]
                                ),
                            )

                        # post_process
                        for d in self.defenses:
                            defended_original_resp = d.post_process(defended_original_resp or "")
                            defended_attacked_resp = d.post_process(defended_attacked_resp or "")

                    except DefenseRejectException:
                        rejected = True

                # ── Step 4: 记录结果 ──
                result = AttackResult(
                    student_qa_data=data,
                    original_response=original_resp,
                    attacked_response=attacked_resp,
                    meta={
                        "defended_original_response": defended_original_resp,
                        "defended_attacked_response": defended_attacked_resp,
                        "rejected": rejected,
                        "attack_suffix": attack_suffix,
                        "attack": attack_meta,
                    }
                )
                logger.result(result.as_dict())
                all_results.append(result.as_dict())

            # ── Step 5: 计算指标 ──
            if all_results:
                flat_results = []
                for r in all_results:
                    flat_r = {
                        "student_qa_data": r["student_qa_data"],
                        "original_response": r["original_response"],
                        "attacked_response": r["attacked_response"],
                        "defended_original_response": r.get("meta", {}).get(
                            "defended_original_response"),
                        "defended_attacked_response": r.get("meta", {}).get(
                            "defended_attacked_response"),
                    }
                    flat_results.append(flat_r)

                metrics: EvalMetrics = compute_metrics(flat_results)
                logger.info(
                    f"[{config.model_config.name}] [{data_config.name}] "
                    f"A_before={metrics.accuracy_before:.4f} "
                    f"ASR={metrics.asr:.4f} "
                    f"ASR_defended={metrics.asr_defended:.4f} "
                    f"CAS={metrics.cas:.4f} "
                    f"CAS_defended={metrics.cas_defended:.4f}"
                )

            logger.info(
                f"[MODEL] {config.model_config.name}  "
                + f"[DATASET] {data_config.name}  "
                + f"Finished"
            )

    def _generate(self, messages: list) -> str:
        inputs = self.tokenizer.apply_chat_template(
            messages, add_generation_prompt=True, return_tensors="pt"
        )
        if hasattr(inputs, "to") and not isinstance(inputs, torch.Tensor):
            inputs = inputs.to(self.device)
            input_ids = inputs["input_ids"]
            generate_kwargs = dict(inputs)
        else:
            input_ids = inputs.to(self.device)
            generate_kwargs = {"input_ids": input_ids}

        outputs = self.model.generate(
            **generate_kwargs,
            do_sample=False,
            max_new_tokens=self.config.generation_config.max_tokens,
            temperature=self.config.generation_config.temperature,
        )
        return self.tokenizer.batch_decode(
            outputs[:, input_ids.shape[1]:], skip_special_tokens=True
        )[0]

    def _generate_with_voting(self, messages: list) -> str:
        """对 SmoothLLM 类防御：生成多个扰动版本，多数投票"""
        prompt = messages[0]["content"]
        all_variants = [prompt]
        for d in self.multi_gen_defenses:
            variants = d.generate_variants(prompt)
            all_variants = variants  # 取最后一个 multi_gen 的变体

        responses = []
        for variant in all_variants:
            msgs = [{"role": "user", "content": variant}]
            resp = self._generate(msgs)
            responses.append(resp)

        # 多数投票：从所有响应中提取 grade
        from collections import Counter
        grades = []
        for r in responses:
            g = __import__('utils.data_utils', fromlist=['extract_grade']).extract_grade(r)
            grades.append(g)
        counter = Counter(g for g in grades if g is not None)
        if counter:
            majority_grade = counter.most_common(1)[0][0]
            # 返回投票最多的 grade 对应的完整响应
            # 若没有匹配的，返回第一个非 None 响应
            for i, g in enumerate(grades):
                if g == majority_grade:
                    return responses[i]
        return responses[0] if responses else ""
