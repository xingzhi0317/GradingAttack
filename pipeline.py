"""
Attack-Defense 联合评估 Pipeline。

控制流:
    data → attack (GCG/RolePlay) → defenses → evaluate → metrics

参考论文:
    - SmoothLLM (Robey et al., 2023)
    - Baseline Defenses (Jain et al., 2023)
    - GradingAttack (ICLR 2026)
"""

import torch
import random
from copy import deepcopy
from typing import List, Optional
from transformers import AutoModelForCausalLM, AutoTokenizer

from utils.config_utils import AttackConfig
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
        self.device = config.params.get("device") or (
            "cuda" if torch.cuda.is_available() else "cpu"
        )

        # 检查是否有 SmoothLLM 类需要多次推理的防御
        self.multi_gen_defenses = [d for d in self.defenses
                                   if d.requires_multiple_generations()]

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
            if data_config.max_samples and data_config.max_samples < len(data_list):
                rng = random.Random(data_config.random_seed)
                data_list = rng.sample(data_list, data_config.max_samples)
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
                attack_meta = {}
                attacked_resp = None
                if config.attack_method.lower() == "gcg":
                    attacked_messages, attacked_resp, attack_meta = self._run_gcg_attack(
                        prompt, original_resp
                    )
                elif config.attack_method.lower() == "roleplay":
                    attacked_messages[0]["content"] = prompt + config.params["adv_prompt"]
                    attack_meta = {"adv_prompt": config.params["adv_prompt"]}

                if attacked_resp is None:
                    attacked_resp = self._generate(attacked_messages)

                # ── Step 3: 防御推理 ──
                defended_original_resp = None
                defended_attacked_resp = None
                rejected = False

                if self.defenses:
                    try:
                        # 对有多次推理需求的 defense 做批量扰动+投票
                        if self.multi_gen_defenses:
                            defended_original_resp = self._generate_with_voting(
                                [{"role": "user", "content": prompt}]
                            )
                            defended_attacked_resp = self._generate_with_voting(
                                attacked_messages
                            )
                        else:
                            # pre_process 单次
                            defended_prompt = prompt
                            for d in self.defenses:
                                defended_prompt = d.pre_process(defended_prompt)

                            defended_original_resp = self._generate(
                                [{"role": "user", "content": defended_prompt}]
                            )
                            defended_attacked_resp = self._generate(
                                [{"role": "user", "content": defended_prompt
                                  + (attacked_messages[0]["content"][len(prompt):])}]
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
                        **attack_meta,
                        "defended_original_response": defended_original_resp,
                        "defended_attacked_response": defended_attacked_resp,
                        "rejected": rejected,
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
                    f"QWK_clean={metrics.qwk_clean:.4f} "
                    f"QWK_attack={metrics.qwk_attack:.4f} "
                    f"ASR={metrics.asr:.4f} "
                    f"ASR_defended={metrics.asr_defended:.4f} "
                    f"CAS={metrics.cas:.4f} "
                    f"CAS_defended={metrics.cas_defended:.4f}"
                )

                # ── Step 6: 保存指标 + 配置摘要 ──
                import json
                summary = metrics.to_dict()
                summary["config"] = {
                    "name": config.name,
                    "attack_method": config.attack_method,
                    "model": config.model_config.name,
                    "model_path": config.model_config.path,
                    "template": config.template if hasattr(config, "template") else "ci",
                    "datasets": [d.name for d in config.data_config],
                    "defenses": [d.__class__.__name__ for d in self.defenses],
                    "params": config.params,
                }
                with open(logger.metrics_path, "w", encoding="utf-8") as f:
                    json.dump(summary, f, indent=2, ensure_ascii=False)
                logger.info(f"Metrics saved to {logger.metrics_path}")

            logger.info(
                f"[MODEL] {config.model_config.name}  "
                + f"[DATASET] {data_config.name}  "
                + f"Finished"
            )

    def _generate(self, messages: list) -> str:
        inputs = self.tokenizer.apply_chat_template(
            messages, add_generation_prompt=True, return_tensors="pt"
        ).to(self.device)
        attention_mask = torch.ones_like(inputs)
        print(f"[pipeline] Generating {self.config.generation_config.max_tokens} tokens on {self.device}...", flush=True)
        outputs = self.model.generate(
            inputs,
            attention_mask=attention_mask,
            do_sample=False,
            max_new_tokens=self.config.generation_config.max_tokens,
        )
        print(f"[pipeline] Done. Generated {outputs.shape[1] - inputs.shape[1]} tokens.", flush=True)
        return self.tokenizer.batch_decode(
            outputs[:, inputs.shape[1]:], skip_special_tokens=True
        )[0]

    def _run_gcg_attack(self, prompt: str, original_resp: str):
        import nanogcg

        target = self.config.params["target"]
        two_stage = self.config.params.get("two_stage", {})
        if not two_stage.get("enabled", False):
            gcg_config = nanogcg.GCGConfig(**self.config.params["gcg_config"])
            gcg_result = nanogcg.run(
                self.model,
                self.tokenizer,
                [{"role": "user", "content": prompt}],
                target,
                gcg_config,
            )
            attacked_messages = [{"role": "user", "content": prompt + gcg_result.best_string}]
            return attacked_messages, None, {
                "best_string": gcg_result.best_string,
                "best_loss": float(gcg_result.best_loss),
                "gcg_mode": "single_stage",
            }

        target_grade = extract_grade(target) or "correct"
        base_config = deepcopy(self.config.params.get("gcg_config", {}))
        screen_config = deepcopy(two_stage.get("screening_config", {}))
        strong_config = deepcopy(two_stage.get("strong_config", {}))
        screen_params = {**base_config, **screen_config}
        strong_params = {**base_config, **strong_config}
        promote_max_loss = two_stage.get("promote_max_loss")
        verify_every = int(two_stage.get("verify_every", strong_params.get("num_steps", 100)))
        max_strong_steps = int(two_stage.get("max_strong_steps", strong_params.get("num_steps", 100)))

        meta = {
            "gcg_mode": "two_stage",
            "target_grade": target_grade,
            "screening_config": screen_params,
            "strong_config": strong_params,
            "promote_max_loss": promote_max_loss,
            "verify_every": verify_every,
            "max_strong_steps": max_strong_steps,
        }

        print("[pipeline] GCG screening stage...", flush=True)
        screen_result = nanogcg.run(
            self.model,
            self.tokenizer,
            [{"role": "user", "content": prompt}],
            target,
            nanogcg.GCGConfig(**screen_params),
        )
        best_string = screen_result.best_string
        best_loss = float(screen_result.best_loss)
        meta.update({
            "screen_best_string": best_string,
            "screen_best_loss": best_loss,
            "best_string": best_string,
            "best_loss": best_loss,
            "strong_steps_run": 0,
            "verified_success": False,
            "skipped_strong": False,
        })

        attacked_messages = [{"role": "user", "content": prompt + best_string}]
        attacked_resp = self._generate(attacked_messages)
        if self._is_target_flip(original_resp, attacked_resp, target_grade):
            meta.update({
                "verified_success": True,
                "success_stage": "screening",
            })
            return attacked_messages, attacked_resp, meta

        if promote_max_loss is not None and best_loss > float(promote_max_loss):
            meta.update({
                "skipped_strong": True,
                "skip_reason": "screen_loss_above_threshold",
            })
            return attacked_messages, attacked_resp, meta

        print("[pipeline] GCG strong stage with generation-verified checkpoints...", flush=True)
        steps_run = 0
        while steps_run < max_strong_steps:
            chunk_steps = min(verify_every, max_strong_steps - steps_run)
            chunk_params = deepcopy(strong_params)
            chunk_params["num_steps"] = chunk_steps
            chunk_params["optim_str_init"] = best_string
            chunk_params["early_stop"] = False

            chunk_result = nanogcg.run(
                self.model,
                self.tokenizer,
                [{"role": "user", "content": prompt}],
                target,
                nanogcg.GCGConfig(**chunk_params),
            )
            steps_run += chunk_steps
            best_string = chunk_result.best_string
            best_loss = float(chunk_result.best_loss)
            attacked_messages = [{"role": "user", "content": prompt + best_string}]
            attacked_resp = self._generate(attacked_messages)

            meta.update({
                "best_string": best_string,
                "best_loss": best_loss,
                "strong_steps_run": steps_run,
                "last_verified_response": attacked_resp,
            })
            if self._is_target_flip(original_resp, attacked_resp, target_grade):
                meta.update({
                    "verified_success": True,
                    "success_stage": f"strong_{steps_run}",
                })
                break

        return attacked_messages, attacked_resp, meta

    @staticmethod
    def _is_target_flip(original_resp: str, attacked_resp: str, target_grade: str) -> bool:
        original_grade = extract_grade(original_resp)
        attacked_grade = extract_grade(attacked_resp)
        return original_grade != target_grade and attacked_grade == target_grade

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
