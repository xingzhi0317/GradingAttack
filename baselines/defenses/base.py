from abc import ABC


class DefenseRejectException(Exception):
    """Pre-processing defense 判定输入为攻击时抛此异常，跳过后续推理"""

    def __init__(self, defense_name: str, reason: str):
        self.defense_name = defense_name
        self.reason = reason
        super().__init__(f"[{defense_name}] Input rejected: {reason}")


class BaseDefense(ABC):
    def __init__(self):
        self.name = self.__class__.__name__

    def pre_process(self, prompt: str) -> str:
        """在推理前修改 prompt。默认透传。

        返回修改后的 prompt，或 raise DefenseRejectException 直接拒答。
        """
        return prompt

    def post_process(self, response: str) -> str:
        """在推理后修正输出。默认透传。"""
        return response

    def requires_multiple_generations(self) -> bool:
        """是否需要多次推理（如 SmoothLLM 多数投票）。默认 False。"""
        return False

    def generate_variants(self, prompt: str) -> list[str]:
        """生成 prompt 的多个变体（pre_process + 复制扰动）。默认返回单元素列表。"""
        return [self.pre_process(prompt)]

    def requires_model_hooks(self) -> bool:
        """是否需要在推理前安装模型级 hook（如 Attention Sharpening / HS）。"""
        return False

    def requires_inference_context(self) -> bool:
        """是否需要在每次推理前设置 prompt/suffix 上下文（如 HS）。"""
        return False

    def set_inference_context(
        self,
        tokenizer,
        prompt_content: str,
        attack_suffix: str = "",
    ) -> None:
        """在安装 hook 前注入当前样本的 prompt / suffix 信息。"""

    def install_model_hooks(self, model) -> list:
        """安装模型 hook，返回 uninstall 回调列表。"""
        return []

    def uninstall_model_hooks(self, handles) -> None:
        for remove in handles:
            remove()
