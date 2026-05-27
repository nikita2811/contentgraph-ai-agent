from langchain_core.callbacks.base import BaseCallbackHandler


class TokenUsageCallback(BaseCallbackHandler):
    def __init__(self):
        self.prompt_tokens     = 0
        self.completion_tokens = 0
        self.total_tokens      = 0
        self.model_name        = ""

    def on_llm_end(self, response, **kwargs):
        # ── Debug: see exactly what Gemini returns ────────
        # print(f"🔍 llm_output: {response.llm_output}")
        # print(f"🔍 generations: {response.generations}")

        # ── Try llm_output first (OpenAI-style) ──────────
        if hasattr(response, "llm_output") and response.llm_output:
            usage = response.llm_output.get("token_usage") or response.llm_output.get("usage", {})
            self.prompt_tokens     += usage.get("prompt_tokens", 0) or usage.get("input_tokens", 0)
            self.completion_tokens += usage.get("completion_tokens", 0) or usage.get("output_tokens", 0)
            self.total_tokens      += usage.get("total_tokens", 0) or usage.get("total_token_count", 0)
            self.model_name         = response.llm_output.get("model_name", "")

        # ── Gemini stores usage in generation_info ────────
        if self.total_tokens == 0 and response.generations:
            for generation_list in response.generations:
                for generation in generation_list:
                    info = getattr(generation, "generation_info", {}) or {}
                    usage = info.get("usage_metadata", {})
                    print(f"🔍 generation_info: {info}")  # ← see what's here
                    self.prompt_tokens     += usage.get("prompt_token_count", 0)
                    self.completion_tokens += usage.get("candidates_token_count", 0)
                    self.total_tokens      += usage.get("total_token_count", 0)

    def to_dict(self) -> dict:
        return {
            "prompt_tokens":     self.prompt_tokens,
            "completion_tokens": self.completion_tokens,
            "total_tokens":      self.total_tokens,
            "model_name":        self.model_name,
        }