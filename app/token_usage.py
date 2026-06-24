from langchain_core.callbacks.base import BaseCallbackHandler


class TokenUsageCallback(BaseCallbackHandler):
    def __init__(self):
        self.prompt_tokens     = 0
        self.completion_tokens = 0
        self.total_tokens      = 0
        self.model_name        = ""

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------
    @staticmethod
    def _extract_openai_usage(llm_output: dict) -> dict | None:
        """Return a normalised usage dict from llm_output, or None."""
        raw = llm_output.get("token_usage") or llm_output.get("usage")
        if not raw:
            return None
        return {
            "prompt_tokens":     raw.get("prompt_tokens")     or raw.get("input_tokens",  0),
            "completion_tokens": raw.get("completion_tokens") or raw.get("output_tokens", 0),
            "total_tokens":      raw.get("total_tokens")      or raw.get("total_token_count", 0),
        }

    @staticmethod
    def _extract_gemini_usage(generation_info: dict) -> dict:
        """Return a normalised usage dict from Gemini generation_info."""
        usage = generation_info.get("usage_metadata", {})
        return {
            "prompt_tokens":     usage.get("prompt_token_count",     0),
            "completion_tokens": usage.get("candidates_token_count", 0),
            "total_tokens":      usage.get("total_token_count",      0),
        }

    def _apply(self, usage: dict) -> None:
        self.prompt_tokens     += usage["prompt_tokens"]
        self.completion_tokens += usage["completion_tokens"]
        self.total_tokens      += usage["total_tokens"]

    # ------------------------------------------------------------------
    # Callback
    # ------------------------------------------------------------------
    def on_llm_end(self, response, **kwargs) -> None:
        populated = False
        print(f"[DEBUG] llm_output keys: {list((response.llm_output or {}).keys())}")
        if response.generations:
          info = getattr(response.generations[0][0], "generation_info", {}) or {}
        print(f"[DEBUG] generation_info keys: {list(info.keys())}")
        print(f"[DEBUG] usage_metadata: {info.get('usage_metadata')}")

        # 1. OpenAI-style: check llm_output first
        if hasattr(response, "llm_output") and response.llm_output:
            self.model_name = response.llm_output.get("model_name", self.model_name)
            usage = self._extract_openai_usage(response.llm_output)
            if usage and any(usage.values()):
                self._apply(usage)
                populated = True

        # 2. Gemini-style: fall back to generation_info only if nothing was found
        if not populated and response.generations:
            for generation_list in response.generations:
                for generation in generation_list:
                    info = getattr(generation, "generation_info", None) or {}
                    if info:
                        self._apply(self._extract_gemini_usage(info))
                        populated = True

    # ------------------------------------------------------------------
    # Output
    # ------------------------------------------------------------------
    def to_dict(self) -> dict:
        return {
            "prompt_tokens":     self.prompt_tokens,
            "completion_tokens": self.completion_tokens,
            "total_tokens":      self.total_tokens,
            "model_name":        self.model_name,
        }