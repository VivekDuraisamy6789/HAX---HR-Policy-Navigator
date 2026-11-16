import os
from dataclasses import dataclass, field
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()
ROOT = Path(__file__).resolve().parents[1]


def _env(name, default=""):
    return os.getenv(name, default)


@dataclass
class Settings:
    data_root: Path = field(default_factory=lambda: Path(_env("DATA_ROOT")).resolve() if _env("DATA_ROOT") else ROOT / "sourcegraph_files")
    app_title: str = _env("APP_TITLE", "HR Policy Navigator")
    base_url: str = _env("OPENAI_BASE_URL")
    api_key: str = _env("OPENAI_API_KEY")
    model: str = _env("OPENAI_MODEL", "qwen-8b-instruct")
    embedding_model: str = _env("EMBEDDING_MODEL")
    # constrained decoding: auto | json_schema | guided_json | json_object | none
    constrained_mode: str = _env("CONSTRAINED_MODE", "auto")
    # Qwen3 "thinking" off: true | false | auto (auto = on when the model name contains "qwen3")
    disable_thinking: str = _env("DISABLE_THINKING", "auto")
    max_tokens: int = int(_env("MAX_TOKENS", "700"))
    # agent safety rails
    max_steps: int = int(_env("MAX_STEPS", "8"))                 # executor tool calls per query
    max_llm_calls: int = int(_env("MAX_LLM_CALLS", "8"))
    max_query_tokens: int = int(_env("MAX_QUERY_TOKENS", "14000"))  # prompt+completion budget per query
    json_retries: int = int(_env("JSON_RETRIES", "2"))
    reranker: str = _env("RERANKER", "lexical")                   # lexical | cross_encoder | none
    top_k: int = int(_env("TOP_K", "6"))
    context_chars: int = int(_env("CONTEXT_CHARS", "1400"))        # per-chunk cap sent to model
    strategy: str = _env("STRATEGY", "structured")               # structured | selfconsistency | decomposition
    # price per 1M tokens (USD) for the cost estimate; set to your host's price
    price_in: float = float(_env("PRICE_IN_PER_M", "0.05"))
    price_out: float = float(_env("PRICE_OUT_PER_M", "0.10"))
    trace_dir: Path = field(default_factory=lambda: ROOT / "traces")
    state_dir: Path = field(default_factory=lambda: ROOT / "state")
    langfuse_public_key: str = _env("LANGFUSE_PUBLIC_KEY")
    langfuse_secret_key: str = _env("LANGFUSE_SECRET_KEY")
    langfuse_host: str = _env("LANGFUSE_HOST")
    rate_limit_per_min: int = int(_env("RATE_LIMIT_PER_MIN", "10"))
    cache_ttl_s: int = int(_env("CACHE_TTL_S", "900"))

    @property
    def no_think(self) -> bool:
        d = self.disable_thinking.lower()
        return d == "true" or (d == "auto" and "qwen3" in self.model.lower())

    @property
    def has_model(self) -> bool:
        return bool(self.base_url and self.api_key)


# backwards-compatible aliases used by older modules / docs
DATA_ROOT = Settings().data_root
APP_TITLE = Settings().app_title
