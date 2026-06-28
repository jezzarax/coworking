import logging
import os
import pathlib
import re
import typing

import pydantic
from ruamel import yaml

LOGGER = logging.getLogger(__name__)
ENV_PATTERN = re.compile(r"\$\{([A-Za-z_][A-Za-z0-9_]*)\}")


class LLMError(RuntimeError):
    """Base class for LLM failures."""


class LLMModelConfig(pydantic.BaseModel):
    model_name: str
    base_url: str = ""
    api_key_env: str
    concurrency: int = pydantic.Field(default=1, ge=1)
    max_tokens: int | None = pydantic.Field(default=16384, ge=1)
    temperature: float | None = None
    top_p: float | None = None
    presence_penalty: float | None = None
    extra_body: dict[str, typing.Any] = pydantic.Field(default_factory=dict)

    model_config = pydantic.ConfigDict(extra="forbid")


class LLMConfig(pydantic.BaseModel):
    models: dict[str, LLMModelConfig] = pydantic.Field(min_length=3, max_length=3)

    model_config = pydantic.ConfigDict(extra="forbid")


def load_llm_config(path: pathlib.Path) -> LLMConfig:
    """Load and validate the YAML model config, resolving env placeholders."""

    yaml_parser = yaml.YAML()

    payload = yaml_parser.load(path.read_text(encoding="utf-8")) or {}
    config = LLMConfig.model_validate(payload)
    return LLMConfig(
        models={
            model_id: model.model_copy(
                update={
                    "model_name": _resolve_env_placeholders(model.model_name),
                    "base_url": _resolve_env_placeholders(model.base_url),
                }
            )
            for model_id, model in config.models.items()
        }
    )


def _resolve_env_placeholders(value: str) -> str:
    def replace(match: re.Match[str]) -> str:
        env_name = match.group(1)
        resolved = os.environ.get(env_name)
        if resolved is None:
            raise LLMError(f"Environment variable {env_name} is not set")
        return resolved

    return ENV_PATTERN.sub(replace, value)
