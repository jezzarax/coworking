import dataclasses
import logging
import os
import typing

import pydantic_ai
import pydantic_graph
from pydantic_ai.models.openai import OpenAIChatModel, OpenAIChatModelSettings
from pydantic_ai.providers.openai import OpenAIProvider
from pydantic_ai.toolsets import FunctionToolset

from coworking import agent_logging, config

logger = logging.getLogger(__name__)


async def drive(run):
    node = run.next_node
    while True:
        agent_logging.log_agent_execution_node(logger, node)
        if isinstance(node, pydantic_graph.End):
            break
        node = await run.next(node)


@dataclasses.dataclass
class Switchboard:
    runs: dict[str, pydantic_ai.AgentRun] = dataclasses.field(default_factory=dict)

    def send(self, to: str, text: str) -> str:
        run = self.runs.get(to)
        if run is None:
            return f"{to} is not connected"
        # priority='asap' is the "steering" semantics: the message is added to
        # the peer's next ModelRequest, or — if the peer was about to End —
        # redirects it into one more request so it can't miss the message.
        run.enqueue(f"[message from {to}'s peer] {text}", priority="asap")
        return "delivered"


@dataclasses.dataclass
class ChatDeps:
    me: str
    peer: str
    board: Switchboard


def build_model_settings(
    parsed_config: config.LLMModelConfig,
) -> OpenAIChatModelSettings:
    settings_config = {}

    if parsed_config.max_tokens is not None:
        settings_config["max_tokens"] = parsed_config.max_tokens
    if parsed_config.temperature is not None:
        settings_config["temperature"] = parsed_config.temperature
    if parsed_config.top_p is not None:
        settings_config["top_p"] = parsed_config.top_p
    if parsed_config.presence_penalty is not None:
        settings_config["presence_penalty"] = parsed_config.presence_penalty
    if parsed_config.extra_body:
        settings_config["extra_body"] = parsed_config.extra_body

    return OpenAIChatModelSettings(**settings_config)


def build_model(parsed_config: config.LLMModelConfig) -> OpenAIChatModel:
    api_key = os.getenv(parsed_config.api_key_env)
    provider_config: dict[str, typing.Any] = {"api_key": api_key}
    if parsed_config.base_url:
        provider_config["base_url"] = parsed_config.base_url

    model_settings = build_model_settings(parsed_config)

    model_config: dict[str, typing.Any] = {"provider": OpenAIProvider(**provider_config)}
    if model_settings is not None:
        model_config["settings"] = model_settings

    return OpenAIChatModel(parsed_config.model_name, **model_config)


def build_agent(
    name: str, model: OpenAIChatModel, *service_toolsets: FunctionToolset
) -> pydantic_ai.Agent[ChatDeps, str]:
    agent = pydantic_ai.Agent(
        model,
        deps_type=ChatDeps,
        toolsets=list(service_toolsets),
        instructions=(
            f"You are {name}. Work with your peer to drive the shared ledger "
            "to exactly 100, then reply DONE and stop. Use send_message to "
            "coordinate; deposit in turns so you don't overshoot."
        ),
    )

    @agent.tool
    def send_message(ctx: pydantic_ai.RunContext[ChatDeps], text: str) -> str:
        """Send a message to your peer agent."""
        return ctx.deps.board.send(ctx.deps.peer, text)

    return agent
