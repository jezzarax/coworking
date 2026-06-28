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


async def drive(run, name: str):
    node = run.next_node
    while True:
        agent_logging.log_agent_execution_node(logger.getChild(name), node)
        if isinstance(node, pydantic_graph.End):
            break
        node = await run.next(node)


@dataclasses.dataclass
class Switchboard:
    runs: dict[str, pydantic_ai.AgentRun] = dataclasses.field(default_factory=dict)

    def send(self, from_: str, to: str, text: str) -> str:
        run = self.runs.get(to)
        if run is None:
            return f"{to} is not connected"
        # priority='asap' is the "steering" semantics: the message is added to
        # the peer's next ModelRequest, or — if the peer was about to End —
        # redirects it into one more request so it can't miss the message.
        run.enqueue(f"[message from {from_}] {text}", priority="asap")
        return "delivered"


@dataclasses.dataclass
class ChatDeps:
    me: str
    peers: list[str]
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
    name: str,
    model: OpenAIChatModel,
    instruction: str,
    *service_toolsets: FunctionToolset,
) -> pydantic_ai.Agent[ChatDeps, str]:
    agent = pydantic_ai.Agent(
        model,
        deps_type=ChatDeps,
        toolsets=list(service_toolsets),
        instructions=instruction,
    )

    @agent.tool
    def send_message(
        ctx: pydantic_ai.RunContext[ChatDeps],
        text: str,
        to: str | None = None,
    ) -> str:
        """Send a message to a peer agent.

        If ``to`` is omitted, the message is broadcast to every peer.
        """
        targets = [to.lower()] if to else list(ctx.deps.peers)
        results = [ctx.deps.board.send(ctx.deps.me, target, text) for target in targets]
        return "; ".join(results)

    return agent
