import asyncio
import contextlib
import logging
import pathlib

import click
import pydantic_ai
from pydantic_ai import usage

from coworking import agent, config, toolsets, util

logger = logging.getLogger(__name__)

AGENT_NAMES: tuple[str, ...] = ("alice", "bob", "carol")

USER_PROMPT = (
    "Begin exploring the paper corpus and coordinate with your peers to "
    "identify how the papers are connected to each other."
)


def _make_instruction(name: str, peers: list[str]) -> str:
    peer_list = ", ".join(p.capitalize() for p in peers)
    return (
        f"You are {name.capitalize()}. You are part of a multi-agent team "
        f"exploring the shared paper corpus in the Papers toolset. Your peers "
        f"are: {peer_list}. Your job: identify how the papers in the folder are "
        "connected to each other (shared topics, methods, datasets, citations, "
        "contradictions, evolution of ideas). Use the Papers tools to read and "
        "search the corpus; jot intermediate findings into your private Notebook "
        "so you can reread them later. Coordinate with your peers via "
        "send_message (broadcast or directed) to avoid duplicating work and to "
        "cross-check interpretations. When you have a clear picture, reply with "
        "a concise synthesis of the connections you found."
    )


@click.command("chatbox")
@click.option(
    "--llm-config",
    type=click.Path(file_okay=True, dir_okay=False, path_type=pathlib.Path),
    default="./configs/chatbox_models.yaml",
    help="YAML model config.",
)
@click.option("--model-alias", type=str, default="qwen")
@click.option(
    "--papers-dir",
    type=click.Path(file_okay=False, dir_okay=True, path_type=pathlib.Path),
    default="./data/papers",
    help="Directory containing paper markdown files.",
)
@util.make_sync
async def chatbox_sim(llm_config: pathlib.Path, model_alias: str, papers_dir: pathlib.Path) -> None:
    llm_configs = config.load_llm_config(llm_config)
    agentic_model = agent.build_model(llm_configs.models[model_alias])

    papers = toolsets.Papers(root=papers_dir)
    board = agent.Switchboard()
    limits = usage.UsageLimits(request_limit=30)

    runs: dict[str, pydantic_ai.AgentRun] = {}
    ctx_managers: list[tuple[str, pydantic_ai.AgentRun]] = []
    for name in AGENT_NAMES:
        peer_names = [n for n in AGENT_NAMES if n != name]
        this_agent = agent.build_agent(
            name.capitalize(),
            agentic_model,
            _make_instruction(name, peer_names),
            papers.toolset(),
            toolsets.Notebook().toolset(),
        )
        ctx_managers.append(
            (name, this_agent.iter(
                USER_PROMPT,
                deps=agent.ChatDeps(me=name, peers=peer_names, board=board),
                usage_limits=limits,
            ))
        )

    async with contextlib.AsyncExitStack() as stack:
        for name, cm in ctx_managers:
            runs[name] = await stack.enter_async_context(cm)
        for name, run in runs.items():
            board.runs[name] = run
        await asyncio.gather(*(agent.drive(run, name) for name, run in runs.items()))

    for name, run in runs.items():
        label = name.capitalize()
        print(f"{label}: {run.result.output if run.result else '(unfinished)'}")