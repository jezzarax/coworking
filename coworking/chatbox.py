import asyncio
import logging
import pathlib

import click
from pydantic_ai import usage

from coworking import agent, config, toolsets, util

logger = logging.getLogger(__name__)


@click.command("chatbox")
@click.option(
    "--llm-config",
    type=click.Path(file_okay=True, dir_okay=False, path_type=pathlib.Path),
    default="./configs/chatbox_models.yaml",
    help="YAML model config.",
)
@click.option("--model-alias", type=str, default="qwen")
@util.make_sync
async def chatbox_sim(llm_config: pathlib.Path, model_alias: str) -> None:
    llm_configs = config.load_llm_config(llm_config)
    agentic_model = agent.build_model(llm_configs.models[model_alias])

    board = agent.Switchboard()
    ledger = toolsets.Ledger(opening=0)  # one shared instance -> shared state
    limits = usage.UsageLimits(request_limit=20)

    alice = agent.build_agent(
        "Alice", agentic_model, ledger.toolset(), toolsets.Notebook().toolset()
    )
    bob = agent.build_agent("Bob", agentic_model, ledger.toolset(), toolsets.Notebook().toolset())

    async with (
        alice.iter(
            "Kick things off with Bob.",
            deps=agent.ChatDeps(me="alice", peer="bob", board=board),
            usage_limits=limits,
        ) as alice_run,
        bob.iter(
            "Wait for Alice, then collaborate.",
            deps=agent.ChatDeps(me="bob", peer="alice", board=board),
            usage_limits=limits,
        ) as bob_run,
    ):
        # Now that both runs exist, wire the switchboard.
        board.runs["alice"] = alice_run
        board.runs["bob"] = bob_run

        await asyncio.gather(agent.drive(alice_run), agent.drive(bob_run))

    print("alice:", alice_run.result.output if alice_run.result else "(unfinished)")
    print("bob:  ", bob_run.result.output if bob_run.result else "(unfinished)")
    pass
