import asyncio
import contextlib
import logging
import pathlib

import click
import pydantic_ai
from pydantic_ai import usage

from coworking import agent, config, structs, toolsets, util

logger = logging.getLogger(__name__)

AGENT_NAMES: tuple[str, ...] = ("Alice", "Bob", "Carol")

USER_PROMPT = (
    "Begin exploring the paper corpus and coordinate with your peers to "
    "identify how the papers are connected to each other."
)


def _make_instruction(name: str, peers: list[str]) -> str:
    peer_list = ", ".join(peers)
    return (
        f"You are {name}. You are part of a multi-agent team "
        f"exploring the shared paper corpus in the Papers toolset. Your peers "
        f"are: {peer_list}.\n\n"
        "Your job: identify how the papers in the folder are connected to each "
        "other (shared topics, methods, datasets, citations, contradictions, "
        "evolution of ideas). Use the Papers tools (list_files, info, read, "
        "grep) to explore the corpus.\n\n"
        "Work efficiently as a team. Split the reading across the team so you "
        "don't duplicate effort: before reading a paper, send a message to "
        "your peers to check whether anyone has already covered it. If a "
        "peer has, ask them to share what they found instead of reading it "
        "yourself. Only re-read a paper yourself when you genuinely need "
        "details they cannot relay. Jot intermediate findings into your "
        "private Notebook so peers can reread them later.\n\n"
        "Coordinate with your peers via send_message (broadcast or directed) "
        "to cross-check interpretations and decide who covers what.\n\n"
        "When you have a clear picture, publish your final report using the "
        "publish_report tool. The team should coordinate so that only one "
        "agent publishes the consensus report. If you disagree with the "
        "consensus, you may publish a separate dissenting report."
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
    report_board = toolsets.ReportBoard()
    board = structs.Switchboard()
    limits = usage.UsageLimits(request_limit=30)

    deps_map: dict[str, structs.ChatDeps] = {}
    runs: dict[str, pydantic_ai.AgentRun] = {}
    ctx_managers: list[tuple[str, pydantic_ai.AgentRun]] = []
    for name in AGENT_NAMES:
        peer_names = [n for n in AGENT_NAMES if n != name]
        deps = structs.ChatDeps(me=name, peers=peer_names, board=board)
        deps_map[name] = deps
        this_agent = agent.build_agent(
            name,
            agentic_model,
            _make_instruction(name, peer_names),
            report_board,
            papers.toolset(),
            toolsets.Notebook().toolset(),
        )
        ctx_managers.append(
            (
                name,
                this_agent.iter(
                    USER_PROMPT,
                    deps=deps,
                    usage_limits=limits,
                ),
            )
        )

    async with contextlib.AsyncExitStack() as stack:
        for name, cm in ctx_managers:
            runs[name] = await stack.enter_async_context(cm)
        for name, run in runs.items():
            board.runs[name] = run
        await asyncio.gather(*(agent.drive(run, name) for name, run in runs.items()))

    for name, run in runs.items():
        print(f"{name}: {run.result.output if run.result else '(unfinished)'}")

    print()
    print("=== Per-agent activity ===")
    for name in AGENT_NAMES:
        deps = deps_map[name]
        print(f"\n--- {name} ---")
        print(f"Reads ({len(deps.reads)}):")
        if not deps.reads:
            print("  (none)")
        for r in deps.reads:
            print(f"  {r.paper_id}: lines {r.start_line}-{r.end_line}")
        print(f"Messages sent ({len(deps.messages_sent)}):")
        if not deps.messages_sent:
            print("  (none)")
        for m in deps.messages_sent:
            preview = m.text if len(m.text) <= 80 else m.text[:77] + "..."
            print(f"  -> {m.to}: {preview}")

    print()
    print("=== Published reports ===")
    if not report_board.reports:
        print("(none)")
    for r in report_board.reports:
        print(f"\n[{r.author}]")
        print(f"  {r.content}")
