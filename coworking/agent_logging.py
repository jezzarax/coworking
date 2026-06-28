from __future__ import annotations

import dataclasses
import json
import logging
from collections.abc import Sequence

import pydantic
from pydantic_ai import CallToolsNode, ModelRequestNode, UserPromptNode, messages
from pydantic_ai._agent_graph import AgentNode
from pydantic_ai.result import FinalResult
from pydantic_graph import End

TEXT_PREVIEW_CHARS = 220
PROMPT_PREVIEW_CHARS = 160
type AgentMessagePart = messages.ModelRequestPart | messages.ModelResponsePart


def log_agent_execution_node[AgentDepsT, OutputDataT](
    logger: logging.Logger,
    execution_node: AgentNode[AgentDepsT, OutputDataT] | End[FinalResult[OutputDataT]],
) -> None:
    """Log a compact, readable representation of a PydanticAI execution node."""
    if isinstance(execution_node, UserPromptNode):
        prompt = _prompt_text(execution_node.user_prompt)
        logger.info(
            "Agent prompt: %s chars: %s",
            len(prompt),
            _preview(prompt, PROMPT_PREVIEW_CHARS),
        )
        return

    if isinstance(execution_node, ModelRequestNode):
        parts = execution_node.request.parts
        logger.info("Agent request: %s part(s)", len(parts))
        _log_parts(logger, parts)
        return

    if isinstance(execution_node, CallToolsNode):
        response = execution_node.model_response
        parts = response.parts
        logger.info("Agent model response: %s", _format_model_response(response))
        _log_parts(logger, parts)
        return

    if isinstance(execution_node, End):
        logger.info("Agent final: %s", _preview(execution_node.data.output))
        return

    logger.info("Agent node: %s", type(execution_node).__name__)


def _log_parts(logger: logging.Logger, parts: Sequence[AgentMessagePart]) -> None:
    for part in parts:
        if isinstance(part, messages.SystemPromptPart):
            content = part.content
            logger.info("System prompt: %s chars", len(content))
        elif isinstance(part, messages.UserPromptPart):
            content = _prompt_text(part.content)
            logger.info(
                "User prompt: %s chars: %s",
                len(content),
                _preview(content, PROMPT_PREVIEW_CHARS),
            )
        elif isinstance(part, messages.ThinkingPart):
            content = part.content
            logger.info("Model reasoning: %s chars", len(content))
        elif isinstance(part, messages.TextPart):
            content = part.content
            logger.info(
                "Model text: %s chars: %s",
                len(content),
                _preview(content, TEXT_PREVIEW_CHARS),
            )
        elif isinstance(part, messages.ToolCallPart | messages.NativeToolCallPart):
            logger.info(
                "Tool call: %s(%s)",
                part.tool_name,
                _format_tool_call_args(part),
            )
        elif isinstance(part, messages.ToolReturnPart | messages.NativeToolReturnPart):
            logger.info(
                "Tool result: %s(%s)",
                part.tool_name,
                _format_tool_return(part),
            )
        elif isinstance(part, messages.RetryPromptPart):
            content = _prompt_text(part.content)
            logger.info(
                "Retry prompt: %s chars: %s",
                len(content),
                _preview(content, PROMPT_PREVIEW_CHARS),
            )
        elif isinstance(part, messages.CompactionPart):
            content = part.content or ""
            logger.info("Compaction: %s chars", len(content))
        elif isinstance(part, messages.FilePart):
            logger.info("Model file: %s", type(part.content).__name__)
        else:
            logger.info("Agent part: %s", type(part).__name__)


def _format_model_response(response: messages.ModelResponse) -> str:
    fields = [f"{len(response.parts)} part(s)"]

    if response.model_name:
        fields.append(f"model={response.model_name}")
    if response.usage.input_tokens or response.usage.output_tokens:
        fields.append(f"tokens=in {response.usage.input_tokens}/out {response.usage.output_tokens}")

    return ", ".join(fields)


def _format_tool_call_args(
    part: messages.ToolCallPart | messages.NativeToolCallPart,
) -> str:
    args = part.args_as_dict()

    if part.tool_name == "label_ticket":
        return _format_label_ticket_call(args)
    if part.tool_name == "report_relevant_tickets":
        ticket_numbers = args.get("ticket_numbers", [])
        return f"ticket_numbers={_format_list(ticket_numbers)}"

    return _compact_json(args)


def _format_label_ticket_call(args: dict[str, object]) -> str:
    ticket_number = args.get("ticket_number", "?")
    canonical_tags = args.get("canonical_tags") or []
    tag_confidence = args.get("tag_confidence") or {}
    new_tag_candidates = args.get("new_tag_candidates") or []

    tags = []
    for tag in _list_items(canonical_tags):
        confidence = tag_confidence.get(tag)
        tags.append(f"{tag}:{confidence}" if confidence else str(tag))

    fields = [f"ticket={ticket_number}", f"tags={_format_list(tags)}"]
    if new_tag_candidates:
        fields.append(f"new={_format_list(new_tag_candidates)}")
    return ", ".join(fields)


def _format_tool_return(
    part: messages.ToolReturnPart | messages.NativeToolReturnPart,
) -> str:
    data = _plain_data(part.content)

    if part.tool_name == "label_ticket" and isinstance(data, dict):
        ticket_number = data.get("ticket_number", "?")
        success = data.get("success", "?")
        recorded = data.get("recorded_ticket_numbers") or []
        return f"ticket={ticket_number}, success={success}, recorded={len(recorded)}"
    if part.tool_name == "report_relevant_tickets" and isinstance(data, dict):
        reported = data.get("reported_ticket_numbers") or []
        missing = data.get("missing_ticket_numbers") or []
        return f"reported={_format_list(reported)}, missing={_format_list(missing)}"

    return _compact_json(data)


def _plain_data(value: object) -> object:
    if isinstance(value, pydantic.BaseModel):
        return value.model_dump(mode="json")
    if dataclasses.is_dataclass(value):
        return _plain_data(dataclasses.asdict(value))
    if isinstance(value, str):
        try:
            return json.loads(value)
        except ValueError:
            return value
    if isinstance(value, dict):
        return {str(key): _plain_data(item) for key, item in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [_plain_data(item) for item in value]
    return value


def _compact_json(value: object) -> str:
    if value is None:
        return ""
    if isinstance(value, str):
        return _preview(value)
    try:
        return _preview(json.dumps(_plain_data(value), ensure_ascii=False, sort_keys=True))
    except (TypeError, ValueError):
        return _preview(value)


def _format_list(values: object) -> str:
    if not isinstance(values, list):
        return _compact_json(values)
    if not values:
        return "[]"
    return "[" + ", ".join(str(value) for value in values) + "]"


def _list_items(value: object) -> list[object]:
    return value if isinstance(value, list) else []


def _prompt_text(value: object) -> str:
    if value is None:
        return ""
    if isinstance(value, str):
        return value
    return _compact_json(value)


def _preview(value: object, max_chars: int = TEXT_PREVIEW_CHARS) -> str:
    text = " ".join(str(value).split())
    if len(text) <= max_chars:
        return text
    return f"{text[:max_chars].rstrip()}..."
