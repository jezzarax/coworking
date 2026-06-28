import dataclasses

import pydantic_ai


@dataclasses.dataclass(frozen=True)
class ReadRecord:
    paper_id: str
    start_line: int
    end_line: int


@dataclasses.dataclass(frozen=True)
class MessageRecord:
    to: str
    text: str


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
    reads: list[ReadRecord] = dataclasses.field(default_factory=list)
    messages_sent: list[MessageRecord] = dataclasses.field(default_factory=list)
