import pydantic_ai


class Ledger:
    """A *shared* stateful service: both agents mutate the same balance."""

    def __init__(self, opening: int = 0) -> None:
        self._balance = opening

    def deposit(self, amount: int) -> str:
        """Add funds to the shared ledger."""
        self._balance += amount
        return f"balance is now {self._balance}"

    def balance(self) -> str:
        """Read the current ledger balance."""
        return f"current balance: {self._balance}"

    def toolset(self) -> pydantic_ai.FunctionToolset:
        ts = pydantic_ai.FunctionToolset()
        ts.add_function(self.deposit)
        ts.add_function(self.balance)
        return ts


class Notebook:
    """A *private* stateful service: each agent gets its own instance."""

    def __init__(self) -> None:
        self._notes: list[str] = []

    def jot(self, note: str) -> str:
        """Append a note to your private notebook."""
        self._notes.append(note)
        return f"noted ({len(self._notes)} total)"

    def toolset(self) -> pydantic_ai.FunctionToolset:
        ts = pydantic_ai.FunctionToolset()
        ts.add_function(self.jot)
        return ts
