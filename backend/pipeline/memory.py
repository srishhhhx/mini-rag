from dataclasses import dataclass, field


@dataclass
class MessagePair:
    human: str
    assistant: str


class ChatMemory:
    """
    Sliding-window chat memory.
    Keeps the last `max_pairs` human/assistant message pairs.
    """

    def __init__(self, max_pairs: int = 6):
        self.max_pairs = max_pairs
        self._history: list[MessagePair] = []

    def add(self, human: str, assistant: str):
        self._history.append(MessagePair(human=human, assistant=assistant))
        if len(self._history) > self.max_pairs:
            self._history = self._history[-self.max_pairs :]

    def format(self) -> str:
        """Return history as a formatted string for the LLM prompt."""
        if not self._history:
            return ""
        lines = []
        for pair in self._history:
            lines.append(f"Human: {pair.human}")
            lines.append(f"Assistant: {pair.assistant}")
        return "\n".join(lines)

    def clear(self):
        self._history = []

    def get_last_assistant(self) -> str:
        """Return the last assistant message if available, else empty string."""
        return self._history[-1].assistant if self._history else ""

    def __len__(self) -> int:
        return len(self._history)


# One memory instance per session; stored in session_store would be redundant —
# instead, session.py will hold a ChatMemory field added at session creation.
