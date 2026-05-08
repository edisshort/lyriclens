# ─────────────────────────────────────────────────────────────────────────────
# memory.py — Conversation Sliding Window
# ─────────────────────────────────────────────────────────────────────────────
# Keeps track of the last N messages so follow-up questions work naturally.
# Example:
#   User:  "Explain DAMN. by Kendrick Lamar"
#   Bot:   "DAMN. is a 2017 album..."
#   User:  "What about its emotional themes?"   ← works because memory exists
#
# We store messages as a plain Python list of dicts:
#   {"role": "user" | "assistant", "content": "...text..."}
#
# We only keep the last WINDOW_SIZE messages (default 6) to avoid blowing
# the LLM's token limit with very long conversations.
# ─────────────────────────────────────────────────────────────────────────────


WINDOW_SIZE = 6  # maximum number of messages to remember (user + assistant combined)


class ConversationMemory:
    """
    A simple sliding-window conversation store.

    Usage:
        memory = ConversationMemory()
        memory.add("user", "Explain DAMN.")
        memory.add("assistant", "DAMN. is a 2017 album by Kendrick Lamar...")
        history = memory.get_history()   # list of {"role":..., "content":...}
        formatted = memory.format()      # plain string for injecting into prompt
    """

    def __init__(self, window_size: int = WINDOW_SIZE):
        self.window_size = window_size
        # Internal storage — ordered list of message dicts
        self._messages: list[dict] = []

    # ── Write ──────────────────────────────────────────────────────────────

    def add(self, role: str, content: str) -> None:
        """
        Append a new message then trim to keep only the last `window_size` messages.

        Args:
            role    : "user" or "assistant"
            content : The message text
        """
        self._messages.append({"role": role, "content": content})

        # Sliding window — drop oldest messages when we exceed the limit
        if len(self._messages) > self.window_size:
            # Remove from the front (oldest)
            self._messages = self._messages[-self.window_size:]

    # ── Read ───────────────────────────────────────────────────────────────

    def get_history(self) -> list[dict]:
        """Return the current window as a list of message dicts."""
        return list(self._messages)  # return a copy so callers can't mutate internals

    def format(self) -> str:
        """
        Return conversation history as a plain-text string suitable for
        injecting into the LLM prompt.

        Output format:
            User: ...
            Assistant: ...
            User: ...
        """
        if not self._messages:
            return "No previous conversation."

        lines = []
        for msg in self._messages:
            # Capitalise the role label for readability
            label = msg["role"].capitalize()
            lines.append(f"{label}: {msg['content']}")

        return "\n".join(lines)

    # ── Utility ────────────────────────────────────────────────────────────

    def clear(self) -> None:
        """Wipe all stored messages (e.g. when starting a new research session)."""
        self._messages = []

    def __len__(self) -> int:
        return len(self._messages)

    def __repr__(self) -> str:
        return f"ConversationMemory(messages={len(self._messages)}, window={self.window_size})"
