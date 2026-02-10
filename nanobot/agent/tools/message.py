"""Message tool for sending messages to users."""

from typing import Any, Awaitable, Callable

from nanobot.agent.tools.base import Tool
from nanobot.bus.events import OutboundMessage


class MessageTool(Tool):
    """Tool to send messages to users on chat channels."""

    def __init__(
        self,
        send_callback: Callable[[OutboundMessage], Awaitable[None]] | None = None,
        default_channel: str = "",
        default_chat_id: str = ""
    ):
        self._send_callback = send_callback
        self._default_channel = default_channel
        self._default_chat_id = default_chat_id
        # Track messages sent during this processing cycle so the agent loop
        # can persist them into the session history.
        # Each entry: {"content": str, "media": list[str]}
        self._sent_messages: list[dict[str, Any]] = []

    def set_context(self, channel: str, chat_id: str) -> None:
        """Set the current message context."""
        self._default_channel = channel
        self._default_chat_id = chat_id
        self._sent_messages = []

    def set_send_callback(self, callback: Callable[[OutboundMessage], Awaitable[None]]) -> None:
        """Set the callback for sending messages."""
        self._send_callback = callback

    @property
    def name(self) -> str:
        return "message"

    @property
    def description(self) -> str:
        return "Send a message to the user, optionally with attachments (media)."

    @property
    def parameters(self) -> dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "content": {
                    "type": "string",
                    "description": "The message content to send"
                },
                "media": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "Optional: file paths or URLs to send as attachments"
                },
                "channel": {
                    "type": "string",
                    "description": "Optional: target channel (telegram, discord, etc.)"
                },
                "chat_id": {
                    "type": "string",
                    "description": "Optional: target chat/user ID"
                }
            },
            "required": ["content"]
        }

    async def execute(
        self,
        content: str,
        media: list[str] | None = None,
        channel: str | None = None,
        chat_id: str | None = None,
        **kwargs: Any
    ) -> str:
        channel = channel or self._default_channel
        chat_id = chat_id or self._default_chat_id

        if not channel or not chat_id:
            return "Error: No target channel/chat specified"

        if not self._send_callback:
            return "Error: Message sending not configured"

        msg = OutboundMessage(
            channel=channel,
            chat_id=chat_id,
            content=content,
            media=media or []
        )

        try:
            await self._send_callback(msg)
            self._sent_messages.append({
                "content": content,
                "media": media or [],
            })
            return f"Message sent to {channel}:{chat_id}"
        except Exception as e:
            return f"Error sending message: {str(e)}"

    def drain_sent_messages(self) -> list[dict[str, Any]]:
        """Return and clear the list of messages sent during this cycle.

        Each entry is ``{"content": str, "media": list[str]}``.
        """
        msgs = self._sent_messages
        self._sent_messages = []
        return msgs
