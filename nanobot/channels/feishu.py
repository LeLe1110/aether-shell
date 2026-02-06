"""Feishu/Lark channel implementation using lark-oapi SDK with WebSocket long connection."""

import asyncio
import json
import threading
from collections import OrderedDict
from typing import Any

from loguru import logger

from nanobot.bus.events import OutboundMessage
from nanobot.bus.queue import MessageBus
from nanobot.channels.base import BaseChannel
from nanobot.config.schema import FeishuConfig

try:
    import lark_oapi as lark
    from lark_oapi.api.im.v1 import (
        CreateMessageRequest,
        CreateMessageRequestBody,
        CreateMessageReactionRequest,
        CreateMessageReactionRequestBody,
        Emoji,
        P2ImMessageReceiveV1,
    )
    FEISHU_AVAILABLE = True
except ImportError:
    FEISHU_AVAILABLE = False
    lark = None
    Emoji = None

# Message type display mapping
MSG_TYPE_MAP = {
    "image": "[image]",
    "audio": "[audio]",
    "file": "[file]",
    "sticker": "[sticker]",
}


class FeishuChannel(BaseChannel):
    """
    Feishu/Lark channel using WebSocket long connection.
    
    Uses WebSocket to receive events - no public IP or webhook required.
    
    Requires:
    - App ID and App Secret from Feishu Open Platform
    - Bot capability enabled
    - Event subscription enabled (im.message.receive_v1)
    """
    
    name = "feishu"
    
    def __init__(self, config: FeishuConfig, bus: MessageBus):
        super().__init__(config, bus)
        self.config: FeishuConfig = config
        self._client: Any = None
        self._ws_client: Any = None
        self._ws_thread: threading.Thread | None = None
        self._processed_message_ids: OrderedDict[str, None] = OrderedDict()  # Ordered dedup cache
        self._loop: asyncio.AbstractEventLoop | None = None
    
    async def start(self) -> None:
        """Start the Feishu bot with WebSocket long connection."""
        if not FEISHU_AVAILABLE:
            logger.error("Feishu SDK not installed. Run: pip install lark-oapi")
            return
        
        if not self.config.app_id or not self.config.app_secret:
            logger.error("Feishu app_id and app_secret not configured")
            return
        
        self._running = True
        self._loop = asyncio.get_running_loop()
        
        # Create Lark client for sending messages
        self._client = lark.Client.builder() \
            .app_id(self.config.app_id) \
            .app_secret(self.config.app_secret) \
            .log_level(lark.LogLevel.INFO) \
            .build()
        
        # Create event handler (register supported events to avoid "processor not found" logs)
        handler_builder = lark.EventDispatcherHandler.builder(
            self.config.encrypt_key or "",
            self.config.verification_token or "",
        )
        handler_builder.register_p2_im_message_receive_v1(self._on_message_sync)
        # Optional events can be emitted by Feishu even if not used by us.
        try:
            for method_name in (
                "register_p2_im_message_read_v1",
                "register_p2_im_message_message_read_v1",
            ):
                method = getattr(handler_builder, method_name, None)
                if callable(method):
                    method(self._on_message_read_sync)
                    break
            method = getattr(
                handler_builder,
                "register_p2_im_message_reaction_created_v1",
                None,
            )
            if callable(method):
                method(self._on_message_reaction_created_sync)
        except Exception as e:
            logger.debug(f"Feishu optional event registration skipped: {e}")
        event_handler = handler_builder.build()
        
        # Create WebSocket client for long connection
        self._ws_client = lark.ws.Client(
            self.config.app_id,
            self.config.app_secret,
            event_handler=event_handler,
            log_level=lark.LogLevel.INFO
        )
        
        # Start WebSocket client in a separate thread
        def run_ws():
            try:
                self._ws_client.start()
            except Exception as e:
                logger.error(f"Feishu WebSocket error: {e}")
        
        self._ws_thread = threading.Thread(target=run_ws, daemon=True)
        self._ws_thread.start()
        
        logger.info("Feishu bot started with WebSocket long connection")
        logger.info("No public IP required - using WebSocket to receive events")
        
        # Keep running until stopped
        while self._running:
            await asyncio.sleep(1)
    
    async def stop(self) -> None:
        """Stop the Feishu bot."""
        self._running = False
        if self._ws_client:
            try:
                self._ws_client.stop()
            except Exception as e:
                logger.warning(f"Error stopping WebSocket client: {e}")
        logger.info("Feishu bot stopped")
    
    def _add_reaction_sync(self, message_id: str, emoji_type: str) -> None:
        """Sync helper for adding reaction (runs in thread pool)."""
        try:
            request = CreateMessageReactionRequest.builder() \
                .message_id(message_id) \
                .request_body(
                    CreateMessageReactionRequestBody.builder()
                    .reaction_type(Emoji.builder().emoji_type(emoji_type).build())
                    .build()
                ).build()
            
            response = self._client.im.v1.message_reaction.create(request)
            
            if not response.success():
                logger.warning(f"Failed to add reaction: code={response.code}, msg={response.msg}")
            else:
                logger.debug(f"Added {emoji_type} reaction to message {message_id}")
        except Exception as e:
            logger.warning(f"Error adding reaction: {e}")

    async def _add_reaction(self, message_id: str, emoji_type: str = "THUMBSUP") -> None:
        """
        Add a reaction emoji to a message (non-blocking).
        
        Common emoji types: THUMBSUP, OK, EYES, DONE, OnIt, HEART
        """
        if not self._client or not Emoji:
            return
        
        loop = asyncio.get_running_loop()
        await loop.run_in_executor(None, self._add_reaction_sync, message_id, emoji_type)
    
    async def send(self, msg: OutboundMessage) -> None:
        """Send a message through Feishu."""
        if not self._client:
            logger.warning("Feishu client not initialized")
            return
        
        try:
            # Determine receive_id_type based on chat_id format
            # open_id starts with "ou_", chat_id starts with "oc_"
            if msg.chat_id.startswith("oc_"):
                receive_id_type = "chat_id"
            else:
                receive_id_type = "open_id"
            
            # Build text message content
            content = json.dumps({"text": msg.content})
            
            request = CreateMessageRequest.builder() \
                .receive_id_type(receive_id_type) \
                .request_body(
                    CreateMessageRequestBody.builder()
                    .receive_id(msg.chat_id)
                    .msg_type("text")
                    .content(content)
                    .build()
                ).build()
            
            response = self._client.im.v1.message.create(request)
            
            if not response.success():
                logger.error(
                    f"Failed to send Feishu message: code={response.code}, "
                    f"msg={response.msg}, log_id={response.get_log_id()}"
                )
            else:
                logger.debug(f"Feishu message sent to {msg.chat_id}")
                
        except Exception as e:
            logger.error(f"Error sending Feishu message: {e}")

    @staticmethod
    def _parse_post_content(raw_content: str) -> str:
        """Parse Feishu/Lark 'post' message content into plain text."""
        try:
            data = json.loads(raw_content) if raw_content else {}
        except json.JSONDecodeError:
            return raw_content or ""

        if not isinstance(data, dict):
            return raw_content or ""

        post_data = data.get("post")
        body: dict[str, Any] = data

        if isinstance(post_data, dict):
            # Multi-locale format: {"post": {"zh_cn": {...}, "en_us": {...}}}
            if "title" in post_data or "content" in post_data:
                body = post_data
            else:
                preferred = post_data.get("zh_cn") or post_data.get("en_us")
                if isinstance(preferred, dict):
                    body = preferred
                elif post_data:
                    first = next(iter(post_data.values()))
                    if isinstance(first, dict):
                        body = first

        lines: list[str] = []
        title = body.get("title") if isinstance(body, dict) else ""
        if isinstance(title, str) and title.strip():
            lines.append(title.strip())

        content_blocks = body.get("content") if isinstance(body, dict) else None
        if isinstance(content_blocks, list):
            for block in content_blocks:
                if not isinstance(block, list):
                    continue
                line_parts: list[str] = []
                for elem in block:
                    if not isinstance(elem, dict):
                        continue
                    tag = elem.get("tag")
                    if tag == "text":
                        text = elem.get("text") or ""
                        if text:
                            line_parts.append(text)
                    elif tag == "a":
                        text = elem.get("text") or ""
                        href = elem.get("href") or ""
                        if text and href:
                            line_parts.append(f"{text} ({href})")
                        elif text:
                            line_parts.append(text)
                        elif href:
                            line_parts.append(href)
                    elif tag == "at":
                        name = elem.get("user_name") or elem.get("user_id") or ""
                        if name:
                            line_parts.append(f"@{name}")
                    elif tag == "emoji":
                        emoji = elem.get("emoji_type") or elem.get("emoji_name") or ""
                        if emoji:
                            line_parts.append(emoji)
                    elif tag == "img":
                        line_parts.append("[image]")
                    elif tag == "media":
                        line_parts.append("[media]")
                    elif tag == "hr":
                        line_parts.append("---")
                    else:
                        text = elem.get("text")
                        if text:
                            line_parts.append(text)
                line = "".join(line_parts).strip()
                if line:
                    lines.append(line)

        if not lines:
            text = data.get("text")
            if isinstance(text, str) and text.strip():
                lines.append(text.strip())

        return "\n".join(lines).strip()
    
    def _on_message_sync(self, data: "P2ImMessageReceiveV1") -> None:
        """
        Sync handler for incoming messages (called from WebSocket thread).
        Schedules async handling in the main event loop.
        """
        if self._loop and self._loop.is_running():
            asyncio.run_coroutine_threadsafe(self._on_message(data), self._loop)

    def _on_message_read_sync(self, data: Any) -> None:
        """Ignore message read events to avoid noisy logs."""
        return

    def _on_message_reaction_created_sync(self, data: Any) -> None:
        """Ignore reaction created events to avoid noisy logs."""
        return
    
    async def _on_message(self, data: "P2ImMessageReceiveV1") -> None:
        """Handle incoming message from Feishu."""
        try:
            event = data.event
            message = event.message
            sender = event.sender
            
            # Deduplication check
            message_id = message.message_id
            if message_id in self._processed_message_ids:
                return
            self._processed_message_ids[message_id] = None
            
            # Trim cache: keep most recent 500 when exceeds 1000
            while len(self._processed_message_ids) > 1000:
                self._processed_message_ids.popitem(last=False)
            
            # Skip bot messages
            sender_type = sender.sender_type
            if sender_type == "bot":
                return
            
            sender_id = sender.sender_id.open_id if sender.sender_id else "unknown"
            chat_id = message.chat_id
            chat_type = message.chat_type  # "p2p" or "group"
            msg_type = message.message_type
            
            # Add reaction to indicate "seen" (optional)
            if self.config.auto_react:
                await self._add_reaction(message_id, "THUMBSUP")
            
            # Parse message content
            if msg_type == "text":
                try:
                    content = json.loads(message.content).get("text", "")
                except json.JSONDecodeError:
                    content = message.content or ""
            elif msg_type == "post":
                content = self._parse_post_content(message.content or "")
                if not content:
                    content = "[post]"
            else:
                content = MSG_TYPE_MAP.get(msg_type, f"[{msg_type}]")
            
            if not content:
                return
            
            # Forward to message bus
            reply_to = chat_id if chat_type == "group" else sender_id
            await self._handle_message(
                sender_id=sender_id,
                chat_id=reply_to,
                content=content,
                metadata={
                    "message_id": message_id,
                    "chat_type": chat_type,
                    "msg_type": msg_type,
                }
            )
            
        except Exception as e:
            logger.error(f"Error processing Feishu message: {e}")
