"""Feishu/Lark channel implementation using lark-oapi SDK with WebSocket long connection."""

import asyncio
import io
import json
import mimetypes
import os
import re
import threading
import time
from collections import OrderedDict
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

from loguru import logger

from nanobot.bus.events import OutboundMessage
from nanobot.bus.queue import MessageBus
from nanobot.channels.base import BaseChannel
from nanobot.config.schema import FeishuConfig

try:
    import lark_oapi as lark
    from lark_oapi.api.im.v1 import (
        CreateFileRequest,
        CreateFileRequestBody,
        CreateImageRequest,
        CreateImageRequestBody,
        CreateMessageReactionRequest,
        CreateMessageReactionRequestBody,
        CreateMessageRequest,
        CreateMessageRequestBody,
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

        self._ensure_no_proxy([
            "open.feishu.cn",
            "open.larksuite.com",
            "msg-frontier.feishu.cn",
            "msg-frontier.larksuite.com",
        ])

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

    @staticmethod
    def _ensure_no_proxy(hosts: list[str]) -> None:
        if not hosts:
            return
        existing = os.environ.get("NO_PROXY") or os.environ.get("no_proxy") or ""
        entries = [item.strip() for item in existing.split(",") if item.strip()]
        updated = False
        for host in hosts:
            if host not in entries:
                entries.append(host)
                updated = True
        if updated:
            value = ",".join(entries)
            os.environ["NO_PROXY"] = value
            os.environ["no_proxy"] = value

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

    # Regex to match markdown tables (header + separator + data rows)
    _TABLE_RE = re.compile(
        r"((?:^[ \t]*\|.+\|[ \t]*\n)(?:^[ \t]*\|[-:\s|]+\|[ \t]*\n)(?:^[ \t]*\|.+\|[ \t]*\n?)+)",
        re.MULTILINE,
    )

    @staticmethod
    def _parse_md_table(table_text: str) -> dict | None:
        """Parse a markdown table into a Feishu table element."""
        lines = [line.strip() for line in table_text.strip().split("\n") if line.strip()]
        if len(lines) < 3:
            return None
        def split(line: str) -> list[str]:
            return [cell.strip() for cell in line.strip("|").split("|")]
        headers = split(lines[0])
        rows = [split(line) for line in lines[2:]]
        columns = [
            {"tag": "column", "name": f"c{i}", "display_name": header, "width": "auto"}
            for i, header in enumerate(headers)
        ]
        return {
            "tag": "table",
            "page_size": len(rows) + 1,
            "columns": columns,
            "rows": [
                {f"c{i}": row[i] if i < len(row) else "" for i in range(len(headers))}
                for row in rows
            ],
        }

    def _build_card_elements(self, content: str) -> list[dict]:
        """Split content into markdown + table elements for Feishu card."""
        elements: list[dict] = []
        last_end = 0
        for match in self._TABLE_RE.finditer(content):
            before = content[last_end:match.start()].strip()
            if before:
                elements.append({"tag": "markdown", "content": before})
            elements.append(
                self._parse_md_table(match.group(1)) or {"tag": "markdown", "content": match.group(1)}
            )
            last_end = match.end()
        remaining = content[last_end:].strip()
        if remaining:
            elements.append({"tag": "markdown", "content": remaining})
        return elements or [{"tag": "markdown", "content": content}]

    @staticmethod
    def _has_markdown(text: str) -> bool:
        """Check if text contains Markdown formatting."""
        md_patterns = [
            r"^#{1,6}\s",  # Headers
            r"\*\*.*\*\*",  # Bold
            r"\*.*\*",  # Italic
            r"`.*`",  # Inline code
            r"```[\s\S]*```",  # Code blocks
            r"^\s*[-*+]\s",  # Unordered lists
            r"^\s*\d+\.\s",  # Ordered lists
            r"\[.*\]\(.*\)",  # Links
            r"^>\s",  # Blockquotes
        ]
        return any(re.search(pattern, text, re.MULTILINE) for pattern in md_patterns)

    async def _send_plain_text(self, to: str, text: str, receive_id_type: str) -> bool:
        """Send plain text message."""
        content = json.dumps({"text": text})
        request = CreateMessageRequest.builder() \
            .receive_id_type(receive_id_type) \
            .request_body(
                CreateMessageRequestBody.builder()
                .receive_id(to)
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
            return False
        logger.debug(f"Feishu message sent to {to}")
        return True

    async def _send_markdown_card(self, to: str, markdown: str, receive_id_type: str) -> bool:
        """Send Markdown as interactive card."""
        elements = self._build_card_elements(markdown)
        card = {
            "config": {"wide_screen_mode": True},
            "elements": [
                {
                    "tag": "markdown",
                    "content": markdown,
                }
            ],
        }
        if elements:
            card["elements"] = elements
        content = json.dumps(card, ensure_ascii=False)

        try:
            request = CreateMessageRequest.builder() \
                .receive_id_type(receive_id_type) \
                .request_body(
                    CreateMessageRequestBody.builder()
                    .receive_id(to)
                    .msg_type("interactive")
                    .content(content)
                    .build()
                ).build()
            response = self._client.im.v1.message.create(request)
            if not response.success():
                logger.error(
                    f"Failed to send Feishu card: code={response.code}, "
                    f"msg={response.msg}, log_id={response.get_log_id()}"
                )
                return False
            logger.debug("Feishu Markdown card sent")
            return True
        except Exception as e:
            logger.error(f"Feishu Markdown card error: {e}")
            return False

    @staticmethod
    def _resolve_receive_id_type(receive_id: str) -> str:
        if receive_id.startswith("oc_"):
            return "chat_id"
        if receive_id.startswith("ou_"):
            return "open_id"
        if receive_id.startswith("on_"):
            return "union_id"
        return "open_id"

    async def send(self, msg: OutboundMessage) -> None:
        """Send a message through Feishu."""
        if not self._client:
            logger.warning("Feishu client not initialized")
            return

        try:
            t_start = time.monotonic()
            trace_id = None
            if isinstance(msg.metadata, dict):
                trace_id = msg.metadata.get("trace_id")

            # Determine receive_id_type based on message metadata or ID format.
            receive_id_type = None
            if msg.metadata:
                receive_id_type = msg.metadata.get("receive_id_type")
            if not receive_id_type:
                receive_id_type = self._resolve_receive_id_type(msg.chat_id)

            text = msg.content or ""
            text = self._append_context_status(text, msg.metadata)
            if msg.media and self._should_suppress_text_for_media(text):
                text = ""

            text_time = 0.0
            media_time = 0.0

            if text and self._has_markdown(text):
                t_text = time.monotonic()
                ok = await self._send_markdown_card(msg.chat_id, text, receive_id_type)
                text_time = time.monotonic() - t_text
                if not ok:
                    t_text = time.monotonic()
                    await self._send_plain_text(msg.chat_id, text, receive_id_type)
                    text_time += time.monotonic() - t_text
            elif text:
                t_text = time.monotonic()
                await self._send_plain_text(msg.chat_id, text, receive_id_type)
                text_time = time.monotonic() - t_text

            if msg.media:
                t_media = time.monotonic()
                await self._send_media_list(msg.media, msg.chat_id, receive_id_type, msg.reply_to)
                media_time = time.monotonic() - t_media

            total_time = time.monotonic() - t_start
            slow_threshold = 2.0
            try:
                slow_threshold = float(os.getenv("NANOBOT_CHANNEL_SEND_SLOW_S", "2"))
            except ValueError:
                slow_threshold = 2.0
            if total_time >= slow_threshold:
                logger.info(
                    f"Trace {trace_id or 'n/a'} Feishu send timings: "
                    f"text={text_time:.3f}s, media={media_time:.3f}s, total={total_time:.3f}s"
                )

        except Exception as e:
            logger.error(f"Error sending Feishu message: {e}")

    def _append_context_status(self, text: str, metadata: dict[str, Any] | None) -> str:
        if not self.config.show_context:
            return text
        if not metadata or not isinstance(metadata, dict):
            return text
        if metadata.get("stream") and not metadata.get("final", False):
            return text

        mode = metadata.get("_context_mode")
        est_tokens = metadata.get("_context_est_tokens")
        ratio = metadata.get("_context_est_ratio")
        summarized = metadata.get("_context_summarized", False)
        source = metadata.get("_context_source")
        synced_reset = metadata.get("_context_synced_reset")

        if mode is None and est_tokens is None and ratio is None:
            return text

        mode_label = {
            "native": "模型连续",
            "reset": "重新绑定",
            "stateless": "本地拼接",
        }.get(str(mode), "未知")

        parts = [
            f"会话模式：{mode_label}",
            f"LLM会话压缩：{'是' if summarized else '否'}",
        ]
        if synced_reset is not None:
            parts.append(f"同步重置：{'是' if synced_reset else '否'}")
        if source:
            parts.append(f"数据来源：{'API' if source == 'usage' else '估算'}")
        if est_tokens is not None:
            parts.append(f"估算Tokens：{est_tokens}")
        if ratio is not None:
            try:
                ratio_val = float(ratio) * 100.0
                parts.append(f"LLM Context：{ratio_val:.2f}%")
            except (TypeError, ValueError):
                parts.append(f"LLM Context：{ratio}")

        line = "｜".join(parts)

        if not text:
            return line
        return f"{text}\n\n{line}"

    @staticmethod
    def _should_suppress_text_for_media(text: str) -> bool:
        if not text:
            return False
        stripped = text.strip()
        patterns = [
            r"^\(Attempt\)\s+Sending .+ as attachment:",
            r"^Sending .+ as attachment:",
            r"(不能|无法|不支持).{0,20}(附件|文件|图片|上传)",
        ]
        return any(re.search(p, stripped) for p in patterns)

    async def _send_media_list(
        self,
        media: list[str],
        receive_id: str,
        receive_id_type: str,
        reply_to_message_id: str | None = None,
    ) -> None:
        for item in media:
            try:
                await self._send_media_item(item, receive_id, receive_id_type, reply_to_message_id)
            except Exception as e:
                logger.error(f"Error sending Feishu media {item}: {e}")

    async def _send_media_item(
        self,
        item: str,
        receive_id: str,
        receive_id_type: str,
        reply_to_message_id: str | None = None,
    ) -> None:
        media_obj, filename, content_type = self._prepare_media(item)
        is_image = self._is_image_file(filename, content_type)
        try:
            if is_image:
                image_key = self._upload_image(media_obj)
                if image_key:
                    self._send_image_message(
                        receive_id, receive_id_type, image_key, reply_to_message_id
                    )
            else:
                file_type = self._detect_file_type(filename)
                file_key = self._upload_file(media_obj, filename, file_type)
                if file_key:
                    self._send_file_message(
                        receive_id, receive_id_type, file_key, reply_to_message_id
                    )
        finally:
            if isinstance(media_obj, tuple) and len(media_obj) > 1:
                stream = media_obj[1]
                if hasattr(stream, "close") and callable(stream.close):
                    try:
                        stream.close()
                    except Exception:
                        pass
            elif hasattr(media_obj, "close") and callable(media_obj.close):
                try:
                    media_obj.close()
                except Exception:
                    pass

    def _prepare_media(self, item: str) -> tuple[io.IOBase | tuple, str, str | None]:
        if not item:
            raise ValueError("Empty media item")

        if self._is_url(item):
            data, filename, content_type = self._download_url(item)
            stream = io.BytesIO(data)
            stream.seek(0)
            return (filename, stream, content_type or "application/octet-stream"), filename, content_type

        path = self._normalize_path(item)
        if not path.exists() or not path.is_file():
            raise FileNotFoundError(f"Media file not found: {path}")

        content_type, _ = mimetypes.guess_type(path.name)
        return path.open("rb"), path.name, content_type

    @staticmethod
    def _normalize_path(item: str) -> Path:
        if item.startswith("file://"):
            parsed = urlparse(item)
            return Path(parsed.path)
        return Path(item).expanduser()

    @staticmethod
    def _is_url(item: str) -> bool:
        return item.startswith("http://") or item.startswith("https://")

    @staticmethod
    def _download_url(url: str) -> tuple[bytes, str, str | None]:
        import requests

        response = requests.get(url, timeout=30)
        response.raise_for_status()
        content_type = response.headers.get("content-type")
        filename = Path(urlparse(url).path).name or "download"
        return response.content, filename, content_type

    @staticmethod
    def _is_image_file(filename: str, content_type: str | None) -> bool:
        if content_type and content_type.startswith("image/"):
            return True
        ext = Path(filename).suffix.lower()
        return ext in {".jpg", ".jpeg", ".png", ".gif", ".webp", ".tiff", ".tif", ".bmp", ".ico"}

    @staticmethod
    def _detect_file_type(filename: str) -> str:
        ext = Path(filename).suffix.lower()
        if ext in {".opus", ".ogg"}:
            return "opus"
        if ext in {".mp4", ".mov", ".avi"}:
            return "mp4"
        if ext in {".pdf"}:
            return "pdf"
        if ext in {".doc", ".docx"}:
            return "doc"
        if ext in {".xls", ".xlsx"}:
            return "xls"
        if ext in {".ppt", ".pptx"}:
            return "ppt"
        return "stream"

    def _upload_image(self, image_obj: io.IOBase | tuple) -> str | None:
        if not self._client:
            return None
        request = CreateImageRequest.builder() \
            .request_body(
                CreateImageRequestBody.builder()
                .image_type("message")
                .image(image_obj)
                .build()
            ).build()
        response = self._client.im.v1.image.create(request)
        if not response.success():
            logger.error(
                f"Failed to upload Feishu image: code={response.code}, "
                f"msg={response.msg}, log_id={response.get_log_id()}"
            )
            return None
        return response.data.image_key if response.data else None

    def _upload_file(self, file_obj: io.IOBase | tuple, filename: str, file_type: str) -> str | None:
        if not self._client:
            return None
        request = CreateFileRequest.builder() \
            .request_body(
                CreateFileRequestBody.builder()
                .file_type(file_type)
                .file_name(filename)
                .file(file_obj)
                .build()
            ).build()
        response = self._client.im.v1.file.create(request)
        if not response.success():
            logger.error(
                f"Failed to upload Feishu file: code={response.code}, "
                f"msg={response.msg}, log_id={response.get_log_id()}"
            )
            return None
        return response.data.file_key if response.data else None

    def _send_image_message(
        self,
        receive_id: str,
        receive_id_type: str,
        image_key: str,
        reply_to_message_id: str | None = None,
    ) -> None:
        content = json.dumps({"image_key": image_key})
        if reply_to_message_id:
            try:
                from lark_oapi.api.im.v1 import ReplyMessageRequest, ReplyMessageRequestBody
                request = ReplyMessageRequest.builder() \
                    .message_id(reply_to_message_id) \
                    .request_body(
                        ReplyMessageRequestBody.builder()
                        .msg_type("image")
                        .content(content)
                        .build()
                    ).build()
                response = self._client.im.v1.message.reply(request)
            except Exception as e:
                logger.error(f"Feishu image reply error: {e}")
                return
        else:
            request = CreateMessageRequest.builder() \
                .receive_id_type(receive_id_type) \
                .request_body(
                    CreateMessageRequestBody.builder()
                    .receive_id(receive_id)
                    .msg_type("image")
                    .content(content)
                    .build()
                ).build()
            response = self._client.im.v1.message.create(request)

        if not response.success():
            logger.error(
                f"Failed to send Feishu image: code={response.code}, "
                f"msg={response.msg}, log_id={response.get_log_id()}"
            )

    def _send_file_message(
        self,
        receive_id: str,
        receive_id_type: str,
        file_key: str,
        reply_to_message_id: str | None = None,
    ) -> None:
        content = json.dumps({"file_key": file_key})
        if reply_to_message_id:
            try:
                from lark_oapi.api.im.v1 import ReplyMessageRequest, ReplyMessageRequestBody
                request = ReplyMessageRequest.builder() \
                    .message_id(reply_to_message_id) \
                    .request_body(
                        ReplyMessageRequestBody.builder()
                        .msg_type("file")
                        .content(content)
                        .build()
                    ).build()
                response = self._client.im.v1.message.reply(request)
            except Exception as e:
                logger.error(f"Feishu file reply error: {e}")
                return
        else:
            request = CreateMessageRequest.builder() \
                .receive_id_type(receive_id_type) \
                .request_body(
                    CreateMessageRequestBody.builder()
                    .receive_id(receive_id)
                    .msg_type("file")
                    .content(content)
                    .build()
                ).build()
            response = self._client.im.v1.message.create(request)

        if not response.success():
            logger.error(
                f"Failed to send Feishu file: code={response.code}, "
                f"msg={response.msg}, log_id={response.get_log_id()}"
            )

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

            sender_open_id = sender.sender_id.open_id if sender.sender_id else ""
            sender_user_id = sender.sender_id.user_id if sender.sender_id else ""
            sender_union_id = sender.sender_id.union_id if sender.sender_id else ""
            sender_id_parts = [sid for sid in (sender_open_id, sender_user_id, sender_union_id) if sid]
            sender_id = "|".join(sender_id_parts) if sender_id_parts else "unknown"
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
            elif msg_type == "image":
                try:
                    content_json = json.loads(message.content) if message.content else {}
                    image_key = content_json.get("image_key")
                    if image_key:
                        content = f"[image:{image_key}]"
                    else:
                        content = "[image]"
                except json.JSONDecodeError:
                    content = "[image]"
            elif msg_type == "file":
                try:
                    content_json = json.loads(message.content) if message.content else {}
                    file_key = content_json.get("file_key")
                    file_name = content_json.get("file_name")
                    if file_key and file_name:
                        content = f"[file:{file_name}:{file_key}]"
                    elif file_key:
                        content = f"[file:{file_key}]"
                    else:
                        content = "[file]"
                except json.JSONDecodeError:
                    content = "[file]"
            else:
                content = MSG_TYPE_MAP.get(msg_type, f"[{msg_type}]")

            if not content:
                return

            # Forward to message bus
            reply_target = sender_open_id or sender_user_id or sender_union_id or sender_id
            reply_to = chat_id if chat_type == "group" else reply_target
            receive_id_type = "chat_id" if chat_type == "group" else (
                "open_id" if sender_open_id else "user_id" if sender_user_id else "union_id"
            )

            # Session routing:
            # - For group chats: session per group chat_id
            # - For p2p chats: session per sender (stable across devices)
            session_key = (
                f"feishu:group:{chat_id}" if chat_type == "group" else f"feishu:p2p:{reply_target}"
            )

            await self._handle_message(
                sender_id=sender_id,
                chat_id=reply_to,
                content=content,
                metadata={
                    "message_id": message_id,
                    "chat_type": chat_type,
                    "msg_type": msg_type,
                    "receive_id_type": receive_id_type,
                    "session_key": session_key,
                }
            )

        except Exception as e:
            logger.error(f"Error processing Feishu message: {e}")
