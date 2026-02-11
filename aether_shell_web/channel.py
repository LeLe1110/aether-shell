"""HTTPChannel — aiohttp-based Web/PWA channel for nanobot."""

import asyncio
import hashlib
import json
import mimetypes
import os
import pathlib
import threading
import time
import uuid
from collections import OrderedDict, deque
from datetime import datetime
from typing import Any

from aiohttp import web
from loguru import logger

from aether_shell_web.auth import AuthManager
from aether_shell_web.rate_limit import RateLimiter
from nanobot.bus.events import OutboundMessage
from nanobot.bus.queue import MessageBus
from nanobot.channels.base import BaseChannel

STATIC_DIR = pathlib.Path(__file__).parent / "static"


class HTTPChannel(BaseChannel):
    """Web channel served over HTTP with SSE streaming."""

    name = "web"

    def __init__(self, config: Any, bus: MessageBus):
        super().__init__(config, bus)
        self._auth = AuthManager(
            secret=config.secret,
            expiry_days=getattr(config, "token_expiry_days", 30),
        )
        self._limiter = RateLimiter(rpm=getattr(config, "rate_limit_rpm", 20))
        self._host = getattr(config, "host", "0.0.0.0")
        self._port = getattr(config, "port", 8080)
        self._show_context = getattr(config, "show_context", False)
        self._max_upload_mb = getattr(config, "max_upload_mb", 10)
        # SSE clients: chat_id -> list of asyncio.Queue
        self._clients: dict[str, list[asyncio.Queue]] = {}
        self._app: web.Application | None = None
        self._runner: web.AppRunner | None = None
        # SSE event ID counter and ring buffer for Last-Event-ID replay
        self._event_id_counter: int = 0
        self._event_id_lock = threading.Lock()
        # chat_id -> deque of (event_id: int, payload: str)
        self._event_buffer: dict[str, deque] = {}
        self._event_buffer_size = 2000
        # Persistent sessions directory
        self._sessions_dir = pathlib.Path.home() / ".aether-shell" / "sessions"
        # Message deduplication: message_id -> None (LRU, max 1000)
        self._processed_messages: OrderedDict[str, None] = OrderedDict()
        # Media registry: file_id -> local path
        self._media_registry: dict[str, str] = {}
        # Upload directory
        self._upload_dir = pathlib.Path.home() / ".aether-shell" / "web_uploads"
        # Branding config + assets (outside repo)
        self._brand_path = pathlib.Path(
            getattr(config, "brand_path", str(pathlib.Path.home() / ".aether-shell" / "brand.json"))
        ).expanduser()
        self._brand_assets_dir = pathlib.Path(
            getattr(config, "brand_assets_dir", str(pathlib.Path.home() / ".aether-shell" / "brand-assets"))
        ).expanduser()
        # Subscribe to outbound messages
        self.bus.subscribe_outbound("web", self.send)

    async def start(self) -> None:
        self._running = True
        max_size = self._max_upload_mb * 1024 * 1024
        self._app = web.Application(client_max_size=max_size)
        self._app.router.add_post("/api/auth/login", self._handle_login)
        self._app.router.add_get("/api/auth/check", self._handle_auth_check)
        self._app.router.add_post("/api/messages", self._handle_send_message)
        self._app.router.add_get("/api/messages/stream", self._handle_sse)
        self._app.router.add_get("/api/sessions", self._handle_list_sessions)
        self._app.router.add_get("/api/sessions/{session_id}/messages", self._handle_session_messages)
        self._app.router.add_post("/api/sessions/new", self._handle_new_session)
        self._app.router.add_post("/api/sessions/switch", self._handle_switch_session)
        self._app.router.add_post("/api/upload", self._handle_upload)
        self._app.router.add_get("/api/media/{file_id}", self._handle_media)
        # Static files and SPA fallback
        self._app.router.add_get("/", self._handle_index)
        self._app.router.add_get("/css/{filename}", self._handle_static_asset)
        self._app.router.add_get("/js/{filename}", self._handle_static_asset)
        self._app.router.add_get("/icons/{filename}", self._handle_static_asset)
        self._app.router.add_get("/manifest.json", self._handle_manifest)
        self._app.router.add_get("/brand.json", self._handle_brand)
        self._app.router.add_get("/brand-assets/{path:.*}", self._handle_brand_asset)
        self._app.router.add_get("/sw.js", self._handle_static_file)

        self._runner = web.AppRunner(self._app)
        await self._runner.setup()
        site = web.TCPSite(self._runner, self._host, self._port)
        await site.start()
        logger.info(f"Web channel listening on http://{self._host}:{self._port}")

        # Start background upload cleanup task
        self._cleanup_task = asyncio.create_task(self._cleanup_uploads_loop())

        # Keep running until stopped
        try:
            while self._running:
                await asyncio.sleep(1)
        except asyncio.CancelledError:
            pass

    async def stop(self) -> None:
        self._running = False
        if hasattr(self, "_cleanup_task"):
            self._cleanup_task.cancel()
        if self._runner:
            await self._runner.cleanup()
        logger.info("Web channel stopped")

    async def send(self, msg: OutboundMessage) -> None:
        """Push an outbound message to all SSE clients for the chat_id."""
        t_start = time.monotonic()
        chat_id = msg.chat_id
        meta = msg.metadata or {}

        # Skip messages flagged to suppress SSE delivery (e.g. /new greeting
        # which is delivered via the HTTP response instead).
        if meta.get("_suppress_outbound"):
            return

        queues = self._clients.get(chat_id, [])
        has_clients = bool(queues)
        if not has_clients:
            logger.debug(f"Web send: no SSE clients for chat_id={chat_id}")
        is_stream = meta.get("stream", False)
        stream_id = meta.get("stream_id")
        is_final = meta.get("final", False)

        # Extract session_id from metadata (session_key format: "web:chat_id:session_id")
        session_key = meta.get("session_key", "")
        session_id = None
        if session_key and session_key.startswith(f"web:{chat_id}:"):
            session_id = session_key[len(f"web:{chat_id}:"):]

        if is_stream and not is_final:
            event_data = {
                "type": "delta",
                "stream_id": stream_id,
                "content": msg.content,
                "chat_id": chat_id,
            }
            if session_id:
                event_data["session_id"] = session_id
            event_name = "delta"
        else:
            event_data = {
                "type": "message",
                "content": msg.content,
                "chat_id": chat_id,
                "stream_id": stream_id,
                "role": "assistant",
            }
            if session_id:
                event_data["session_id"] = session_id
            event_name = "message"

            # Context metadata (only on final/complete messages, when enabled)
            if self._show_context:
                ctx = {}
                if meta.get("_context_mode"):
                    ctx["mode"] = meta["_context_mode"]
                if meta.get("_context_est_tokens") is not None:
                    ctx["est_tokens"] = meta["_context_est_tokens"]
                if meta.get("_context_est_ratio") is not None:
                    ctx["est_ratio"] = meta["_context_est_ratio"]
                if meta.get("_context_summarized"):
                    ctx["summarized"] = True
                if meta.get("_context_source"):
                    ctx["source"] = meta["_context_source"]
                if meta.get("_context_synced_reset") is not None:
                    ctx["synced_reset"] = meta["_context_synced_reset"]
                if ctx:
                    event_data["context"] = ctx

                timing = {}
                if meta.get("_agent_total_s") is not None:
                    timing["total_s"] = meta["_agent_total_s"]
                if meta.get("_agent_llm_s") is not None:
                    timing["llm_s"] = meta["_agent_llm_s"]
                if timing:
                    event_data["timing"] = timing

            # Media attachments (bot -> user)
            media_items = getattr(msg, "media", None) or meta.get("media")
            if media_items:
                media_list = []
                for item in media_items:
                    item_path = str(item)
                    file_id = hashlib.sha256(item_path.encode()).hexdigest()[:16]
                    self._media_registry[file_id] = item_path
                    filename = pathlib.Path(item_path).name
                    content_type = mimetypes.guess_type(filename)[0] or "application/octet-stream"
                    is_image = content_type.startswith("image/")
                    media_list.append({
                        "file_id": file_id,
                        "filename": filename,
                        "content_type": content_type,
                        "is_image": is_image,
                    })
                event_data["media"] = media_list

        # Assign incremental event ID for Last-Event-ID support
        with self._event_id_lock:
            self._event_id_counter += 1
            event_id = self._event_id_counter
        payload = f"id: {event_id}\nevent: {event_name}\ndata: {json.dumps(event_data, ensure_ascii=False)}\n\n"

        # Buffer event for replay on reconnect
        if chat_id not in self._event_buffer:
            self._event_buffer[chat_id] = deque(maxlen=self._event_buffer_size)
        self._event_buffer[chat_id].append((event_id, payload))

        if has_clients:
            dead: list[asyncio.Queue] = []
            for q in queues:
                try:
                    q.put_nowait(payload)
                except asyncio.QueueFull:
                    dead.append(q)
            for q in dead:
                queues.remove(q)

        # Performance monitoring
        elapsed = time.monotonic() - t_start
        trace_id = meta.get("trace_id", "n/a")
        slow_threshold = float(os.environ.get("NANOBOT_CHANNEL_SEND_SLOW_S", "2"))
        if elapsed >= slow_threshold:
            logger.info(f"Trace {trace_id} Web send slow: {elapsed:.3f}s")
        else:
            logger.debug(f"Trace {trace_id} Web send: {elapsed:.3f}s")

    # ---- Auth routes ----

    async def _handle_login(self, request: web.Request) -> web.Response:
        try:
            body = await request.json()
        except Exception:
            return web.json_response({"error": "invalid json"}, status=400)

        invite_code = body.get("invite_code", "")
        token = self._auth.login(invite_code)
        if token is None:
            return web.json_response({"error": "invalid invite code"}, status=401)

        payload = self._auth.validate(token)
        chat_id = payload["chat_id"] if payload else ""
        return web.json_response({"token": token, "chat_id": chat_id})

    async def _handle_auth_check(self, request: web.Request) -> web.Response:
        payload = self._extract_auth(request)
        if payload is None:
            return web.json_response({"valid": False}, status=401)
        return web.json_response({"valid": True, "chat_id": payload["chat_id"]})

    # ---- Message routes ----

    async def _handle_send_message(self, request: web.Request) -> web.Response:
        payload = self._extract_auth(request)
        if payload is None:
            return web.json_response({"error": "unauthorized"}, status=401)

        chat_id = payload["chat_id"]
        sender_id = payload.get("sub", chat_id)

        if not self.is_allowed(sender_id):
            return web.json_response({"error": "forbidden"}, status=403)

        if not self._limiter.check(chat_id):
            return web.json_response({"error": "rate limited"}, status=429)

        try:
            body = await request.json()
        except Exception:
            return web.json_response({"error": "invalid json"}, status=400)

        content = body.get("content", "").strip()
        if not content:
            return web.json_response({"error": "empty message"}, status=400)

        # Message deduplication
        message_id = body.get("message_id")
        if message_id:
            if message_id in self._processed_messages:
                return web.json_response({"status": "duplicate"})
            self._processed_messages[message_id] = None
            if len(self._processed_messages) > 1000:
                self._processed_messages.popitem(last=False)

        session_id = body.get("session_id")
        metadata: dict[str, Any] = {}
        if session_id:
            # Pass the full session key so the agent can locate the exact session
            # without relying on active.json (avoids race conditions).
            metadata["session_key"] = f"web:{chat_id}:{session_id}"
            logger.debug(
                f"_handle_send_message: session_id={session_id} "
                f"session_key={metadata['session_key']}"
            )

        # Validate media paths (uploaded files)
        media_paths = body.get("media")
        validated_media = None
        if media_paths and isinstance(media_paths, list):
            validated_media = []
            upload_root = str(self._upload_dir.resolve())
            for p in media_paths:
                resolved = str(pathlib.Path(p).resolve())
                if resolved.startswith(upload_root) and pathlib.Path(resolved).exists():
                    validated_media.append(resolved)

        await self._handle_message(
            sender_id=sender_id,
            chat_id=chat_id,
            content=content,
            metadata=metadata,
            media=validated_media,
        )

        # Broadcast user message to all SSE clients so other devices see it
        user_event_data = {
            "type": "user_message",
            "content": content,
            "chat_id": chat_id,
            "role": "user",
            "message_id": message_id or "",
        }
        # Add session_id if available
        if session_id:
            user_event_data["session_id"] = session_id
        with self._event_id_lock:
            self._event_id_counter += 1
            eid = self._event_id_counter
        user_payload = f"id: {eid}\nevent: user_message\ndata: {json.dumps(user_event_data, ensure_ascii=False)}\n\n"
        if chat_id not in self._event_buffer:
            self._event_buffer[chat_id] = deque(maxlen=self._event_buffer_size)
        self._event_buffer[chat_id].append((eid, user_payload))
        for q in list(self._clients.get(chat_id, [])):
            try:
                q.put_nowait(user_payload)
            except asyncio.QueueFull:
                pass

        return web.json_response({"status": "ok"})

    # ---- SSE ----

    async def _handle_sse(self, request: web.Request) -> web.StreamResponse:
        token = request.query.get("token", "")
        payload = self._auth.validate(token)
        if payload is None:
            return web.json_response({"error": "unauthorized"}, status=401)

        chat_id = payload["chat_id"]
        queue: asyncio.Queue = asyncio.Queue(maxsize=256)

        if chat_id not in self._clients:
            self._clients[chat_id] = []
        self._clients[chat_id].append(queue)

        response = web.StreamResponse(
            status=200,
            reason="OK",
            headers={
                "Content-Type": "text/event-stream",
                "Cache-Control": "no-cache",
                "Connection": "keep-alive",
                "X-Accel-Buffering": "no",
            },
        )
        await response.prepare(request)

        # Send connected event
        connected_payload = f"event: connected\ndata: {json.dumps({'chat_id': chat_id})}\n\n"
        await response.write(connected_payload.encode("utf-8"))

        # Replay missed events if client sends Last-Event-ID
        last_event_id_str = request.headers.get("Last-Event-ID") or request.query.get("lastEventId", "")
        if last_event_id_str:
            try:
                last_event_id = int(last_event_id_str)
                buf = self._event_buffer.get(chat_id)
                if buf:
                    for eid, epayload in buf:
                        if eid > last_event_id:
                            await response.write(epayload.encode("utf-8"))
            except (ValueError, TypeError):
                pass

        try:
            while True:
                try:
                    data = await asyncio.wait_for(queue.get(), timeout=30.0)
                    await response.write(data.encode("utf-8"))
                except asyncio.TimeoutError:
                    # Keepalive comment
                    await response.write(b": keepalive\n\n")
                except ConnectionResetError:
                    break
        except (asyncio.CancelledError, ConnectionResetError):
            pass
        finally:
            if chat_id in self._clients and queue in self._clients[chat_id]:
                self._clients[chat_id].remove(queue)
                if not self._clients[chat_id]:
                    del self._clients[chat_id]

        return response

    # ---- Sessions (disk-based) ----

    def _scan_sessions_for_chat(self, chat_id: str) -> list[dict]:
        """Scan ~/.aether-shell/sessions/ for JSONL files belonging to this chat_id."""
        if not self._sessions_dir.exists():
            return []

        # Web session files are named like: web_{chat_id}_{session_name}#timestamp.jsonl
        prefix = f"web_{chat_id}_"
        results = []

        for path in self._sessions_dir.glob("*.jsonl"):
            fname = path.stem  # e.g. web_web_d1a7a68d0b3e_default#20260208000954
            if not fname.startswith(prefix):
                continue

            # Parse session_id from filename
            # fname = "web_web_abc123_default#20260208000954"
            # after stripping prefix "web_web_abc123_" → "default#20260208000954"
            suffix = fname[len(prefix):]
            # session_id is the full suffix (used to reconstruct session_key)
            session_id = suffix

            # Read the JSONL file to get metadata and first user message
            title = ""
            message_count = 0
            created_at = ""
            updated_at = ""
            try:
                with open(path) as f:
                    for line in f:
                        line = line.strip()
                        if not line:
                            continue
                        data = json.loads(line)
                        if data.get("_type") == "metadata":
                            created_at = data.get("created_at", "")
                            updated_at = data.get("updated_at", "")
                        else:
                            message_count += 1
                            if not title and data.get("role") == "user":
                                # Use first user message as title (truncated)
                                raw = data.get("content", "")
                                title = raw[:30] + ("..." if len(raw) > 30 else "")
            except Exception:
                continue

            if not title:
                # Fallback: use date from filename
                title = "New Chat"
                if "#" in suffix:
                    ts_part = suffix.split("#")[-1]
                    try:
                        dt = datetime.strptime(ts_part, "%Y%m%d%H%M%S")
                        title = dt.strftime("%m/%d %H:%M")
                    except ValueError:
                        pass

            results.append({
                "session_id": session_id,
                "title": title,
                "message_count": message_count,
                "created_at": created_at,
                "updated_at": updated_at,
            })

        # Sort by updated_at descending (most recent first)
        results.sort(key=lambda x: x.get("updated_at", ""), reverse=True)

        # Mark the active one
        active_key = self._get_active_session_key(chat_id)
        for r in results:
            r["active"] = (r["session_id"] == active_key)

        return results

    def _get_active_session_key(self, chat_id: str) -> str | None:
        """Read active.json to find which session_id is active for this chat_id."""
        active_path = self._sessions_dir / "active.json"
        if not active_path.exists():
            return None
        try:
            with open(active_path) as f:
                data = json.load(f)
            # Look for keys matching web:{chat_id}:*
            for base_key, active_val in data.items():
                if base_key.startswith(f"web:{chat_id}:"):
                    # active_val = "web:chat_id:session_name#timestamp"
                    # We need the suffix after "web_{chat_id}_" in filename form
                    # Convert colon-key to filename form
                    # base_key = "web:chat_id:default" → file prefix "web_chat_id_default"
                    # active_val = "web:chat_id:default#20260208000954"
                    # suffix after base prefix = "default#20260208000954"
                    base_prefix = f"web:{chat_id}:"
                    if active_val.startswith(base_prefix):
                        return active_val[len(base_prefix):]
            return None
        except Exception:
            return None

    def _read_session_messages(self, chat_id: str, session_id: str, limit: int = 50) -> list[dict]:
        """Read messages from a session JSONL file."""
        prefix = f"web_{chat_id}_"
        path = self._sessions_dir / f"{prefix}{session_id}.jsonl"
        if not path.exists():
            return []

        messages = []
        try:
            with open(path) as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    data = json.loads(line)
                    if data.get("_type") == "metadata":
                        continue
                    msg: dict = {
                        "role": data.get("role", ""),
                        "content": data.get("content", ""),
                        "timestamp": data.get("timestamp", ""),
                    }
                    # Include media metadata when present so the frontend
                    # can re-render attachments after a catchUp reload.
                    media_paths: list[str] = data.get("media") or []
                    if media_paths:
                        media_list = []
                        for item_path in media_paths:
                            item_path = str(item_path)
                            file_id = hashlib.sha256(item_path.encode()).hexdigest()[:16]
                            # Ensure the file is registered so /api/media/ can serve it
                            self._media_registry[file_id] = item_path
                            filename = pathlib.Path(item_path).name
                            content_type = mimetypes.guess_type(filename)[0] or "application/octet-stream"
                            is_image = content_type.startswith("image/")
                            media_list.append({
                                "file_id": file_id,
                                "filename": filename,
                                "content_type": content_type,
                                "is_image": is_image,
                            })
                        msg["media"] = media_list
                    messages.append(msg)
        except Exception:
            return []

        # Return last N messages
        if len(messages) > limit:
            messages = messages[-limit:]
        return messages

    async def _handle_list_sessions(self, request: web.Request) -> web.Response:
        payload = self._extract_auth(request)
        if payload is None:
            return web.json_response({"error": "unauthorized"}, status=401)
        chat_id = payload["chat_id"]
        sessions = self._scan_sessions_for_chat(chat_id)
        return web.json_response({"sessions": sessions})

    async def _handle_session_messages(self, request: web.Request) -> web.Response:
        payload = self._extract_auth(request)
        if payload is None:
            return web.json_response({"error": "unauthorized"}, status=401)
        chat_id = payload["chat_id"]
        session_id = request.match_info.get("session_id", "")
        limit = int(request.query.get("limit", "50"))
        messages = self._read_session_messages(chat_id, session_id, limit)
        return web.json_response({"messages": messages})

    async def _handle_new_session(self, request: web.Request) -> web.Response:
        payload = self._extract_auth(request)
        if payload is None:
            return web.json_response({"error": "unauthorized"}, status=401)

        chat_id = payload["chat_id"]
        sender_id = payload.get("sub", chat_id)

        # Send /new command through the agent pipeline so SessionManager
        # creates the new session in its own memory cache and on disk.
        # Mark _suppress_outbound so the SSE channel won't push the greeting
        # (the HTTP response includes it instead, avoiding a race condition).
        await self._handle_message(
            sender_id=sender_id,
            chat_id=chat_id,
            content="/new",
            metadata={
                "session_key": f"web:{chat_id}:default",
                "_suppress_outbound": True,
            },
        )

        # The /new is processed async via the message bus.
        # Poll active.json briefly to pick up the new session_id.
        old_key = self._get_active_session_key(chat_id)
        session_id = old_key or "default"
        for _ in range(20):  # up to 2 seconds
            await asyncio.sleep(0.1)
            new_key = self._get_active_session_key(chat_id)
            if new_key and new_key != old_key:
                session_id = new_key
                break

        return web.json_response({
            "session": {
                "session_id": session_id,
                "title": "New Chat",
                "greeting": "✅ 已开启新会话（历史已保留）。你好！我能帮你做什么？",
            }
        })

    async def _handle_switch_session(self, request: web.Request) -> web.Response:
        """Switch the active session pointer to a specific historical session."""
        payload = self._extract_auth(request)
        if payload is None:
            return web.json_response({"error": "unauthorized"}, status=401)

        chat_id = payload["chat_id"]
        try:
            body = await request.json()
        except Exception:
            return web.json_response({"error": "invalid json"}, status=400)

        session_id = body.get("session_id", "")
        if not session_id:
            return web.json_response({"error": "missing session_id"}, status=400)

        # Verify the session file exists
        prefix = f"web_{chat_id}_"
        path = self._sessions_dir / f"{prefix}{session_id}.jsonl"
        if not path.exists():
            return web.json_response({"error": "session not found"}, status=404)

        # Mark pending_reset in the session file so the agent uses reset mode
        # (avoids relying on a stale previous_response_id from an old session)
        self._mark_pending_reset(path)

        # Determine base_key from session_id
        # session_id = "default#20260208000954" → base_name = "default"
        base_name = session_id.split("#")[0] if "#" in session_id else session_id
        base_key = f"web:{chat_id}:{base_name}"
        active_key = f"web:{chat_id}:{session_id}"
        self._update_active_index(base_key, active_key)

        return web.json_response({"status": "ok"})

    def _mark_pending_reset(self, session_path: pathlib.Path) -> None:
        """Set pending_reset=True in a session JSONL file's metadata."""
        try:
            lines = session_path.read_text().splitlines()
            if not lines:
                return
            first = json.loads(lines[0])
            if first.get("_type") != "metadata":
                return
            meta = first.setdefault("metadata", {})
            llm_session = meta.setdefault("llm_session", {})
            llm_session["pending_reset"] = True
            lines[0] = json.dumps(first, ensure_ascii=False)
            session_path.write_text("\n".join(lines) + "\n")
        except Exception as e:
            logger.warning(f"Failed to mark pending_reset: {e}")

    def _update_active_index(self, base_key: str, active_key: str) -> None:
        """Update ~/.aether-shell/sessions/active.json."""
        active_path = self._sessions_dir / "active.json"
        data = {}
        if active_path.exists():
            try:
                with open(active_path) as f:
                    data = json.load(f)
            except Exception:
                data = {}
        data[base_key] = active_key
        try:
            with open(active_path, "w") as f:
                json.dump(data, f, indent=2)
        except Exception:
            logger.warning(f"Failed to update active.json: {base_key} -> {active_key}")

    # ---- Static files ----

    def _no_cache_headers(self, resp: web.StreamResponse) -> None:
        """Apply aggressive no-cache headers (defeats Cloudflare edge cache)."""
        resp.headers["Cache-Control"] = "no-cache, no-store, must-revalidate, max-age=0"
        resp.headers["Pragma"] = "no-cache"
        resp.headers["Expires"] = "0"
        resp.headers["CDN-Cache-Control"] = "no-store"
        resp.headers["Cloudflare-CDN-Cache-Control"] = "no-store"

    async def _handle_index(self, request: web.Request) -> web.Response:
        filepath = STATIC_DIR / "index.html"
        content = filepath.read_text("utf-8")
        resp = web.Response(text=content, content_type="text/html", charset="utf-8")
        self._no_cache_headers(resp)
        return resp

    async def _handle_static_file(self, request: web.Request) -> web.Response:
        filename = request.path.lstrip("/")
        filepath = STATIC_DIR / filename
        if not filepath.exists():
            filepath = STATIC_DIR / "index.html"
        resp = web.FileResponse(filepath)
        self._no_cache_headers(resp)
        return resp

    async def _handle_static_asset(self, request: web.Request) -> web.Response:
        """Serve CSS/JS/icon files with no-cache headers."""
        rel_path = request.path.lstrip("/")
        filepath = STATIC_DIR / rel_path
        if not filepath.exists() or not filepath.is_file():
            return web.Response(status=404, text="Not found")
        resp = web.FileResponse(filepath)
        self._no_cache_headers(resp)
        return resp

    def _load_json_file(self, filepath: pathlib.Path) -> dict:
        if not filepath.exists():
            return {}
        try:
            return json.loads(filepath.read_text("utf-8"))
        except Exception:
            logger.warning(f"Failed to read {filepath}")
            return {}

    def _load_static_json(self, filename: str) -> dict:
        return self._load_json_file(STATIC_DIR / filename)

    def _load_brand_json(self) -> dict:
        brand = self._load_json_file(self._brand_path)
        if brand:
            return brand
        # Optional fallback for development (if a static brand.json exists)
        return self._load_static_json("brand.json")

    def _build_manifest(self) -> dict:
        manifest = self._load_static_json("manifest.json")
        if not isinstance(manifest, dict):
            manifest = {}
        brand = self._load_brand_json()
        if not isinstance(brand, dict):
            brand = {}

        product_name = brand.get("productName")
        if product_name:
            manifest["name"] = product_name
            manifest["short_name"] = brand.get("shortName") or product_name

        tagline = brand.get("tagline")
        if tagline:
            manifest["description"] = tagline

        theme_color = brand.get("themeColor")
        if theme_color:
            manifest["theme_color"] = theme_color

        background_color = brand.get("backgroundColor")
        if background_color:
            manifest["background_color"] = background_color

        manifest_icons = brand.get("manifestIcons")
        if isinstance(manifest_icons, list):
            manifest["icons"] = manifest_icons

        brand_manifest = brand.get("manifest")
        if isinstance(brand_manifest, dict):
            manifest.update(brand_manifest)

        return manifest

    async def _handle_manifest(self, request: web.Request) -> web.Response:
        manifest = self._build_manifest()
        resp = web.json_response(manifest)
        self._no_cache_headers(resp)
        return resp

    async def _handle_brand(self, request: web.Request) -> web.Response:
        brand = self._load_brand_json()
        resp = web.json_response(brand)
        self._no_cache_headers(resp)
        return resp

    async def _handle_brand_asset(self, request: web.Request) -> web.Response:
        rel_path = request.match_info.get("path", "")
        if not rel_path:
            return web.Response(status=404, text="Not found")
        base = self._brand_assets_dir.resolve()
        candidate = (base / rel_path).resolve()
        if base not in candidate.parents and candidate != base:
            return web.Response(status=403, text="Forbidden")
        if not candidate.exists() or not candidate.is_file():
            return web.Response(status=404, text="Not found")
        resp = web.FileResponse(candidate)
        self._no_cache_headers(resp)
        return resp

    # ---- Upload (user -> bot) ----

    async def _handle_upload(self, request: web.Request) -> web.Response:
        payload = self._extract_auth(request)
        if payload is None:
            return web.json_response({"error": "unauthorized"}, status=401)

        chat_id = payload["chat_id"]
        reader = await request.multipart()
        if reader is None:
            return web.json_response({"error": "multipart required"}, status=400)

        results = []
        while True:
            part = await reader.next()
            if part is None:
                break
            if part.filename is None:
                continue

            filename = pathlib.Path(part.filename).name  # sanitize
            file_id = uuid.uuid4().hex[:8]
            chat_dir = self._upload_dir / chat_id
            chat_dir.mkdir(parents=True, exist_ok=True)
            dest = chat_dir / f"{file_id}_{filename}"

            size = 0
            max_bytes = self._max_upload_mb * 1024 * 1024
            with open(dest, "wb") as f:
                while True:
                    chunk = await part.read_chunk(8192)
                    if not chunk:
                        break
                    size += len(chunk)
                    if size > max_bytes:
                        f.close()
                        dest.unlink(missing_ok=True)
                        return web.json_response(
                            {"error": f"file too large (max {self._max_upload_mb}MB)"},
                            status=413,
                        )
                    f.write(chunk)

            content_type = mimetypes.guess_type(filename)[0] or "application/octet-stream"
            results.append({
                "file_id": file_id,
                "filename": filename,
                "content_type": content_type,
                "path": str(dest),
            })

        if not results:
            return web.json_response({"error": "no files uploaded"}, status=400)

        return web.json_response({"files": results})

    # ---- Media service (bot -> user) ----

    async def _handle_media(self, request: web.Request) -> web.Response:
        # Auth via query param
        token_str = request.query.get("token", "")
        payload = self._auth.validate(token_str)
        if payload is None:
            return web.json_response({"error": "unauthorized"}, status=401)

        file_id = request.match_info.get("file_id", "")
        path_str = self._media_registry.get(file_id)
        if not path_str:
            return web.json_response({"error": "not found"}, status=404)

        file_path = pathlib.Path(path_str)
        if not file_path.exists():
            return web.json_response({"error": "file not found"}, status=404)

        content_type = mimetypes.guess_type(file_path.name)[0] or "application/octet-stream"
        return web.FileResponse(
            file_path,
            headers={
                "Content-Type": content_type,
                "Content-Disposition": f'inline; filename="{file_path.name}"',
            },
        )

    # ---- Background cleanup ----

    async def _cleanup_uploads_loop(self) -> None:
        """Periodically clean up uploaded files older than 24 hours."""
        while True:
            try:
                await asyncio.sleep(3600)  # every hour
                cutoff = time.time() - 86400  # 24 hours
                if not self._upload_dir.exists():
                    continue
                for chat_dir in self._upload_dir.iterdir():
                    if not chat_dir.is_dir():
                        continue
                    for f in chat_dir.iterdir():
                        try:
                            if f.stat().st_mtime < cutoff:
                                f.unlink()
                        except OSError:
                            pass
                    # Remove empty chat dirs
                    try:
                        if not any(chat_dir.iterdir()):
                            chat_dir.rmdir()
                    except OSError:
                        pass
            except asyncio.CancelledError:
                return
            except Exception:
                logger.exception("Upload cleanup error")

    # ---- Helpers ----

    def _extract_auth(self, request: web.Request) -> dict | None:
        """Extract and validate JWT from Authorization header or query param."""
        auth_header = request.headers.get("Authorization", "")
        if auth_header.startswith("Bearer "):
            token = auth_header[7:]
        else:
            token = request.query.get("token", "")
        if not token:
            return None
        return self._auth.validate(token)
