"""Context management for conversation sessions."""

from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import datetime
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from nanobot.session.manager import Session

from loguru import logger

from nanobot.agent.context import ContextBuilder
from nanobot.config.schema import ContextConfig
from nanobot.providers.base import LLMProvider, LLMResponse


@dataclass
class ContextBundle:
    messages: list[dict[str, Any]]
    session_state: dict[str, Any] | None
    stats: dict[str, Any]


class ContextManager:
    """Prepare conversation context, summaries, and native session state."""

    def __init__(
        self,
        provider: LLMProvider,
        config: ContextConfig,
        builder: ContextBuilder,
        default_model: str,
    ) -> None:
        self.provider = provider
        self.config = config
        self.builder = builder
        self.default_model = default_model

    async def build_context(
        self,
        session: "Session",
        current_message: str,
        media: list[str] | None,
        channel: str | None,
        chat_id: str | None,
    ) -> ContextBundle:
        from nanobot.session.manager import Session

        if not isinstance(session, Session):
            raise TypeError("session must be a Session")

        ctx_meta = session.metadata.setdefault("context", {})
        llm_meta = session.metadata.setdefault("llm_session", {})

        summary = str(ctx_meta.get("summary") or "").strip()
        summary_index = int(ctx_meta.get("summary_index") or 0)
        if summary_index < 0:
            summary_index = 0
        if summary_index > len(session.messages):
            summary_index = len(session.messages)

        native_enabled = bool(self.config.enable_native_session)
        native_supported = bool(self.provider.supports_native_session())
        native_ready = native_enabled and native_supported

        summary, summary_index, summarized = await self._maybe_summarize(
            session=session,
            summary=summary,
            summary_index=summary_index,
        )

        ctx_meta["summary"] = summary
        ctx_meta["summary_index"] = summary_index
        if summarized:
            ctx_meta["summary_updated_at"] = datetime.now().isoformat()

        pending_reset = bool(llm_meta.get("pending_reset"))
        if summarized and native_ready:
            pending_reset = True

        bootstrap_fingerprint = self.builder.get_bootstrap_fingerprint()
        stored_fingerprint = llm_meta.get("bootstrap_fingerprint")
        needs_bootstrap_reset = False
        if native_ready:
            if stored_fingerprint is None and llm_meta.get("previous_response_id"):
                needs_bootstrap_reset = True
            elif stored_fingerprint and stored_fingerprint != bootstrap_fingerprint:
                needs_bootstrap_reset = True
        if needs_bootstrap_reset:
            pending_reset = True
            llm_meta["pending_reset"] = True
        llm_meta["bootstrap_fingerprint"] = bootstrap_fingerprint
        last_ratio = float(llm_meta.get("last_context_ratio") or 0.0)
        force_reset = pending_reset or (last_ratio >= self.config.hard_limit_threshold)

        remaining = session.messages[summary_index:]
        remaining_tokens = self._estimate_messages_tokens(
            [{"role": "assistant", "content": summary}] if summary else []
        ) + self._estimate_messages_tokens(
            [{"role": m.get("role"), "content": m.get("content")} for m in remaining]
        )
        remaining_ratio = remaining_tokens / self._effective_window()
        if native_ready and not force_reset and remaining_ratio >= self.config.hard_limit_threshold:
            force_reset = True

        messages: list[dict[str, Any]]
        session_state: dict[str, Any] | None
        mode = "stateless"

        if native_ready and not force_reset and llm_meta.get("previous_response_id"):
            # Continue native server-side session; send only new user input.
            session_state = {"previous_response_id": llm_meta.get("previous_response_id")}
            messages = self.builder.build_messages(
                history=[],
                current_message=current_message,
                media=media,
                channel=channel,
                chat_id=chat_id,
                include_system=False,
            )
            mode = "native"
        else:
            # New or reset session: seed with summary + recent history.
            recent = self._select_recent_messages(session, summary_index)
            messages = self.builder.build_messages(
                history=recent,
                current_message=current_message,
                media=media,
                channel=channel,
                chat_id=chat_id,
                summary=summary or None,
                include_system=True,
            )
            messages = self._shrink_history_to_budget(
                messages=messages,
                summary=summary,
                recent=recent,
                current_message=current_message,
                media=media,
                channel=channel,
                chat_id=chat_id,
            )
            session_state = None if not native_ready else {}
            mode = "reset" if native_ready and force_reset else "stateless"
            if native_ready and force_reset:
                llm_meta["previous_response_id"] = None
                llm_meta["pending_reset"] = False
                llm_meta["last_reset_at"] = datetime.now().isoformat()

        estimated_tokens = self._estimate_messages_tokens(messages)
        ratio = estimated_tokens / self._effective_window()
        llm_meta["last_local_tokens"] = estimated_tokens
        llm_meta["last_local_ratio"] = round(ratio, 4)

        synced_reset = bool(mode == "reset" and native_ready)
        stats = {
            "mode": mode,
            "estimated_tokens": estimated_tokens,
            "estimated_ratio": round(ratio, 4),
            "conversation_tokens": remaining_tokens,
            "conversation_ratio": round(remaining_ratio, 4),
            "summary_chars": len(summary),
            "summary_index": summary_index,
            "summarized": summarized,
            "synced_reset": synced_reset,
            "native_supported": native_supported,
            "native_enabled": native_enabled,
        }

        return ContextBundle(messages=messages, session_state=session_state, stats=stats)

    def _shrink_history_to_budget(
        self,
        messages: list[dict[str, Any]],
        summary: str,
        recent: list[dict[str, Any]],
        current_message: str,
        media: list[str] | None,
        channel: str | None,
        chat_id: str | None,
    ) -> list[dict[str, Any]]:
        if not recent:
            return messages

        min_recent = max(1, int(self.config.min_recent_messages))
        working = list(recent)
        budget = self._effective_window()
        estimated = self._estimate_messages_tokens(messages)
        if estimated <= budget:
            return messages

        while len(working) > min_recent and estimated > budget:
            working = working[1:]
            messages = self.builder.build_messages(
                history=working,
                current_message=current_message,
                media=media,
                channel=channel,
                chat_id=chat_id,
                summary=summary or None,
                include_system=True,
            )
            estimated = self._estimate_messages_tokens(messages)

        return messages

    def update_after_response(
        self,
        session: "Session",
        response: LLMResponse,
    ) -> None:
        from nanobot.session.manager import Session

        if not isinstance(session, Session):
            return

        llm_meta = session.metadata.setdefault("llm_session", {})

        # On LLM error, clear native session state to prevent stale ID usage
        if response.finish_reason == "error":
            if llm_meta.get("previous_response_id"):
                logger.warning("LLM error detected, clearing native session state")
                llm_meta["previous_response_id"] = None
                llm_meta["pending_reset"] = True
            return

        # Only store valid Responses API IDs (resp_* prefix)
        if response.response_id and response.response_id.startswith("resp_"):
            llm_meta["previous_response_id"] = response.response_id
        if response.conversation_id:
            llm_meta["conversation_id"] = response.conversation_id
        if response.model:
            llm_meta["model"] = response.model
        if response.usage:
            llm_meta["last_usage"] = response.usage
            prompt_tokens = response.usage.get("prompt_tokens") or response.usage.get("input_tokens")
            if isinstance(prompt_tokens, int) and prompt_tokens > 0:
                ratio = prompt_tokens / self._effective_window()
                llm_meta["last_context_tokens"] = prompt_tokens
                llm_meta["last_context_ratio"] = round(ratio, 4)
                if ratio >= self.config.hard_limit_threshold:
                    llm_meta["pending_reset"] = True

        finish_reason = response.finish_reason or ""
        if isinstance(finish_reason, str) and "length" in finish_reason.lower():
            llm_meta["pending_reset"] = True

    def _select_recent_messages(self, session: "Session", summary_index: int) -> list[dict[str, Any]]:
        total = len(session.messages)
        if total == 0:
            return []

        recent_target = max(1, int(self.config.recent_messages))
        min_recent = max(1, int(self.config.min_recent_messages))

        start = max(summary_index, total - recent_target)
        recent = session.messages[start:]

        if len(recent) < min_recent and total > len(recent):
            start = max(0, total - min_recent)
            recent = session.messages[start:]

        return [{"role": m.get("role"), "content": m.get("content")} for m in recent]

    async def _maybe_summarize(
        self,
        session: "Session",
        summary: str,
        summary_index: int,
    ) -> tuple[str, int, bool]:
        total = len(session.messages)
        if total == 0:
            return summary, summary_index, False

        recent_target = max(1, int(self.config.recent_messages))
        cutoff = max(summary_index, total - recent_target)
        if cutoff <= summary_index:
            return summary, summary_index, False

        # Estimate local conversation size (summary + unsummarized messages).
        to_summarize = session.messages[summary_index:cutoff]
        local_tokens = self._estimate_messages_tokens(
            [{"role": "assistant", "content": summary}] if summary else []
        ) + self._estimate_messages_tokens(
            [{"role": m.get("role"), "content": m.get("content")} for m in to_summarize]
        )
        ratio = local_tokens / self._effective_window()
        if ratio < self.config.summarize_threshold:
            return summary, summary_index, False

        new_summary = await self._summarize_messages(summary, to_summarize)
        if not new_summary:
            return summary, summary_index, False

        return new_summary, cutoff, True

    async def _summarize_messages(
        self, summary: str, messages: list[dict[str, Any]]
    ) -> str | None:
        if not messages:
            return summary

        formatted = self._format_messages(messages)
        summary_intro = summary.strip()

        system_prompt = (
            "You are a conversation summarizer. Produce a concise rolling summary that preserves: "
            "user goals, preferences, constraints, decisions, TODOs, and important facts/names. "
            "Omit small talk and repetitions. Use the same language as the conversation."
        )

        user_prompt = (
            "Existing summary (may be empty):\n"
            f"{summary_intro if summary_intro else '(none)'}\n\n"
            "New conversation excerpt to fold in:\n"
            f"{formatted}\n\n"
            "Return ONLY the updated summary text."
        )

        try:
            response = await self.provider.chat(
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ],
                tools=None,
                model=self.config.summary_model or self.default_model,
                max_tokens=self.config.summary_max_tokens,
                temperature=0.2,
                session_state=None,
            )
        except Exception as e:
            logger.warning(f"Failed to summarize context: {e}")
            return None

        if response and response.content:
            return response.content.strip()
        return None

    def _format_messages(self, messages: list[dict[str, Any]]) -> str:
        lines: list[str] = []
        for msg in messages:
            role = (msg.get("role") or "").strip().lower()
            content = msg.get("content")
            if isinstance(content, list):
                try:
                    content_text = json.dumps(content, ensure_ascii=False)
                except Exception:
                    content_text = str(content)
            else:
                content_text = str(content)
            if role == "user":
                label = "User"
            elif role == "assistant":
                label = "Assistant"
            else:
                label = role or "Message"
            lines.append(f"{label}: {content_text}")
        return "\n".join(lines)

    def _estimate_messages_tokens(self, messages: list[dict[str, Any]]) -> int:
        if not messages:
            return 0
        total_chars = 0
        for msg in messages:
            role = msg.get("role") or ""
            content = msg.get("content")
            total_chars += len(str(role))
            if isinstance(content, list):
                try:
                    total_chars += len(json.dumps(content, ensure_ascii=False))
                except Exception:
                    total_chars += len(str(content))
            else:
                total_chars += len(str(content))
        # Rough heuristic: 4 chars per token
        return max(1, total_chars // 4)

    def _effective_window(self) -> int:
        reserve = max(0, int(self.config.reserve_tokens))
        window = max(1, int(self.config.window_tokens))
        return max(1, window - reserve)
