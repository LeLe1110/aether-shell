"""Session management for conversation history."""

import json
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any

from loguru import logger

from nanobot.utils.helpers import ensure_dir, get_sessions_path, safe_filename


@dataclass
class Session:
    """
    A conversation session.

    Stores messages in JSONL format for easy reading and persistence.
    """

    key: str  # channel:chat_id
    messages: list[dict[str, Any]] = field(default_factory=list)
    created_at: datetime = field(default_factory=datetime.now)
    updated_at: datetime = field(default_factory=datetime.now)
    metadata: dict[str, Any] = field(default_factory=dict)

    def add_message(self, role: str, content: str, **kwargs: Any) -> None:
        """Add a message to the session."""
        msg = {
            "role": role,
            "content": content,
            "timestamp": datetime.now().isoformat(),
            **kwargs
        }
        self.messages.append(msg)
        self.updated_at = datetime.now()

    def get_history(self, max_messages: int = 50) -> list[dict[str, Any]]:
        """
        Get message history for LLM context.

        Args:
            max_messages: Maximum messages to return.

        Returns:
            List of messages in LLM format.
        """
        # Get recent messages
        recent = self.messages[-max_messages:] if len(self.messages) > max_messages else self.messages

        # Convert to LLM format (just role and content)
        return [{"role": m["role"], "content": m["content"]} for m in recent]

    def clear(self) -> None:
        """Clear all messages in the session."""
        self.messages = []
        self.updated_at = datetime.now()


class SessionManager:
    """
    Manages conversation sessions.

    Sessions are stored as JSONL files in the sessions directory.
    """

    def __init__(self, workspace: Path):
        self.workspace = workspace
        self.sessions_dir = ensure_dir(Path.home() / ".aether-bot" / "sessions")
        self._active_index_path = get_sessions_path() / "active.json"
        self._active_sessions: dict[str, str] = {}
        self._load_active_index()
        self._cache: dict[str, Session] = {}

    def _load_active_index(self) -> None:
        if not self._active_index_path.exists():
            return
        try:
            with open(self._active_index_path) as f:
                data = json.load(f)
            if isinstance(data, dict):
                old = dict(self._active_sessions)
                self._active_sessions = {str(k): str(v) for k, v in data.items()}
                # Only log actual runtime changes (skip initial load from None)
                for k in set(list(old.keys()) + list(self._active_sessions.keys())):
                    ov = old.get(k)
                    nv = self._active_sessions.get(k)
                    if ov != nv and ov is not None:
                        logger.debug(f"active_index changed: {k}: {ov} -> {nv}")
        except Exception as e:
            logger.warning(f"Failed to load session index: {e}")

    def _save_active_index(self) -> None:
        try:
            with open(self._active_index_path, "w") as f:
                json.dump(self._active_sessions, f, indent=2)
        except Exception as e:
            logger.warning(f"Failed to save session index: {e}")

    def _make_session_key(self, base_key: str) -> str:
        stamp = datetime.now().strftime("%Y%m%d%H%M%S")
        return f"{base_key}#{stamp}"

    def _get_active_key(self, base_key: str) -> str | None:
        return self._active_sessions.get(base_key)

    def _set_active_key(self, base_key: str, session_key: str) -> None:
        old = self._active_sessions.get(base_key)
        self._active_sessions[base_key] = session_key
        if old != session_key:
            logger.debug(f"_set_active_key: {base_key}: {old} -> {session_key}")
        self._save_active_index()

    def _get_session_path(self, key: str) -> Path:
        """Get the file path for a session."""
        safe_key = safe_filename(key.replace(":", "_"))
        return self.sessions_dir / f"{safe_key}.jsonl"

    def get_or_create(self, key: str) -> Session:
        """
        Get an existing session or create a new one.

        Reloads active.json each time to pick up external changes
        (e.g. web channel switching sessions).

        If the key contains '#' it is treated as a direct session key
        (e.g. "web:chat_id:default#20260208000954") and resolved without
        relying on the active index.

        Args:
            key: Session key (usually channel:chat_id or direct key with #).

        Returns:
            The session.
        """
        # Direct session key (contains #timestamp) — resolve without active index
        if "#" in key:
            if key in self._cache:
                logger.debug(f"get_or_create direct hit cache: {key}")
                return self._cache[key]
            session = self._load(key)
            if session is not None:
                self._cache[key] = session
                logger.debug(f"get_or_create direct loaded: {key} msgs={len(session.messages)}")
                return session
            # File doesn't exist yet — create it
            session = Session(key=key)
            self._cache[key] = session
            logger.debug(f"get_or_create direct created: {key}")
            return session

        # Base key — reload active index to pick up external changes
        self._load_active_index()

        # Check cache for active session
        active_key = self._get_active_key(key)
        logger.debug(
            f"get_or_create key={key} active_key={active_key} "
            f"cached={active_key in self._cache if active_key else 'n/a'}"
        )
        if active_key and active_key in self._cache:
            return self._cache[active_key]

        # Active key changed or not cached — load from disk
        if active_key:
            session = self._load(active_key)
            if session is None:
                session = Session(key=active_key)
            self._cache[active_key] = session
            logger.debug(f"get_or_create loaded from disk: {active_key} msgs={len(session.messages)}")
            return session

        # Backward compatibility: if base session file exists, use it
        legacy = self._load(key)
        if legacy is not None:
            self._set_active_key(key, key)
            self._cache[key] = legacy
            return legacy

        # Create a new session for this base key
        new_key = self._make_session_key(key)
        session = Session(key=new_key)
        self._set_active_key(key, new_key)
        self._cache[new_key] = session
        return session

    def start_new(self, base_key: str) -> Session:
        """Create and activate a new session for a base key."""
        new_key = self._make_session_key(base_key)
        session = Session(key=new_key)
        self._set_active_key(base_key, new_key)
        self._cache[new_key] = session
        return session

    def _load(self, key: str) -> Session | None:
        """Load a session from disk."""
        path = self._get_session_path(key)

        if not path.exists():
            return None

        try:
            messages = []
            metadata = {}
            created_at = None

            with open(path) as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue

                    data = json.loads(line)

                    if data.get("_type") == "metadata":
                        metadata = data.get("metadata", {})
                        created_at = datetime.fromisoformat(data["created_at"]) if data.get("created_at") else None
                    else:
                        messages.append(data)

            return Session(
                key=key,
                messages=messages,
                created_at=created_at or datetime.now(),
                metadata=metadata
            )
        except Exception as e:
            logger.warning(f"Failed to load session {key}: {e}")
            return None

    def save(self, session: Session) -> None:
        """Save a session to disk."""
        path = self._get_session_path(session.key)

        with open(path, "w") as f:
            # Write metadata first
            metadata_line = {
                "_type": "metadata",
                "created_at": session.created_at.isoformat(),
                "updated_at": session.updated_at.isoformat(),
                "metadata": session.metadata
            }
            f.write(json.dumps(metadata_line) + "\n")

            # Write messages
            for msg in session.messages:
                f.write(json.dumps(msg) + "\n")

        self._cache[session.key] = session

    def delete(self, key: str) -> bool:
        """
        Delete a session.

        Args:
            key: Session key.

        Returns:
            True if deleted, False if not found.
        """
        # Remove from cache
        self._cache.pop(key, None)
        # Remove active pointer if needed
        base_to_remove = None
        for base_key, active_key in self._active_sessions.items():
            if active_key == key:
                base_to_remove = base_key
                break
        if base_to_remove:
            self._active_sessions.pop(base_to_remove, None)
            self._save_active_index()

        # Remove file
        path = self._get_session_path(key)
        if path.exists():
            path.unlink()
            return True
        return False

    def list_sessions(self) -> list[dict[str, Any]]:
        """
        List all sessions.

        Returns:
            List of session info dicts.
        """
        sessions = []

        for path in self.sessions_dir.glob("*.jsonl"):
            try:
                # Read just the metadata line
                with open(path) as f:
                    first_line = f.readline().strip()
                    if first_line:
                        data = json.loads(first_line)
                        if data.get("_type") == "metadata":
                            sessions.append({
                                "key": path.stem.replace("_", ":"),
                                "created_at": data.get("created_at"),
                                "updated_at": data.get("updated_at"),
                                "path": str(path)
                            })
            except Exception:
                continue

        return sorted(sessions, key=lambda x: x.get("updated_at", ""), reverse=True)
