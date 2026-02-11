"""JWT authentication using pure stdlib (hmac + json + base64)."""

import base64
import hashlib
import hmac
import json
import time
import uuid


class AuthManager:
    """Manage JWT-based authentication with invite code login."""

    def __init__(self, secret: str, expiry_days: int = 30):
        self._secret = secret.encode("utf-8")
        self._expiry_seconds = expiry_days * 86400

    def login(self, invite_code: str) -> str | None:
        """Validate invite code and return a JWT, or None if invalid."""
        if not hmac.compare_digest(invite_code.encode("utf-8"), self._secret):
            return None
        now = int(time.time())
        chat_id = self._make_chat_id(invite_code)
        payload = {
            "sub": "web_user",
            "chat_id": chat_id,
            "iat": now,
            "exp": now + self._expiry_seconds,
            "jti": uuid.uuid4().hex[:8],
        }
        return self._encode(payload)

    def validate(self, token: str) -> dict | None:
        """Validate a JWT and return its payload, or None if invalid/expired."""
        payload = self._decode(token)
        if payload is None:
            return None
        exp = payload.get("exp", 0)
        if int(time.time()) > exp:
            return None
        return payload

    def _make_chat_id(self, invite_code: str) -> str:
        """Generate a stable chat_id from the invite code."""
        h = hashlib.sha256(invite_code.encode("utf-8")).hexdigest()[:12]
        return f"web_{h}"

    def _encode(self, payload: dict) -> str:
        """Encode payload into a JWT string."""
        header = {"alg": "HS256", "typ": "JWT"}
        h = self._b64url_encode(json.dumps(header, separators=(",", ":")).encode())
        p = self._b64url_encode(json.dumps(payload, separators=(",", ":")).encode())
        signing_input = f"{h}.{p}"
        sig = hmac.new(self._secret, signing_input.encode("utf-8"), hashlib.sha256).digest()
        s = self._b64url_encode(sig)
        return f"{h}.{p}.{s}"

    def _decode(self, token: str) -> dict | None:
        """Decode and verify a JWT. Returns payload dict or None."""
        parts = token.split(".")
        if len(parts) != 3:
            return None
        h, p, s = parts
        signing_input = f"{h}.{p}"
        expected_sig = hmac.new(
            self._secret, signing_input.encode("utf-8"), hashlib.sha256
        ).digest()
        try:
            actual_sig = self._b64url_decode(s)
        except Exception:
            return None
        if not hmac.compare_digest(expected_sig, actual_sig):
            return None
        try:
            payload = json.loads(self._b64url_decode(p))
        except Exception:
            return None
        return payload

    @staticmethod
    def _b64url_encode(data: bytes) -> str:
        return base64.urlsafe_b64encode(data).rstrip(b"=").decode("ascii")

    @staticmethod
    def _b64url_decode(s: str) -> bytes:
        padding = 4 - len(s) % 4
        if padding != 4:
            s += "=" * padding
        return base64.urlsafe_b64decode(s)
