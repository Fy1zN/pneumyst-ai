"""Analysis images on local disk, served through short-lived signed links.

Browsers load images with plain <img> tags, which cannot send the Clerk token, so a link carries its own proof:
an HMAC of the file key and an expiry time, made with a server-only secret.
"""
from __future__ import annotations

import hashlib
import hmac
import os
import re
import shutil
import time
import uuid
from collections.abc import Callable
from pathlib import Path
from urllib.parse import urlencode

KEY_PATTERN = re.compile(r"analyses/[0-9a-f-]{36}/(image|heatmaps/[A-Za-z_]+)\.png")


class LocalStorage:
    def __init__(self, root: Path, signing_key: str, ttl_s: int = 300, clock: Callable[[], float] = time.time):
        self.root = Path(root).resolve()
        self.signing_key = signing_key.encode()
        self.ttl_s = ttl_s
        self.clock = clock

    def path(self, key: str) -> Path:
        if not KEY_PATTERN.fullmatch(key):
            raise ValueError(f"invalid storage key: {key!r}")
        return self.root.joinpath(*key.split("/"))

    def put(self, key: str, data: bytes) -> None:
        path = self.path(key)
        path.parent.mkdir(parents=True, exist_ok=True)
        # Write then rename, so a crash never leaves a half-written image behind a valid key.
        tmp = path.with_suffix(".tmp")
        tmp.write_bytes(data)
        os.replace(tmp, path)

    def delete_analysis(self, analysis_id: str) -> None:
        folder = self.root / "analyses" / str(uuid.UUID(analysis_id))
        if folder.exists():
            shutil.rmtree(folder)

    def _signature(self, key: str, expires: int) -> str:
        return hmac.new(self.signing_key, f"{key}\n{expires}".encode(), hashlib.sha256).hexdigest()

    def url(self, key: str) -> str:
        self.path(key)
        expires = int(self.clock()) + self.ttl_s
        return f"/files/{key}?" + urlencode({"expires": expires, "sig": self._signature(key, expires)})

    def verify(self, key: str, expires: int, sig: str) -> bool:
        if not KEY_PATTERN.fullmatch(key) or expires < self.clock():
            return False
        return hmac.compare_digest(self._signature(key, expires), sig)