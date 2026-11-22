"""Проверка PNG и пропорциональное уменьшение копии для API без изменения оригинала."""

import base64
import hashlib
from io import BytesIO
from pathlib import Path

from PIL import Image


def prepare_image(source: Path, max_side: int | None) -> tuple[str, dict]:
    payload = source.read_bytes()
    info: dict = {
        "name": source.name,
        "sha256": hashlib.sha256(payload).hexdigest(),
        "original_bytes": len(payload),
    }
    with Image.open(BytesIO(payload)) as image:
        if image.format != "PNG":
            raise ValueError("Ожидается PNG")
        if getattr(image, "n_frames", 1) != 1:
            raise ValueError("Ожидается статический PNG с одним кадром")
        image.verify()
    with Image.open(BytesIO(payload)) as image:
        image.load()
        info["original_size"] = list(image.size)
        if max_side is not None and max(image.size) > max_side:
            image.thumbnail((max_side, max_side), Image.Resampling.LANCZOS)
            buffer = BytesIO()
            image.save(buffer, format="PNG")
            payload = buffer.getvalue()
        info.update(sent_size=list(image.size), mode=image.mode)
    info.update(
        sent_bytes=len(payload),
        sent_sha256=hashlib.sha256(payload).hexdigest(),
        resized=info["original_size"] != info["sent_size"],
    )
    return "data:image/png;base64," + base64.b64encode(payload).decode("ascii"), info
