"""Slack画像ファイルの取得・処理"""
import base64
import io
import logging
from typing import Optional

import requests

logger = logging.getLogger(__name__)

SUPPORTED_MIMETYPES = {"image/jpeg", "image/png", "image/gif", "image/webp"}
MAX_IMAGES = 5
MAX_DIMENSION = 1568  # Bedrock Claude推奨上限

FORMAT_MAP = {
    "image/jpeg": "JPEG",
    "image/png": "PNG",
    "image/gif": "GIF",
    "image/webp": "WEBP",
}


def extract_image_files(event: dict) -> list:
    """Slackイベントからサポート対象の画像ファイル情報を抽出"""
    files = event.get("files", [])
    images = []
    for f in files:
        mimetype = f.get("mimetype", "")
        if mimetype in SUPPORTED_MIMETYPES:
            images.append({
                "url_private": f["url_private"],
                "mimetype": mimetype,
                "name": f.get("name", "image"),
            })
    return images[:MAX_IMAGES]


def download_image_as_base64(url_private: str, bot_token: str, media_type: str = "image/png") -> Optional[str]:
    """Slack url_privateから画像をダウンロードしbase64エンコード"""
    try:
        resp = requests.get(
            url_private,
            headers={"Authorization": f"Bearer {bot_token}"},
            timeout=15,
        )
        resp.raise_for_status()
        image_bytes = _resize_if_needed(resp.content, media_type)
        return base64.b64encode(image_bytes).decode("utf-8")
    except Exception as e:
        logger.error(f"Failed to download image {url_private}: {e}")
        return None


def _resize_if_needed(image_bytes: bytes, media_type: str) -> bytes:
    """画像が大きすぎる場合はリサイズ"""
    try:
        from PIL import Image

        img = Image.open(io.BytesIO(image_bytes))
        w, h = img.size
        if max(w, h) <= MAX_DIMENSION:
            return image_bytes

        ratio = MAX_DIMENSION / max(w, h)
        new_size = (int(w * ratio), int(h * ratio))
        img = img.resize(new_size, Image.LANCZOS)

        buf = io.BytesIO()
        fmt = FORMAT_MAP.get(media_type, "PNG")
        img.save(buf, format=fmt)
        logger.info(f"Resized image from {w}x{h} to {new_size[0]}x{new_size[1]}")
        return buf.getvalue()
    except ImportError:
        logger.warning("Pillow not installed, skipping resize")
        return image_bytes


def fetch_images_from_event(event: dict, bot_token: str) -> list:
    """イベントから画像を取得しbase64データ付きリストを返す

    Returns:
        list[dict]: [{"base64_data": str, "media_type": str, "name": str}, ...]
    """
    image_files = extract_image_files(event)
    if not image_files:
        return []

    results = []
    for img_info in image_files:
        b64 = download_image_as_base64(
            img_info["url_private"],
            bot_token,
            img_info["mimetype"],
        )
        if b64:
            results.append({
                "base64_data": b64,
                "media_type": img_info["mimetype"],
                "name": img_info["name"],
            })

    logger.info(f"Fetched {len(results)}/{len(image_files)} images from event")
    return results
