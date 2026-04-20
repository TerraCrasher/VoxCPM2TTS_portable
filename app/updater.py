"""업데이트 체크 모듈"""

import json
import urllib.request
from app.config import VERSION, GITHUB_REPO, GITHUB_URL


def check_update() -> dict:
    """
    GitHub 최신 릴리즈 확인
    Returns: {"available": bool, "latest": str, "current": str, "url": str, "message": str}
    """
    result = {
        "available": False,
        "latest": VERSION,
        "current": VERSION,
        "url": GITHUB_URL,
        "message": "",
    }

    try:
        api_url = f"https://api.github.com/repos/{GITHUB_REPO}/releases/latest"
        req = urllib.request.Request(api_url, headers={"User-Agent": "VoxCPM2TTS"})
        with urllib.request.urlopen(req, timeout=5) as resp:
            data = json.loads(resp.read().decode())

        latest = data.get("tag_name", "").lstrip("v")
        result["latest"] = latest

        if latest and latest != VERSION:
            result["available"] = True
            result["message"] = (
                f"🔔 새 버전 발견! v{VERSION} → v{latest}\n"
                f"   {GITHUB_URL}/releases/latest"
            )
        else:
            result["message"] = f"✅ 최신 버전입니다. (v{VERSION})"

    except Exception as e:
        result["message"] = f"⚠️ 업데이트 확인 실패 (오프라인?)\n   현재 버전: v{VERSION}"

    return result