"""Admin data store for Rafiqa Pre-Marital Health Assistant."""
import json
import os
import threading
from datetime import datetime, timezone

# Use a persistent path — on HF Spaces, /data is persistent storage
# Locally, falls back to a local file
_DATA_DIR = os.environ.get("RAFIQA_DATA_DIR", os.path.join(os.path.dirname(__file__), "..", "admin_data"))
_STORE_PATH = os.path.join(_DATA_DIR, "admin_store.json")
_lock = threading.Lock()

# Maximum items to keep in memory/file
MAX_CONVERSATIONS = 200
MAX_ALERTS = 100
MAX_ACTIVITIES = 50


def _ensure_dir():
    os.makedirs(_DATA_DIR, exist_ok=True)


def _default_store():
    return {
        "stats": {
            "totalConversations": 0,
            "totalMessages": 0,
            "emergencyAlerts": 0,
        },
        "conversations": [],
        "alerts": [],
        "activities": [],
        "uploadedFiles": [],
    }


def _load():
    """Load the store from disk."""
    _ensure_dir()
    if os.path.exists(_STORE_PATH):
        try:
            with open(_STORE_PATH, "r", encoding="utf-8") as f:
                return json.load(f)
        except (json.JSONDecodeError, IOError):
            return _default_store()
    return _default_store()


def _save(store):
    """Save the store to disk."""
    _ensure_dir()
    with open(_STORE_PATH, "w", encoding="utf-8") as f:
        json.dump(store, f, ensure_ascii=False, indent=2)


def _now_str():
    return datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M")


# ──────────────────────────────────────────────
# Public API
# ──────────────────────────────────────────────

def log_conversation(session_id: str, user_name: str, messages: list,
                     is_emergency: bool = False, emergency_symptom: str = ""):
    """
    Log a complete conversation turn.
    """
    with _lock:
        store = _load()

        # Update stats
        store["stats"]["totalMessages"] += len(messages)

        # Find existing conversation or create new
        existing = next((c for c in store["conversations"] if c["sessionId"] == session_id), None)

        if existing:
            existing["messages"] = messages
            existing["messagesCount"] = len(messages)
            existing["lastMessage"] = messages[-1]["content"] if messages else ""
            existing["date"] = _now_str()
            if is_emergency:
                existing["status"] = "emergency"
        else:
            store["stats"]["totalConversations"] += 1
            conv_num = store["stats"]["totalConversations"]
            conv = {
                "id": conv_num,
                "sessionId": session_id,
                "userName": user_name or f"مستخدم #{conv_num}",
                "status": "emergency" if is_emergency else "normal",
                "lastMessage": messages[-1]["content"] if messages else "",
                "date": _now_str(),
                "messagesCount": len(messages),
                "duration": "",
                "messages": messages,
            }
            store["conversations"].insert(0, conv)
            store["conversations"] = store["conversations"][:MAX_CONVERSATIONS]

            # Activity
            if is_emergency:
                activity_text = f"مستخدم أبلغ عن {emergency_symptom} — تم التوجيه لمختص"
                activity_type = "emergency"
                activity_icon = "fas fa-exclamation-triangle"
            else:
                short_msg = (messages[-1]["content"][:50] + "...") if messages and len(messages[-1]["content"]) > 50 else (messages[-1]["content"] if messages else "")
                activity_text = f"محادثة جديدة — {short_msg}"
                activity_type = "conversation"
                activity_icon = "fas fa-comment"

            store["activities"].insert(0, {
                "id": len(store["activities"]) + 1,
                "type": activity_type,
                "icon": activity_icon,
                "text": activity_text,
                "time": _now_str(),
            })
            store["activities"] = store["activities"][:MAX_ACTIVITIES]

        _save(store)


def log_emergency(user_name: str, symptom: str, user_message: str, session_id: str = ""):
    """Log an emergency alert."""
    with _lock:
        store = _load()
        store["stats"]["emergencyAlerts"] += 1

        alert = {
            "id": len(store["alerts"]) + 1,
            "symptom": symptom,
            "severity": "critical",
            "userName": user_name,
            "userMessage": user_message,
            "time": _now_str(),
            "handled": False,
            "sessionId": session_id,
        }
        store["alerts"].insert(0, alert)
        store["alerts"] = store["alerts"][:MAX_ALERTS]

        # Activity
        store["activities"].insert(0, {
            "id": len(store["activities"]) + 1,
            "type": "emergency",
            "icon": "fas fa-exclamation-triangle",
            "text": f"تنبيه طوارئ: {symptom} — {user_name}",
            "time": _now_str(),
        })
        store["activities"] = store["activities"][:MAX_ACTIVITIES]

        _save(store)


def log_upload(file_name: str, file_size: str, chunks: int, pages: int):
    """Log a successful file upload."""
    with _lock:
        store = _load()
        upload = {
            "name": file_name,
            "size": file_size,
            "chunks": chunks,
            "pages": pages,
            "date": _now_str(),
        }
        store["uploadedFiles"].insert(0, upload)

        # Activity
        store["activities"].insert(0, {
            "id": len(store["activities"]) + 1,
            "type": "success",
            "icon": "fas fa-check",
            "text": f"تم رفع ملف جديد: {file_name}",
            "time": _now_str(),
        })
        store["activities"] = store["activities"][:MAX_ACTIVITIES]

        _save(store)


def get_stats():
    """Return stats dict."""
    with _lock:
        store = _load()
        return store["stats"]


def get_conversations(limit: int = 50):
    """Return recent conversations."""
    with _lock:
        store = _load()
        convs = store["conversations"][:limit]
        summaries = []
        for c in convs:
            summaries.append({
                "id": c["id"],
                "sessionId": c["sessionId"],
                "userName": c["userName"],
                "status": c["status"],
                "lastMessage": c["lastMessage"],
                "date": c["date"],
                "messagesCount": c["messagesCount"],
                "duration": c.get("duration", ""),
                "messages": c.get("messages", []),
            })
        return summaries


def get_alerts(limit: int = 50):
    """Return recent emergency alerts."""
    with _lock:
        store = _load()
        return store["alerts"][:limit]


def get_activities(limit: int = 20):
    """Return recent activities."""
    with _lock:
        store = _load()
        return store["activities"][:limit]


def get_uploaded_files():
    """Return uploaded files list."""
    with _lock:
        store = _load()
        return store["uploadedFiles"]


def get_full_admin_data(admin_key: str, expected_key: str):
    """
    Return all admin data in one call (for the admin panel).
    Requires admin key for authentication.
    """
    try:
        if not admin_key or admin_key.strip() != expected_key:
            return json.dumps({"error": "مفتاح الأدمن غير صحيح"}, ensure_ascii=False)

        with _lock:
            store = _load()
            return json.dumps({
                "stats": store.get("stats", {"totalConversations": 0, "totalMessages": 0, "emergencyAlerts": 0}),
                "conversations": store.get("conversations", [])[:50],
                "alerts": store.get("alerts", [])[:50],
                "activities": store.get("activities", [])[:20],
                "uploadedFiles": store.get("uploadedFiles", []),
            }, ensure_ascii=False)
    except Exception as e:
        return json.dumps({"error": f"خطأ في السيرفر: {str(e)}"}, ensure_ascii=False)
