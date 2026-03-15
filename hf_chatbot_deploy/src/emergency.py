"""
Safety escalation detector for the Pre-Marital Health Assistant.
Detects high-risk situations where the bot should stop normal flow and escalate.
"""

# These patterns are intentionally broad to catch high-anxiety or dangerous contexts.
ANXIETY_KEYWORDS = {
    "خايف": "قلق شديد",
    "خايفة": "قلق شديد",
    "قلقان": "قلق شديد",
    "قلقانة": "قلق شديد",
    "متوتر": "توتر شديد",
    "متوترة": "توتر شديد",
    "مرعوب": "قلق شديد",
    "panic": "قلق شديد",
    "anxious": "قلق شديد",
}

DIAGNOSIS_REQUEST_KEYWORDS = {
    "شخصني": "طلب تشخيص",
    "شخّصني": "طلب تشخيص",
    "هل عندي": "طلب تشخيص",
    "اكيد عندي": "طلب تشخيص",
    "عايز تشخيص": "طلب تشخيص",
    "diagnose": "طلب تشخيص",
    "diagnosis": "طلب تشخيص",
    "فسر نتيجة التحليل": "طلب تفسير سريري",
    "اقرالي التحليل": "طلب تفسير سريري",
}

ALARMING_RESULT_KEYWORDS = {
    "النتيجة خطيرة": "نتيجة مقلقة",
    "نتيجة التحليل وحشة": "نتيجة مقلقة",
    "high risk": "نتيجة مقلقة",
    "positive": "نتيجة مقلقة",
    "critical": "نتيجة مقلقة",
    "غير طبيعي": "نتيجة مقلقة",
    "abnormal": "نتيجة مقلقة",
    "خارج المعدل": "نتيجة مقلقة",
}

SELF_HARM_KEYWORDS = {
    "عايز اموت": "أفكار إيذاء النفس",
    "عايزة اموت": "أفكار إيذاء النفس",
    "مش عايز اعيش": "أفكار إيذاء النفس",
    "مش عايزة اعيش": "أفكار إيذاء النفس",
    "هأذي نفسي": "أفكار إيذاء النفس",
    "هاذي نفسي": "أفكار إيذاء النفس",
    "suicide": "أفكار إيذاء النفس",
    "kill myself": "أفكار إيذاء النفس",
}


def _find_matches(message: str, table: dict) -> list:
    matches = []
    for keyword, label in table.items():
        if keyword in message and label not in matches:
            matches.append(label)
    return matches


def check_emergency(message: str) -> dict:
    """Check whether a message needs safety escalation."""
    message_clean = (message or "").strip().lower()

    anxiety = _find_matches(message_clean, ANXIETY_KEYWORDS)
    diagnosis = _find_matches(message_clean, DIAGNOSIS_REQUEST_KEYWORDS)
    alarming = _find_matches(message_clean, ALARMING_RESULT_KEYWORDS)
    self_harm = _find_matches(message_clean, SELF_HARM_KEYWORDS)

    detected_symptoms = self_harm + diagnosis + alarming + anxiety
    if not detected_symptoms:
        return {
            "is_emergency": False,
            "detected_symptoms": [],
            "category": None,
            "response": None,
        }

    if self_harm:
        response = (
            "🚨 أنا سامعاك ومقدرة قد إيه الموقف صعب عليك الآن.\n\n"
            "سلامتك هي الأولوية. أنا أداة تعليمية ومش بديل للرعاية الطارئة.\n"
            "لو في خطر مباشر عليك الآن، اتصل بالطوارئ فورًا أو اطلب المساعدة من شخص قريب فورًا.\n\n"
            "مهم جدًا تتواصل مع مختص نفسي أو طبي في أسرع وقت."
        )
        category = "self_harm"
    else:
        response = (
            "أنا فاهمة قلقك، وده شعور طبيعي قبل الزواج خاصة مع الفحوصات.\n\n"
            "للتوضيح: أنا أقدر أقدم توعية عامة، لكن ما أقدرش أشخص أو أفسر نتيجة تحليل بشكل سريري.\n"
            "الأفضل الآن إنك تراجع/ي طبيب مختص علشان تقييم دقيق وآمن حسب حالتك.\n\n"
            "لو تحب/ي، أقدر أساعدك تجهز/ي قائمة أسئلة تاخدها معاك للطبيب."
        )
        category = "medical_escalation"

    return {
        "is_emergency": True,
        "detected_symptoms": detected_symptoms,
        "category": category,
        "response": response,
    }


def get_emergency_response() -> str:
    """Get a generic escalation response."""
    return (
        "أنا هنا للدعم التوعوي العام، لكن ما أقدرش أشخص أو أقرر طبيًا. "
        "لو في أعراض مقلقة أو نتيجة تحليل غير مطمئنة، الأفضل مراجعة طبيب مختص في أقرب وقت."
    )
