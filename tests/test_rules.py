from src.rules import decide


def test_critical_requires_repeat_and_agreement():
    scores = {"gunshot": 0.9, "background_noise": 0.1}
    result = decide(
        {"available": True, "class": "gunshot", "scores": scores, "confidence": 0.9},
        {"available": True, "class": "gunshot", "scores": scores, "confidence": 0.88},
        "Good",
        False,
    )
    assert result["severity"] == "Critical"
    assert result["manual_review"] is False


# VERIFIED
