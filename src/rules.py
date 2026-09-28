import json
import time
from collections import OrderedDict
from threading import Lock
from config.settings import RULES_PATH

recent = OrderedDict()
recent_lock = Lock()


def decide(py, gtm, quality, overlap, stream_id="default", *, prefer_gtm=False):
    rules = json.loads(RULES_PATH.read_text(encoding="utf-8"))
    default = rules["defaults"]
    # Live microphone windows use GTM as the primary event model when it is
    # available. The Python model remains stored and visible for comparison.
    primary = gtm if prefer_gtm and gtm["available"] else (
        py if py["available"] else (gtm if gtm["available"] else None)
    )
    label = primary["class"] if primary else "unknown"
    category = rules["categories"].get(label, {"severity": "Informational", "action": "Route for review.", "critical": False})
    top_two = sorted(primary.get("scores", {}).values(), reverse=True)[:2] if primary else []
    margin = top_two[0] - top_two[1] if len(top_two) == 2 else 0
    agree = py["available"] and gtm["available"] and py["class"] == gtm["class"]
    confidence = primary.get("confidence", 0) if primary else 0
    review = (not py["available"] or not gtm["available"] or not agree
              or confidence < default["minimum_confidence"]
              or margin < default["top_two_margin"]
              or quality not in default["required_quality"] or overlap)
    # Only consecutive eligible windows count; an uncertain window resets confirmation.
    eligible = (primary is not None and confidence >= default["minimum_confidence"]
                and margin >= default["top_two_margin"] and not overlap
                and quality in default["required_quality"]
                and (not default["require_model_agreement_for_critical"] or agree))
    clock = time.monotonic()
    with recent_lock:
        previous, count, updated = recent.get(stream_id, (None, 0, 0))
        repeat = (count + 1 if previous == label and clock - updated <= default.get("confirmation_seconds", 15) else 1) if eligible else 0
        recent[stream_id] = (label, repeat, clock)
        recent.move_to_end(stream_id)
        while len(recent) > 1000:
            recent.popitem(last=False)
    alert = category["critical"] and eligible and repeat >= default["repeat_windows"]
    status = "Alert Generated" if alert else ("Manual Review" if review else "Classified")
    match = ("Acceptable Match" if agree and not review else "Weak Match" if agree
             else "Model Disagreement" if py["available"] and gtm["available"] else "Uncertain Result")
    return {"final_class": label, "severity": category["severity"],
            "recommended_action": category["action"], "manual_review": review,
            "alert_status": status, "agreement_status": match,
            "confidence_difference": abs(py.get("confidence", 0) - gtm.get("confidence", 0)) if py["available"] and gtm["available"] else None,
            "top_two_margin": margin, "repeat_count": repeat}
