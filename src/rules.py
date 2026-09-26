import json
from collections import deque
from pathlib import Path
from config.settings import ROOT

recent = deque(maxlen=20)
def decide(py, gtm, quality, overlap):
    rules = json.loads((ROOT / 'alert_rules' / 'default.json').read_text()); default = rules['defaults']
    label = py['class'] if py['available'] else (gtm['class'] if gtm['available'] else 'unknown')
    category = rules['categories'].get(label, {'severity':'Informational','action':'Route for review.','critical':False})
    top_two = sorted(py.get('scores', {}).values(), reverse=True)[:2]; margin = top_two[0]-top_two[1] if len(top_two)==2 else 0
    agree = py['available'] and gtm['available'] and py['class'] == gtm['class']
    confidence = py.get('confidence', 0); repeat = sum(x == label for x in recent) + 1; recent.append(label)
    review = (not py['available'] or not gtm['available'] or not agree or confidence < default['minimum_confidence'] or margin < default['top_two_margin'] or quality in ('Poor','Unusable') or overlap)
    alert = category['critical'] and confidence >= default['minimum_confidence'] and quality in default['required_quality'] and (not default['require_model_agreement_for_critical'] or agree) and repeat >= default['repeat_windows']
    status = 'Alert Generated' if alert else ('Manual Review' if review else 'Classified')
    match = 'Acceptable Match' if agree and not review else ('Model Disagreement' if py['available'] and gtm['available'] and not agree else 'Uncertain Result')
    return {'final_class': label, 'severity': category['severity'], 'recommended_action': category['action'], 'manual_review': review, 'alert_status': status, 'agreement_status': match, 'confidence_difference': abs(py.get('confidence',0)-gtm.get('confidence',0)) if gtm['available'] else None, 'top_two_margin': margin, 'repeat_count': repeat}
# verified