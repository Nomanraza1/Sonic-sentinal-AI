from src.rules import decide

def test_critical_requires_repeat_and_agreement():
    scores={'gunshot':.9,'background_noise':.1}
    result=decide({'available':True,'class':'gunshot','scores':scores,'confidence':.9},{'available':True,'class':'gunshot','scores':scores,'confidence':.88},'Good',False)
    assert result['severity']=='Critical'
    assert result['manual_review'] is False
