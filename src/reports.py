from html import escape
from pathlib import Path


def write_report(item, folder):
    folder = Path(folder)
    folder.mkdir(parents=True, exist_ok=True)
    path = folder / f"analysis_{item['detection_id']}.html"
    rows = "".join(
        f'<tr><th>{escape(str(k).replace("_", " ").title())}</th><td>{escape(str(v))}</td></tr>'
        for k, v in item.items()
        if k not in {"python_scores", "gtm_scores", "quality_details", "stored_path"}
    )
    scores = (
        "<h2>Python confidence scores</h2><pre>"
        + escape(item.get("python_scores", ""))
        + "</pre><h2>GTM confidence scores</h2><pre>"
        + escape(item.get("gtm_scores", ""))
        + "</pre>"
    )
    path.write_text(
        f'<!doctype html><html><head><meta charset="utf-8"><title>SonicSentinel report</title><style>body{{font:16px Arial;margin:40px;color:#18332d}}table{{border-collapse:collapse;width:100%}}th,td{{padding:9px;border:1px solid #ddd;text-align:left}}th{{background:#eef8f2;width:32%}}pre{{white-space:pre-wrap}}</style></head><body><h1>SonicSentinel AI analysis report</h1><table>{rows}</table>{scores}</body></html>',
        encoding="utf-8",
    )
    return path


# verified
