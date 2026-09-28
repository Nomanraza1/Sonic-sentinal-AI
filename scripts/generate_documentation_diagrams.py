"""Generate the system and GTM flow figures used in project documentation."""
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

OUT = Path(__file__).resolve().parents[1] / "documentation" / "assets"
OUT.mkdir(parents=True, exist_ok=True)


def diagram(filename, title, nodes, edges, note):
    fig, ax = plt.subplots(figsize=(12, 6.8), dpi=180)
    fig.patch.set_facecolor("#f8fafc")
    ax.set_facecolor("#f8fafc")
    ax.set_xlim(0, 12)
    ax.set_ylim(0, 7)
    ax.axis("off")
    ax.text(0.55, 6.48, title, fontsize=19, weight="bold", color="#152238")
    for x, y, w, h, label, fill in nodes.values():
        ax.add_patch(FancyBboxPatch(
            (x, y), w, h, boxstyle="round,pad=0.08,rounding_size=0.14",
            facecolor=fill, edgecolor="#8291a6", linewidth=1.1,
        ))
        ax.text(x + w / 2, y + h / 2, label, ha="center", va="center",
                fontsize=10.3, color="#152238", linespacing=1.35)
    for source, target, label in edges:
        x1, y1, w1, h1, *_ = nodes[source]
        x2, y2, w2, h2, *_ = nodes[target]
        if abs(x2 - x1) >= abs(y2 - y1):
            start = (x1 + w1, y1 + h1 / 2) if x2 >= x1 else (x1, y1 + h1 / 2)
            end = (x2, y2 + h2 / 2) if x2 >= x1 else (x2 + w2, y2 + h2 / 2)
        else:
            start = (x1 + w1 / 2, y1) if y2 < y1 else (x1 + w1 / 2, y1 + h1)
            end = (x2 + w2 / 2, y2 + h2) if y2 < y1 else (x2 + w2 / 2, y2)
        ax.add_patch(FancyArrowPatch(start, end, arrowstyle="-|>", mutation_scale=12,
                                     linewidth=1.25, color="#52657c"))
        if label:
            ax.text((start[0] + end[0]) / 2, (start[1] + end[1]) / 2 + 0.12,
                    label, ha="center", fontsize=8.5, color="#42546a")
    ax.text(0.55, 0.32, note, fontsize=8.7, color="#52657a")
    fig.savefig(OUT / filename, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close(fig)


system_nodes = {
    "audio": (0.5, 3.2, 2.1, 1.2, "Audio upload or\npermission-based mic", "#dbeafe"),
    "prep": (3.2, 3.2, 2.2, 1.2, "Validate, decode, check\nquality and segment", "#e0f2fe"),
    "py": (6.1, 4.7, 2.1, 1.2, "Python model\nYAMNet plus SVM", "#dcfce7"),
    "gtm": (6.1, 1.8, 2.1, 1.2, "GTM model\nlog-mel CNN", "#fef3c7"),
    "rules": (9.0, 3.2, 2.1, 1.2, "Compare scores, apply\nalert rules and review", "#f3e8ff"),
    "store": (9.0, 0.65, 2.1, 1.2, "Save event, scores,\nquality and audit trail", "#fee2e2"),
    "review": (9.0, 5.35, 2.1, 1.0, "History, report and\nhuman review", "#e2e8f0"),
}
diagram("system-flow.png", "How SonicSentinel processes audio", system_nodes,
        [("audio", "prep", ""), ("prep", "py", "features"),
         ("prep", "gtm", "audio features"), ("py", "rules", "scores"),
         ("gtm", "rules", "scores"), ("rules", "store", "result"),
         ("rules", "review", "")],
        "Each classifier receives audio independently. Uncertain or disagreeing results can be routed for human review.")

gtm_nodes = {
    "audio": (0.5, 3.2, 2.0, 1.15, "Mono audio\nfrom the app", "#dbeafe"),
    "resample": (3.0, 3.2, 2.0, 1.15, "Resample to\n16 kHz", "#e0f2fe"),
    "window": (5.5, 3.2, 2.0, 1.15, "1 second windows\npad to 16,384 samples", "#fef3c7"),
    "mel": (8.0, 3.2, 2.0, 1.15, "31 x 64 log-mel\nnormalized features", "#dcfce7"),
    "cnn": (8.0, 1.15, 2.0, 1.15, "Converted GTM\nCNN, 10 scores", "#f3e8ff"),
    "average": (5.0, 1.15, 2.1, 1.15, "Average window\nprobabilities", "#fee2e2"),
    "result": (2.0, 1.15, 2.1, 1.15, "GTM class and\nconfidence scores", "#e2e8f0"),
}
diagram("gtm-model-flow.png", "GTM audio input and inference", gtm_nodes,
        [("audio", "resample", ""), ("resample", "window", ""),
         ("window", "mel", ""), ("mel", "cnn", ""),
         ("cnn", "average", ""), ("average", "result", "")],
        "Features follow the dimensions in gtm_model/metadata.json. The app averages scores when an audio file has multiple windows.")
