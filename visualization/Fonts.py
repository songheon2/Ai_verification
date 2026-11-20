"""One bundled font for browser reports and rendered graphs; no runtime network."""
from functools import lru_cache
from pathlib import Path
import shutil
from threading import Lock


FONT_FAMILY = "Noto Sans KR"
FONT_DIRECTORY = Path(__file__).with_name("fonts")
FONT_ASSETS = ("NotoSansKR-Regular.ttf", "NotoSansKR-Bold.ttf", "OFL.txt")
WEB_FONT_DIRECTORY = "_fonts"
_web_font_lock = Lock()


@lru_cache(maxsize=1)
def _register_matplotlib_fonts():
    from matplotlib import font_manager
    for name in FONT_ASSETS[:2]:
        font_manager.fontManager.addfont(str(FONT_DIRECTORY / name))


def configure_matplotlib_fonts():
    import matplotlib
    _register_matplotlib_fonts()
    # Explicit monospace labels use the same face as the surrounding graphs.
    matplotlib.rcParams.update({
        "font.family": [FONT_FAMILY], "font.sans-serif": [FONT_FAMILY],
        "font.monospace": [FONT_FAMILY], "axes.unicode_minus": False,
        "mathtext.fontset": "custom", "mathtext.rm": FONT_FAMILY,
        "mathtext.it": FONT_FAMILY, "mathtext.bf": FONT_FAMILY + ":bold",
        "mathtext.sf": FONT_FAMILY, "mathtext.tt": FONT_FAMILY,
        "mathtext.cal": FONT_FAMILY,
    })


def font_style(output_path):
    """Publish fonts once beside reports, keeping pages small and self-hosted."""
    root = Path(output_path).resolve().parent
    target = root / WEB_FONT_DIRECTORY
    if target.resolve().parent != root:
        raise ValueError("Visualization font directory must stay inside the report directory")
    with _web_font_lock:
        target.mkdir(parents=True, exist_ok=True)
        for name in FONT_ASSETS:
            destination = target / name
            if destination.resolve().parent != target.resolve():
                raise ValueError("Visualization font assets cannot be external symlinks")
            if not destination.exists():
                shutil.copyfile(FONT_DIRECTORY / name, destination)
    return '''<style data-visualization-font>
@font-face{font-family:"Noto Sans KR";font-style:normal;font-weight:400;font-display:swap;src:url("_fonts/NotoSansKR-Regular.ttf") format("truetype")}
@font-face{font-family:"Noto Sans KR";font-style:normal;font-weight:700;font-display:swap;src:url("_fonts/NotoSansKR-Bold.ttf") format("truetype")}
:root{--visualization-font:"Noto Sans KR",sans-serif}
body,button,input,select,textarea,svg,text,code,pre{font-family:var(--visualization-font)}
</style>'''


def apply_web_font(document, output_path):
    style = font_style(output_path)
    if "data-visualization-font" in document:
        return document
    if "</head>" in document:
        return document.replace("</head>", style + "</head>", 1)
    return style + document
