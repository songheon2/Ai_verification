"""ReLU split과 DPLL/Simplex progress를 하나의 realtime HTML로 표시한다."""
from __future__ import annotations

import json
from pathlib import Path
import webbrowser


class DedicatedRealtimeView:
    """하나의 실시간 PNG만 자동 갱신하는 전용 브라우저 화면."""

    def __init__(
        self,
        image_path: str | Path,
        output_path: str | Path,
        *,
        title: str,
        description: str,
        refresh_interval_ms: int = 500,
    ) -> None:
        if refresh_interval_ms <= 0:
            raise ValueError("refresh_interval_ms는 양수여야 합니다")
        self.image_path = Path(image_path)
        self.output_path = Path(output_path)
        self.title = title
        self.description = description
        self.refresh_interval_ms = refresh_interval_ms

    def write(self, *, open_browser: bool = False) -> Path:
        image_uri = json.dumps(self.image_path.resolve().as_uri())
        title = json.dumps(self.title, ensure_ascii=False)[1:-1]
        description = json.dumps(self.description, ensure_ascii=False)[1:-1]
        html = f"""<!doctype html>
<html lang="ko">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>{title}</title>
  <style>
    :root {{ color-scheme: dark; }}
    body {{ margin: 0; background: #0f172a; color: #f8fafc; font-family: Arial, sans-serif; }}
    header {{ position: sticky; top: 0; padding: 14px 20px; background: #111827ee; }}
    h1 {{ margin: 0; font-size: 22px; }}
    p {{ margin: 4px 0 0; color: #94a3b8; }}
    main {{ max-width: 1550px; margin: 0 auto; padding: 18px; }}
    img {{ display: block; width: 100%; min-height: 120px; object-fit: contain; background: white; border-radius: 8px; }}
    #status {{ color: #22c55e; font-weight: bold; }}
  </style>
</head>
<body>
  <header>
    <h1>{title}</h1>
    <p><span id="status">LIVE</span> · {self.refresh_interval_ms}ms 간격 자동 갱신 · {description}</p>
  </header>
  <main><img id="dashboard" alt="{title}"></main>
  <script>
    const imageUri = {image_uri};
    const image = document.getElementById("dashboard");
    function refresh() {{ image.src = imageUri + "?t=" + Date.now(); }}
    refresh();
    setInterval(refresh, {self.refresh_interval_ms});
  </script>
</body>
</html>
"""
        self.output_path.parent.mkdir(parents=True, exist_ok=True)
        self.output_path.write_text(html, encoding="utf-8")
        if open_browser:
            webbrowser.open(self.output_path.resolve().as_uri())
        return self.output_path


class UnifiedRealtimeDashboard:
    def __init__(
        self,
        split_image_path: str | Path,
        dpll_theory_image_path: str | Path,
        simplex_image_path: str | Path,
        output_path: str | Path,
        *,
        refresh_interval_ms: int = 500,
    ) -> None:
        if refresh_interval_ms <= 0:
            raise ValueError("refresh_interval_ms는 양수여야 합니다")
        self.split_image_path = Path(split_image_path)
        self.dpll_theory_image_path = Path(dpll_theory_image_path)
        self.simplex_image_path = Path(simplex_image_path)
        self.output_path = Path(output_path)
        self.refresh_interval_ms = refresh_interval_ms

    def write(self, *, open_browser: bool = False) -> Path:
        split_uri = json.dumps(self.split_image_path.resolve().as_uri())
        dpll_theory_uri = json.dumps(self.dpll_theory_image_path.resolve().as_uri())
        simplex_uri = json.dumps(self.simplex_image_path.resolve().as_uri())
        html = f"""<!doctype html>
<html lang="ko">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>AI Verification Realtime Dashboard</title>
  <style>
    :root {{ color-scheme: dark; }}
    body {{ margin: 0; background: #0f172a; color: #f8fafc; font-family: Arial, sans-serif; }}
    header {{ position: sticky; top: 0; z-index: 2; padding: 14px 20px; background: #111827ee; backdrop-filter: blur(8px); }}
    h1 {{ margin: 0; font-size: 22px; }}
    header p {{ margin: 4px 0 0; color: #94a3b8; }}
    main {{ max-width: 1550px; margin: 0 auto; padding: 18px; display: grid; gap: 18px; }}
    section {{ padding: 14px; background: #1e293b; border: 1px solid #334155; border-radius: 12px; }}
    h2 {{ margin: 0 0 10px; font-size: 18px; }}
    img {{ display: block; width: 100%; min-height: 120px; object-fit: contain; background: white; border-radius: 8px; }}
    #status {{ color: #22c55e; font-weight: bold; }}
  </style>
</head>
<body>
  <header>
    <h1>AI Verification Realtime Dashboard</h1>
    <p><span id="status">LIVE</span> · {self.refresh_interval_ms}ms 간격 자동 갱신 · DPLL + Theory + Simplex + ReLU</p>
  </header>
  <main>
    <section>
      <h2>DPLL Rounds &amp; Theory Flow</h2>
      <img id="dpll-theory-dashboard" alt="DPLL rounds and theory flow dashboard">
    </section>
    <section>
      <h2>Simplex Internals</h2>
      <img id="simplex-dashboard" alt="Simplex internal progress dashboard">
    </section>
    <section>
      <h2>Realtime ReLU Split History</h2>
      <img id="split-dashboard" alt="Realtime ReLU split history">
    </section>
  </main>
  <script>
    const dpllTheoryUri = {dpll_theory_uri};
    const simplexUri = {simplex_uri};
    const splitUri = {split_uri};
    const dpllTheoryImage = document.getElementById("dpll-theory-dashboard");
    const simplexImage = document.getElementById("simplex-dashboard");
    const splitImage = document.getElementById("split-dashboard");
    function refresh() {{
      const cacheBuster = "?t=" + Date.now();
      dpllTheoryImage.src = dpllTheoryUri + cacheBuster;
      simplexImage.src = simplexUri + cacheBuster;
      splitImage.src = splitUri + cacheBuster;
    }}
    refresh();
    setInterval(refresh, {self.refresh_interval_ms});
  </script>
</body>
</html>
"""
        self.output_path.parent.mkdir(parents=True, exist_ok=True)
        self.output_path.write_text(html, encoding="utf-8")
        if open_browser:
            webbrowser.open(self.output_path.resolve().as_uri())
        return self.output_path
