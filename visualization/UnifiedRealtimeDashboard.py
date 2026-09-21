"""선택된 ReLU/DPLL/Simplex 패널을 하나의 realtime HTML로 표시한다."""
from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Mapping, Optional
from urllib.parse import quote

from visualization.RenderProcess import REALTIME_RENDER_INTERVAL_SECONDS
import webbrowser


PANEL_SPECS = {
    "dpll-theory": (
        "DPLL Rounds &amp; Theory Flow",
        "DPLL rounds and theory flow dashboard",
    ),
    "simplex": (
        "Simplex Internals",
        "Simplex internal progress dashboard",
    ),
    "relu": (
        "Realtime ReLU Split History",
        "Realtime ReLU split history",
    ),
}

PAGE_SPECS = {
    "relu-tree": ("Interactive ReLU Search Tree", "Realtime ReLU search tree"),
}


class UnifiedRealtimeDashboard:
    """하나 이상의 선택 패널을 단일 자동 갱신 HTML로 묶는다."""

    def __init__(
        self,
        panel_images: Mapping[str, str | Path],
        output_path: str | Path,
        *,
        refresh_interval_ms: int = 500,
        embedded_pages: Optional[Mapping[str, str | Path]] = None,
    ) -> None:
        if refresh_interval_ms <= 0:
            raise ValueError("refresh_interval_ms는 양수여야 합니다")
        unknown = set(panel_images) - set(PANEL_SPECS)
        if unknown:
            raise ValueError(f"지원하지 않는 realtime panel: {sorted(unknown)}")
        if not panel_images:
            raise ValueError("하나 이상의 realtime panel이 필요합니다")
        self.panel_images = {
            name: Path(panel_images[name])
            for name in PANEL_SPECS
            if name in panel_images
        }
        self.output_path = Path(output_path)
        self.refresh_interval_ms = refresh_interval_ms
        self.embedded_pages = {
            name: Path(path) for name, path in (embedded_pages or {}).items()
        }
        unknown_pages = set(self.embedded_pages) - set(PAGE_SPECS)
        if unknown_pages:
            raise ValueError(f"unsupported realtime page: {sorted(unknown_pages)}")

    def write(self, *, open_browser: bool = False) -> Path:
        sections = []
        browser_panels = []
        dashboard_directory = self.output_path.parent.resolve()
        for name, image_path in self.panel_images.items():
            title, alt = PANEL_SPECS[name]
            element_id = f"{name}-dashboard"
            sections.append(
                "    <section>\n"
                f"      <h2>{title}</h2>\n"
                f'      <img id="{element_id}" alt="{alt}">\n'
                "    </section>"
            )
            try:
                relative_path = os.path.relpath(
                    image_path.resolve(), dashboard_directory
                )
            except ValueError:
                # Windows paths on different drives cannot be represented with
                # a relative path.  Keep an absolute file URI in that rare case.
                image_uri = image_path.resolve().as_uri()
            else:
                image_uri = quote(Path(relative_path).as_posix(), safe="/.")
            browser_panels.append({"id": element_id, "uri": image_uri})

        browser_pages = []
        for name, page_path in self.embedded_pages.items():
            title, iframe_title = PAGE_SPECS[name]
            element_id = f"{name}-dashboard"
            sections.append(
                "    <section>\n"
                f"      <h2>{title}</h2>\n"
                '      <label class="auto-refresh"><input type="checkbox" '
                f'id="{element_id}-auto" checked> auto refresh</label>\n'
                f'      <iframe id="{element_id}" title="{iframe_title}"></iframe>\n'
                "    </section>"
            )
            try:
                relative_path = os.path.relpath(
                    page_path.resolve(), dashboard_directory
                )
            except ValueError:
                page_uri = page_path.resolve().as_uri()
            else:
                page_uri = quote(Path(relative_path).as_posix(), safe="/.")
            browser_pages.append({
                "id": element_id,
                "autoId": f"{element_id}-auto",
                "uri": page_uri,
            })

        selected_names = " + ".join(self.panel_images)
        panel_json = json.dumps(browser_panels, ensure_ascii=False)
        page_json = json.dumps(browser_pages, ensure_ascii=False)
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
    iframe {{ display: block; width: 100%; height: 760px; border: 0; background: #0b1020; border-radius: 8px; }}
    .auto-refresh {{ display: inline-flex; gap: 6px; align-items: center; margin: -2px 0 10px; color: #cbd5e1; font-size: 13px; }}
    #status {{ color: #22c55e; font-weight: bold; }}
  </style>
</head>
<body>
  <header>
    <h1>AI Verification Realtime Dashboard</h1>
    <p><span id="status">LIVE</span> · 이미지 약 {REALTIME_RENDER_INTERVAL_SECONDS:g}초 주기 생성 · 브라우저 {self.refresh_interval_ms}ms 간격 확인 · {selected_names}</p>
  </header>
  <main>
{chr(10).join(sections)}
  </main>
  <script>
    const panels = {panel_json};
    const pages = {page_json};
    let verificationComplete = false;
    let imageTimer;
    let pageTimer;
    function refresh() {{
      if (verificationComplete) return;
      const cacheBuster = "?t=" + Date.now();
      for (const panel of panels) {{
        document.getElementById(panel.id).src = panel.uri + cacheBuster;
      }}
    }}
    function refreshPages() {{
      if (verificationComplete) return;
      const cacheBuster = "?t=" + Date.now();
      for (const page of pages) {{
        if (document.getElementById(page.autoId).checked) {{
          document.getElementById(page.id).src = page.uri + cacheBuster;
        }}
      }}
    }}
    window.addEventListener("message", (event) => {{
      const message = event.data || {{}};
      if (message.type !== "relu-search-tree-status" || message.status === "RUNNING") return;
      const knownPage = pages.some((page) =>
        document.getElementById(page.id).contentWindow === event.source
      );
      if (!knownPage) return;
      verificationComplete = true;
      clearInterval(imageTimer);
      clearInterval(pageTimer);
      for (const page of pages) document.getElementById(page.autoId).checked = false;
      const status = document.getElementById("status");
      status.textContent = "DONE · " + message.status;
      status.style.color = message.status === "SAT" ? "#22c55e" :
        message.status === "UNSAT" ? "#ef4444" : "#94a3b8";
    }});
    refresh();
    imageTimer = setInterval(refresh, {self.refresh_interval_ms});
    refreshPages();
    pageTimer = setInterval(refreshPages, {max(1000, self.refresh_interval_ms * 4)});
  </script>
</body>
</html>
"""
        self.output_path.parent.mkdir(parents=True, exist_ok=True)
        self.output_path.write_text(html, encoding="utf-8")
        if open_browser:
            webbrowser.open(self.output_path.resolve().as_uri())
        return self.output_path
