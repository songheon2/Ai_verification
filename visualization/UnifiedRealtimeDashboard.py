"""선택된 ReLU/DPLL/Simplex 패널을 하나의 realtime HTML로 표시한다."""
from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Mapping, Optional
from urllib.parse import quote

from visualization.RenderProcess import REALTIME_RENDER_INTERVAL_SECONDS
from visualization.AtomicFile import write_atomic_text
from visualization.ReluNeuronHover import HOVER_SCRIPT
from visualization.LogAnalysisPanel import panel_markup
from visualization.Fonts import font_style
import webbrowser

_ASSET_REFRESH_SCRIPT = Path(__file__).with_name("AssetRefresh.js").read_text(encoding="utf-8")
def publish_dashboard_status(dashboard_path: str | Path, status: str) -> Path:
    path = Path(dashboard_path).with_suffix(".status.js")
    return write_atomic_text(path, f"window.receiveVerificationStatus?.({json.dumps(status, ensure_ascii=True)});\n")

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
    "solver-mechanisms": ("Reluplex Solver Mechanisms", "Reluplex 검증 기법"),
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
        managed_completion: bool = False,
    ) -> None:
        if refresh_interval_ms <= 0:
            raise ValueError("refresh_interval_ms는 양수여야 합니다")
        unknown = set(panel_images) - set(PANEL_SPECS)
        if unknown:
            raise ValueError(f"지원하지 않는 realtime panel: {sorted(unknown)}")
        if not panel_images and not embedded_pages:
            raise ValueError("하나 이상의 realtime panel이 필요합니다")
        self.panel_images = {
            name: Path(panel_images[name])
            for name in PANEL_SPECS
            if name in panel_images
        }
        self.output_path = Path(output_path)
        self.refresh_interval_ms = refresh_interval_ms
        self.managed_completion = managed_completion
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
                f'    <section class="{name}-section">\n'
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
            section_class = ' class="mechanisms-section"' if name == "solver-mechanisms" else ""
            sections.append(
                f"    <section{section_class}>\n"
                f"      <h2>{title}</h2>\n"
                '      <label class="auto-refresh"><input type="checkbox" '
                f'id="{element_id}-auto" checked> auto refresh</label>\n'
                f'      <iframe id="{element_id}" title="{iframe_title}" allowfullscreen></iframe>\n'
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

        selected_names = " + ".join([*self.panel_images, *self.embedded_pages])
        panel_json = json.dumps(browser_panels, ensure_ascii=False)
        page_json = json.dumps(browser_pages, ensure_ascii=False)
        completion_uri = json.dumps(quote(self.output_path.with_suffix(".status.js").name)
                                    if self.managed_completion else None)
        html = f"""<!doctype html>
<html lang="ko">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>AI Verification Realtime Dashboard</title>
  {font_style(self.output_path)}
  <style>
    :root {{ color-scheme: dark; }}
    body {{ margin: 0; background: #0f172a; color: #f8fafc; font-family: var(--visualization-font); }}
    header {{ position: sticky; top: 0; z-index: 2; padding: 14px 20px; background: #111827ee; backdrop-filter: blur(8px); }}
    h1 {{ margin: 0; font-size: 22px; }}
    header p {{ margin: 4px 0 0; color: #94a3b8; }}
    main {{ max-width: 1550px; margin: 0 auto; padding: 18px; display: grid; grid-template-columns: minmax(0, 1fr); gap: 18px; }}
    .mechanisms-section {{ width: min(1680px, calc(100vw - 36px)); justify-self: center; box-sizing: border-box; }}
    section {{ padding: 14px; background: #1e293b; border: 1px solid #334155; border-radius: 12px; }}
    .dpll-theory-section, .simplex-section, .relu-section {{ width: min(100%, 1080px); justify-self: center; box-sizing: border-box; }}
    .dpll-theory-section {{ width: min(var(--dpll-panel-width, 1080px), calc(100vw - 36px)); }}
    h2 {{ margin: 0 0 10px; font-size: 18px; }}
    img {{ display: block; width: 100%; min-height: 120px; object-fit: contain; background: white; border-radius: 8px; }}
    #dpll-theory-dashboard, #simplex-dashboard, #relu-dashboard {{ width: 100%; height: auto; margin-inline: auto; }}
    #dpll-theory-dashboard:not([src]), #simplex-dashboard:not([src]), #relu-dashboard:not([src]) {{ aspect-ratio: 35 / 24; }}
    iframe {{ display: block; width: 100%; height: 760px; border: 0; background: white; color-scheme: light; border-radius: 8px; }}
    #solver-mechanisms-dashboard {{ height: auto; aspect-ratio: 3 / 2; min-height: 900px; }}
    .auto-refresh {{ display: inline-flex; gap: 6px; align-items: center; margin: -2px 0 10px; color: #cbd5e1; font-size: 13px; }}
    #status {{ color: #22c55e; font-weight: bold; }}
  </style>
</head>
<body>
  <header>
    <h1>AI Verification Realtime Dashboard</h1>
    <p><span id="status">LIVE</span> · 이미지 약 {REALTIME_RENDER_INTERVAL_SECONDS:g}초 주기 생성 · 브라우저 {self.refresh_interval_ms}ms 간격 확인 · {selected_names}</p>
  </header>
  <main class="dashboard-layout">
    <div class="dashboard-graphs">
{chr(10).join(sections)}
    </div>
{panel_markup()}
  </main>
  <script>
    {_ASSET_REFRESH_SCRIPT}
    {HOVER_SCRIPT}
    const panels = {panel_json};
    const pages = {page_json};
    const completionUri = {completion_uri};
    const completionAsset = {{}};
    let completionStatus = null;
    let verificationComplete = false;
    let imageTimer;
    let pageTimer;
    let inputRefreshQueued = false;
    let inputFinalQueued = false;
    const inputBusy = () => window.logAnalysisInputBusy?.() === true;
    async function refreshPanel(panel, finalSnapshot = false) {{
      if(inputBusy()){{inputRefreshQueued=true;inputFinalQueued=inputFinalQueued||finalSnapshot;return;}}
      if (panel.pending) {{
        panel.finalQueued = panel.finalQueued || (finalSnapshot && !panel.finalPending);
        return;
      }}
      panel.pending = true;
      panel.finalPending = finalSnapshot;
      const current = document.getElementById(panel.id);
      const next = new Image();
      let loadTimer;
      const finished = () => {{
        clearTimeout(loadTimer);
        next.onload = next.onerror = null;
        panel.pending = false;
        panel.finalPending = false;
        if (panel.finalQueued) {{ panel.finalQueued = false; refreshPanel(panel, true); }}
      }};
      next.onload = async () => {{
        clearTimeout(loadTimer);
        if(inputBusy()){{inputRefreshQueued=true;inputFinalQueued=inputFinalQueued||finalSnapshot;finished();return;}}
        const [selection, hover] = await Promise.all([
          window.loadAnalysisSelection ? loadAnalysisSelection(panel.uri) : Promise.resolve(null),
          panel.id === "relu-dashboard" ? loadReluNeuronHover(panel.uri) : Promise.resolve(null),
        ]);
        if(inputBusy()){{inputRefreshQueued=true;inputFinalQueued=inputFinalQueued||finalSnapshot;finished();return;}}
        if ((panel.hasSelectionMap && selection === null) || (panel.hasHoverMap && hover === null)) {{ finished(); return; }}
        if (panel.id === "dpll-theory-dashboard" && next.naturalHeight > 0) {{
          const width = Math.max(1050, Math.min(1500, 720 * next.naturalWidth / next.naturalHeight));
          current.closest("section").style.setProperty("--dpll-panel-width", (width + 30) + "px");
        }}
        panel.displayed = next;
        current.src = next.src;
        current.dataset.renderVersion = String(Number(current.dataset.renderVersion || 0) + 1);
        if (panel.id === "relu-dashboard") commitReluNeuronHover(current,panel.uri,hover);
        if (window.commitAnalysisSelection) commitAnalysisSelection(current,selection);
        panel.hasSelectionMap = selection !== null;
        panel.hasHoverMap = hover !== null;
        panel.version = update.version;
        if (finalSnapshot && completionUri) panel.finalPublished = true;
        finished();
        finishIfReady();
      }};
      next.onerror = finished;
      let update;
      try {{ update = await visualizationAssetUpdate(panel.uri, panel, finalSnapshot); }}
      catch (_error) {{ finished(); return; }}
      if (!update) {{ finished(); return; }}
      loadTimer = setTimeout(finished, 5000);
      next.src = update.url;
    }}
    function refresh(finalSnapshot = false) {{
      if (verificationComplete && !finalSnapshot) return;
      if (completionUri) {{
        pollCompletion();
        finalSnapshot = completionStatus !== null;
      }}
      for (const panel of panels) {{
        if (!completionUri || !panel.finalPublished) refreshPanel(panel, finalSnapshot);
      }}
      refreshPages(finalSnapshot);
      finishIfReady();
    }}
    async function pollCompletion() {{
      if (completionStatus !== null || completionAsset.pending) return;
      completionAsset.pending = true;
      try {{
        const update = await visualizationAssetUpdate(completionUri, completionAsset);
        if (!update) return;
        await new Promise((resolve, reject) => {{
          const script = document.createElement("script");
          const timer = setTimeout(() => done(false), 5000);
          const done = (loaded) => {{
            clearTimeout(timer);
            script.remove();
            if (loaded) resolve(); else reject(Error("Verification status unavailable"));
          }};
          script.onload = () => done(true);
          script.onerror = () => done(false);
          script.src = update.url;
          document.head.append(script);
        }});
        completionAsset.version = update.version;
      }} catch (_error) {{
        // A later tick retries transient file and connection failures.
      }} finally {{
        completionAsset.pending = false;
      }}
    }}
    window.receiveVerificationStatus = (status) => {{
      if (!completionUri || verificationComplete || completionStatus !== null) return;
      if (status === "FINALIZING") {{
        document.getElementById("status").textContent = "FINALIZING";
        return;
      }}
      if (status === "CANCELLED") {{
        markComplete(status);
        return;
      }}
      if (!["SAT", "UNSAT", "UNKNOWN"].includes(status)) return;
      completionStatus = status;
      refresh(true);
    }};
    function finishIfReady() {{
      if (!completionUri || verificationComplete || completionStatus === null ||
          inputBusy() || panels.some(panel => !panel.finalPublished)) return;
      markComplete(completionStatus);
    }}
    function refreshPage(page, finalSnapshot = false) {{
      if(inputBusy()){{inputRefreshQueued=true;inputFinalQueued=inputFinalQueued||finalSnapshot;page.inputQueued=true;return;}}
      page.inputQueued=false;
      const frame = document.getElementById(page.id);
      if (!frame.dataset.loaded) {{
        if (frame.dataset.loading) {{ frame.dataset.refreshQueued = "true"; return; }}
        frame.dataset.loading = "true";
        frame.src = page.uri + "?t=" + Date.now();
      }} else {{
        frame.contentWindow.postMessage({{type: "relu-search-tree-refresh", final: finalSnapshot}}, "*");
      }}
    }}
    function refreshPages(finalSnapshot = false) {{
      if (verificationComplete && !finalSnapshot) return;
      for (const page of pages) {{
        if (document.getElementById(page.autoId).checked || (finalSnapshot && page.inputQueued)) {{
          refreshPage(page, finalSnapshot);
        }}
      }}
    }}
    for (const page of pages) {{
      const frame = document.getElementById(page.id);
      frame.addEventListener("load", () => {{
        if (!frame.dataset.loading) return;
        delete frame.dataset.loading;
        frame.dataset.loaded = "true";
        frame.contentWindow.postMessage({{type:'visualization-input-activity',active:inputBusy()}},'*');
        const refreshQueued = frame.dataset.refreshQueued === "true";
        delete frame.dataset.refreshQueued;
        if (refreshQueued || document.getElementById(page.autoId).checked) refreshPage(page, verificationComplete);
      }});
      document.getElementById(page.autoId).addEventListener("change", (event) => {{
        if (event.target.checked) refreshPage(page, verificationComplete);
      }});
    }}
    window.addEventListener("message", (event) => {{
      if (verificationComplete || completionUri) return;
      const message = event.data || {{}};
      if (message.type !== "relu-search-tree-status" || message.status === "RUNNING") return;
      const knownPage = pages.some((page) =>
        document.getElementById(page.id).contentWindow === event.source
      );
      if (!knownPage) return;
      // Let the independent report consume its final snapshot before polling stops.
      refresh(true);
      markComplete(message.status);
    }});
    function markComplete(result) {{
      verificationComplete = true;
      clearInterval(imageTimer);
      clearInterval(pageTimer);
      for (const page of pages) document.getElementById(page.autoId).checked = false;
      const status = document.getElementById("status");
      status.textContent = "DONE · " + result;
      status.style.color = result === "SAT" ? "#22c55e" :
        result === "UNSAT" ? "#ef4444" : "#94a3b8";
    }}
    window.addEventListener("log-analysis-disconnected", () => {{
      verificationComplete = true;
      clearInterval(imageTimer);
      clearInterval(pageTimer);
      for (const page of pages) document.getElementById(page.autoId).checked = false;
      document.getElementById("status").textContent = "OFFLINE";
      document.getElementById("status").style.color = "#94a3b8";
    }});
    window.addEventListener('log-analysis-input-activity',event=>{{
      if(event.detail.active||!inputRefreshQueued)return;
      const finalSnapshot=inputFinalQueued;inputRefreshQueued=false;inputFinalQueued=false;
      refresh(finalSnapshot);
    }});
    refresh();
    imageTimer = setInterval(refresh, {self.refresh_interval_ms});
  </script>
</body>
</html>
"""
        self.output_path.parent.mkdir(parents=True, exist_ok=True)
        if self.managed_completion:
            publish_dashboard_status(self.output_path, "RUNNING")
        write_atomic_text(self.output_path, html)
        if open_browser:
            webbrowser.open(self.output_path.resolve().as_uri())
        return self.output_path
