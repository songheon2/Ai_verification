"""Standalone report for the four Reluplex/Simplex mechanisms."""
from html import escape
from pathlib import Path
from visualization.AtomicFile import write_atomic_text
from visualization.Fonts import apply_web_font
from visualization.LiveSnapshot import initial_page_data


_SCRIPT = (Path(__file__).with_name("AssetRefresh.js").read_text(encoding="utf-8") + "\n" +
           Path(__file__).with_name("SolverMechanismsClient.js").read_text(encoding="utf-8"))


def preview_mechanisms(full):
    report = full["report"]
    return {**full, "report": {
        "available": report.get("available", False), "totals": report.get("totals", {}),
        "settings": report.get("settings", [])[-1:], "eventCount": report.get("eventCount", 0),
    }}


def mechanism_report_path(tree_path: str | Path) -> Path:
    path = Path(tree_path)
    return path.with_name(f"{path.stem}_mechanisms.html")


def write_solver_mechanisms_html(report, output_path, *, status, title="Reluplex 검증 기법", panels=None,
                                 live_snapshots=False, snapshot_only=False, snapshot_version=None) -> Path:
    payload = {"report": report, "status": status, "title": title,
               "panels": list(panels) if panels is not None else ["bounds", "backjump", "numerical", "underapprox"]}
    if live_snapshots:
        payload["snapshotVersion"] = snapshot_version
    data = initial_page_data(
        output_path, payload, "solver-mechanisms-snapshot",
        preview_mechanisms,
        live_snapshots=live_snapshots, snapshot_only=snapshot_only,
    )
    if snapshot_only:
        return Path(output_path)
    data = data.replace("<", "\\u003c")
    document = _HTML.replace("__TITLE__", escape(title)).replace("__SCRIPT__", _SCRIPT.replace("__DATA__", data))
    return write_atomic_text(output_path, apply_web_font(document, output_path))


_HTML = r'''<!doctype html>
<html lang="ko"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>__TITLE__</title><style>
*{box-sizing:border-box}body{margin:0;background:#fff;color:#1f2937;font:18px/1.5 var(--visualization-font)}
header,main{padding:10px 14px}header{position:relative;padding-right:180px;border-bottom:1px solid #dbe3ed}h1{font-size:22px;margin:0 0 2px}
#expandCharts{position:absolute;right:14px;top:10px}html:fullscreen{background:white}
.muted{color:#64748b;font-size:14px}.cards{display:grid;grid-template-columns:repeat(2,minmax(0,1fr));grid-template-rows:repeat(2,minmax(0,1fr));gap:8px;height:calc(100dvh - var(--report-header-height,85px) - 85px);min-height:560px}
.card{display:flex;flex-direction:column;padding:8px;border:1px solid #cbd5e1;border-radius:10px;background:#f8fafc;min-width:0;min-height:0;overflow-wrap:anywhere}
.card[hidden]{display:none}
.card-content{flex:none;min-height:var(--mechanism-controls-height,0px)}
.card h2{font-size:19px;margin:0 0 3px;line-height:1.3}.value{font-size:16px;font-weight:650;line-height:1.35}
.chart{display:flex;flex-direction:column;flex:1;min-height:0;margin-top:4px;background:white;border:1px solid #e2e8f0;border-radius:8px;padding:5px;min-width:0}
.chart svg{display:block;flex:1;min-height:0;width:100%;height:100%;font:17px var(--visualization-font)}.chart-title{font-weight:650;font-size:15px;margin-bottom:2px;line-height:1.25}
.chart-empty{color:#64748b;padding:9px 5px;text-align:center;font-size:14px}
.records{margin-top:10px;padding:8px 12px;border:1px solid #dbe3ed;border-radius:10px}.records>summary{cursor:pointer;font-size:18px;font-weight:650;line-height:1.4}.records[open]>.controls{margin-top:12px}
select,button,input{font:inherit;color:inherit;background:white;border:1px solid #cbd5e1;border-radius:6px;padding:7px 10px;max-width:100%}
input{width:130px}.card select,.card button,.card input{font-size:14px;padding:4px 6px}.graph-controls{margin-top:4px;display:flex;gap:6px;align-items:center;flex-wrap:wrap;font-size:14px}.graph-controls label{display:flex;gap:4px;align-items:center;min-width:0}
#boundStep,#boundSearchStep{width:100%;height:18px;padding:0}.bound-playback{margin-top:6px;display:flex;gap:5px;align-items:center;flex-wrap:wrap}.bound-progress{flex:1;min-width:100px;font-size:15px}
.graph-controls[hidden],.bound-playback[hidden]{display:none}.bound-toolbar{display:flex;gap:6px;align-items:center;flex-wrap:wrap;margin-top:6px}.bound-mode{display:flex;gap:5px;flex-wrap:wrap}.bound-mode button[aria-pressed="true"]{background:#dbeafe;border-color:#60a5fa;color:#1d4ed8}#boundSearch{width:150px}
#boundPending{color:#166534;background:#dcfce7;border-radius:6px;padding:2px 5px;font-size:15px;font-weight:700;white-space:nowrap}
.bound-toolbar>.bound-playback{margin-top:0;margin-left:auto;justify-content:flex-end;flex-wrap:nowrap}
.bound-timeline{display:flex;align-items:center;gap:8px;margin-top:4px}
.bound-timeline[hidden]{display:none}.bound-timeline .bound-progress{flex:0 1 auto;min-width:0;font-size:14px}
.bound-timeline #boundStep{flex:1;min-width:80px;width:auto}
.controls{display:flex;align-items:center;flex-wrap:wrap;gap:10px}.events{margin-top:12px}
.event{padding:10px 0;border-top:1px solid #e2e8f0;overflow-wrap:anywhere}.event strong{display:block;font-size:17px;color:#475569}
@media(max-width:850px){.cards{grid-template-columns:1fr;grid-template-rows:none;height:auto;min-height:0}.card{min-height:360px}.chart{min-height:180px}header,main{padding:10px}#expandCharts{position:static;margin-bottom:6px}.bound-toolbar{align-items:flex-start}}
#snapshotLoader{display:none}
.mechanism-combo{position:relative;display:inline-flex;width:132px;min-width:0}.card .mechanism-combo input{width:0;min-width:0;flex:1;padding:4px 6px;border-radius:6px 0 0 6px}.card .mechanism-combo>button{width:24px;padding:4px 0;border-left:0;border-radius:0 6px 6px 0}
.mechanism-choices{position:absolute;top:calc(100% + 3px);left:0;z-index:20;min-width:100%;width:max-content;max-width:230px;max-height:200px;overflow:auto;background:white;border:1px solid #cbd5e1;border-radius:6px;padding:3px;box-shadow:0 4px 12px #0002}.mechanism-choices[hidden]{display:none}.card .mechanism-choices button{display:block;width:100%;border:0;text-align:left;padding:5px 8px}
</style></head><body>
<header><button id="expandCharts" type="button">그림 크게 보기</button><h1 id="title"></h1><div id="status" class="muted"></div></header>
<main><section class="cards" aria-label="검증 기법 요약">
<article class="card"><h2>1. 경계 유도</h2><div id="boundsValue" class="value"></div>
<div class="bound-toolbar"><select id="boundScope" aria-label="경계 유도 표시 범위"><option value="stable">STABLE 확정만</option><option value="all">모든 경계 변경</option></select>
<div class="bound-mode"><button id="boundFollowMode" aria-pressed="true">실시간 과정 따라가기</button><button id="boundSearchMode" aria-pressed="false">뉴런 검색</button></div>
<div class="bound-playback"><button id="boundPrev" aria-label="이전 경계 변경">←</button><button id="boundPlay">일시정지</button><button id="boundNext" aria-label="다음 경계 변경">→</button><span id="boundPending" aria-live="polite">+0개 재생 대기</span></div></div>
<div class="bound-timeline" id="boundTimeline"><span id="boundPosition" class="bound-progress"></span><input id="boundStep" type="range" min="1" max="1" value="1" aria-label="경계 변경 순서"></div>
<div class="graph-controls" id="boundSearchControls" hidden><label>DPLL <select id="boundRound" aria-label="경계 유도 DPLL round"></select></label><label>Layer <input id="boundLayerSearch" inputmode="numeric" placeholder="Layer 번호" aria-label="경계 유도 레이어 입력 또는 선택"></label><label>Neuron <input id="boundNeuronSearch" inputmode="numeric" placeholder="Neuron 번호" aria-label="경계 유도 뉴런 입력 또는 선택"></label><select id="boundNeuron" hidden aria-label="경계 유도 선택 변수"></select></div>
<div class="bound-playback" id="boundSearchPlayback" hidden><button id="boundSearchPrev" aria-label="검색 뉴런 이전 경계 변경">←</button><button id="boundSearchNext" aria-label="검색 뉴런 다음 경계 변경">→</button><span id="boundSearchPosition" class="bound-progress"></span><input id="boundSearchStep" type="range" min="1" max="1" value="1" aria-label="검색 뉴런 경계 변경 순서"></div></article>
<article class="card"><h2>2. 충돌 분석·백점프</h2><div id="backjumpValue" class="value"></div></article>
<article class="card"><h2>3. 부동소수점 연산</h2><div id="numericalValue" class="value"></div>
<div class="graph-controls"><select id="numericalMetric" aria-label="부동소수점 그래프"><option value="drift">상대 계수 오차 · 복원</option><option value="residual">원래 등식의 할당 잔차 합</option><option value="pivot">피벗 계수 (표본)</option></select></div></article>
<article class="card"><h2>4. 과소근사</h2><div id="underapproxValue" class="value">솔버 미적용 · 구간 미리보기</div>
<div class="graph-controls"><label>ε <input id="epsilon" type="number" min="0" step="any" value="0.000001" aria-label="과소근사 미리보기 epsilon"></label><label>뉴런 <select id="underNeuron" aria-label="과소근사 미리보기 뉴런"></select></label></div></article>
</section>
<details class="records" id="occurrenceRecords"><summary>기법별 발생 기록</summary><div class="controls">
<select id="mechanismFilter" aria-label="기법 선택"><option value="all">모든 기법</option><option value="bounds">경계 유도</option><option value="backjump">충돌 분석·백점프</option><option value="numerical">부동소수점 연산</option><option value="underapprox">과소근사</option></select>
<span id="eventCount" class="muted"></span></div><div id="events" class="events"></div>
</details>
</main><script>__SCRIPT__</script></body></html>'''
