"""Standalone report for the four Reluplex/Simplex mechanisms."""
from html import escape
import json
from pathlib import Path
from visualization.AtomicFile import write_atomic_text
from visualization.LiveSnapshot import snapshot_path, write_snapshot


_SCRIPT = Path(__file__).with_name("SolverMechanismsClient.js").read_text(encoding="utf-8")


def mechanism_report_path(tree_path: str | Path) -> Path:
    path = Path(tree_path)
    return path.with_name(f"{path.stem}_mechanisms.html")


def write_solver_mechanisms_html(report, output_path, *, status, title="Reluplex 검증 기법", panels=None,
                                 live_snapshots=False, snapshot_only=False, snapshot_version=None) -> Path:
    payload = {"report": report, "status": status, "title": title,
               "panels": list(panels) if panels is not None else ["bounds", "backjump", "numerical", "underapprox"]}
    if live_snapshots:
        payload["snapshotVersion"] = snapshot_version
        payload["snapshotFile"] = snapshot_path(output_path).name
        write_snapshot(output_path, payload, "solver-mechanisms-snapshot")
    if snapshot_only:
        return Path(output_path)
    data = json.dumps(payload, ensure_ascii=False).replace("<", "\\u003c")
    document = _HTML.replace("__TITLE__", escape(title)).replace("__SCRIPT__", _SCRIPT.replace("__DATA__", data))
    return write_atomic_text(output_path, document)


_HTML = r'''<!doctype html>
<html lang="ko"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>__TITLE__</title><style>
*{box-sizing:border-box}body{margin:0;background:#fff;color:#1f2937;font:20px/1.6 system-ui,sans-serif}
header,main{padding:18px 22px}header{border-bottom:1px solid #dbe3ed}h1{font-size:28px;margin:0 0 6px}
.muted{color:#64748b;font-size:18px}.cards{display:grid;grid-template-columns:repeat(2,minmax(0,1fr));gap:18px}
.card{padding:16px;border:1px solid #cbd5e1;border-radius:10px;background:#f8fafc;min-width:0;overflow-wrap:anywhere}
.card h2{font-size:24px;margin:0 0 8px}.value{font-size:22px;font-weight:650}
.chart{margin-top:16px;background:white;border:1px solid #e2e8f0;border-radius:8px;padding:12px;min-width:0}
.chart svg{display:block;width:100%;height:auto;font:19px system-ui,sans-serif}.chart-title{font-weight:650;font-size:19px;margin-bottom:8px}
.chart-empty{color:#64748b;padding:32px 8px;text-align:center;font-size:19px}
.records{margin-top:20px;padding:16px;border:1px solid #dbe3ed;border-radius:10px}.records h2{font-size:24px;margin:0 0 12px}
select,button,input{font:inherit;color:inherit;background:white;border:1px solid #cbd5e1;border-radius:6px;padding:7px 10px;max-width:100%}
input{width:130px}.graph-controls{margin-top:12px;display:flex;gap:10px;align-items:center;flex-wrap:wrap}.graph-controls label{display:flex;gap:6px;align-items:center;min-width:0}
#boundStep,#boundSearchStep{width:100%;padding:0}.bound-playback{margin-top:12px;display:flex;gap:8px;align-items:center;flex-wrap:wrap}.bound-playback button{font-size:18px}.bound-progress{flex:1;min-width:140px;font-size:18px}
.graph-controls[hidden],.bound-playback[hidden]{display:none}.bound-mode{display:flex;gap:8px;flex-wrap:wrap;margin-top:12px}.bound-mode button[aria-pressed="true"]{background:#dbeafe;border-color:#60a5fa;color:#1d4ed8}#boundSearch{width:180px}
#boundPending{color:#166534;background:#dcfce7;border-radius:6px;padding:3px 8px;font-size:18px;font-weight:700;white-space:nowrap}
.controls{display:flex;align-items:center;flex-wrap:wrap;gap:10px}.events{max-height:460px;overflow:auto;margin-top:12px}
.event{padding:12px 0;border-top:1px solid #e2e8f0;overflow-wrap:anywhere}.event strong{display:block;font-size:19px;color:#475569}
@media(max-width:850px){.cards{grid-template-columns:1fr}header,main{padding:14px}}
#snapshotLoader{display:none}
</style></head><body>
<header><h1 id="title"></h1><div id="status" class="muted"></div></header>
<main><section class="cards" aria-label="검증 기법 요약">
<article class="card"><h2>1. 경계 유도</h2><div id="boundsValue" class="value"></div>
<div class="graph-controls"><select id="boundScope" aria-label="경계 유도 표시 범위"><option value="stable">STABLE 확정만</option><option value="all">모든 경계 변경</option></select></div>
<div class="bound-mode"><button id="boundFollowMode" aria-pressed="true">실시간 과정 따라가기</button><button id="boundSearchMode" aria-pressed="false">뉴런 검색</button></div>
<div class="bound-playback"><button id="boundPrev" aria-label="이전 경계 변경">←</button><button id="boundPlay">일시정지</button><button id="boundNext" aria-label="다음 경계 변경">→</button><span id="boundPosition" class="bound-progress"></span><span id="boundPending" aria-live="polite">+0개 재생 대기</span><input id="boundStep" type="range" min="1" max="1" value="1" aria-label="경계 변경 순서"></div>
<div class="graph-controls" id="boundSearchControls" hidden><label>검색 <input id="boundSearch" type="search" placeholder="뉴런 이름" aria-label="경계 유도 뉴런 검색"></label><label>DPLL <select id="boundRound" aria-label="경계 유도 DPLL round"></select></label><label>뉴런 <select id="boundNeuron" aria-label="경계 유도 뉴런"></select></label></div>
<div class="bound-playback" id="boundSearchPlayback" hidden><button id="boundSearchPrev" aria-label="검색 뉴런 이전 경계 변경">←</button><button id="boundSearchNext" aria-label="검색 뉴런 다음 경계 변경">→</button><span id="boundSearchPosition" class="bound-progress"></span><input id="boundSearchStep" type="range" min="1" max="1" value="1" aria-label="검색 뉴런 경계 변경 순서"></div></article>
<article class="card"><h2>2. 충돌 분석·백점프</h2><div id="backjumpValue" class="value"></div></article>
<article class="card"><h2>3. 부동소수점 연산</h2><div id="numericalValue" class="value"></div>
<div class="graph-controls"><select id="numericalMetric" aria-label="부동소수점 그래프"><option value="drift">상대 계수 오차 · 복원</option><option value="residual">원래 등식의 할당 잔차 합</option><option value="pivot">피벗 계수 (표본)</option></select></div></article>
<article class="card"><h2>4. 과소근사</h2><div id="underapproxValue" class="value">솔버 미적용 · 구간 미리보기</div>
<div class="graph-controls"><label>ε <input id="epsilon" type="number" min="0" step="any" value="0.000001" aria-label="과소근사 미리보기 epsilon"></label><label>뉴런 <select id="underNeuron" aria-label="과소근사 미리보기 뉴런"></select></label></div></article>
</section>
<section class="records"><h2>기법별 발생 기록</h2><div class="controls">
<select id="mechanismFilter" aria-label="기법 선택"><option value="all">모든 기법</option><option value="bounds">경계 유도</option><option value="backjump">충돌 분석·백점프</option><option value="numerical">부동소수점 연산</option><option value="underapprox">과소근사</option></select>
<span id="eventCount" class="muted"></span></div><div id="events" class="events"></div>
</section>
</main><script>__SCRIPT__</script></body></html>'''
