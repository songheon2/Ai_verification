"""The dashboard's item-scoped LLM panel (API keys never enter the page)."""
from pathlib import Path
import json
from visualization.VerificationPicker import analysis_styles, analysis_usage_markup, picker_markup


def panel_markup():
    script = Path(__file__).with_suffix(".js").read_text(encoding="utf-8")
    selection = Path(__file__).with_name("AnalysisSelection.js").read_text(encoding="utf-8")
    markdown = Path(__file__).with_name("LogAnalysisMarkdown.js").read_text(encoding="utf-8")
    vocabulary = json.loads(Path(__file__).with_name("LogAnalysisTerms.json").read_text(encoding="utf-8"))
    vocabulary_script = "window.logAnalysisVocabulary=" + json.dumps(vocabulary, ensure_ascii=False).replace("<", "\\u003c") + ";\n"
    return r'''<!-- LLM_ANALYSIS_PANEL_START --><section id="llm-analysis-panel" aria-label="항목별 LLM 분석">
''' + analysis_styles() + r'''<style>
#llm-analysis-panel{box-sizing:border-box;width:min(100%,1080px);justify-self:center;color:#e2e8f0}
main.dashboard-layout{--analysis-edge:16px;--analysis-top:calc(var(--dashboard-header-height,0px) + var(--analysis-edge));--analysis-height:calc(100dvh - var(--analysis-top) - var(--analysis-edge));display:grid;grid-template-columns:minmax(0,1fr);align-items:start}
.dashboard-graphs{display:grid;grid-template-columns:minmax(0,1fr);gap:18px;min-width:0}
.dashboard-graphs>.mechanisms-section{width:min(100%,1680px)}
.dashboard-graphs>.dpll-theory-section{width:min(var(--dpll-panel-width,1080px),100%)}
main.dashboard-layout>#llm-analysis-panel{min-width:0;scroll-margin-top:var(--analysis-top)}
@media(min-width:1080px){
  main.dashboard-layout{max-width:2000px;grid-template-columns:minmax(0,1fr) clamp(360px,32vw,510px);gap:14px;padding:var(--analysis-edge)}
  main.dashboard-layout>.dashboard-graphs :is(#dpll-theory-dashboard,#simplex-dashboard,#relu-dashboard){width:min(100%,calc(max(180px,var(--analysis-height) - 64px) * var(--graph-aspect,1.45833)));min-height:0}
  main.dashboard-layout>#llm-analysis-panel{display:flex;flex-direction:column;position:sticky;top:var(--analysis-top);width:100%;height:var(--analysis-height);padding:12px;overflow:auto;scrollbar-gutter:stable;overscroll-behavior-y:contain}
  main.dashboard-layout>#llm-analysis-panel>:not(.llm-report){flex:none}
  main.dashboard-layout #llm-analysis-panel h2{margin-bottom:6px}
  main.dashboard-layout #llm-analysis-panel .llm-controls{gap:6px 8px;margin:6px 0}
  main.dashboard-layout #llm-analysis-panel .llm-action-row{gap:6px;margin:8px 0 6px}
  main.dashboard-layout #llm-analysis-panel .llm-selection-options{gap:6px;margin:6px 0}
  main.dashboard-layout #llm-analysis-panel .llm-usage-row{gap:5px 8px;margin:0 0 5px}
  main.dashboard-layout #llm-analysis-panel .llm-button{min-height:30px;padding:4px 10px}
  main.dashboard-layout #llm-analysis-panel .llm-state:empty{display:none}
  main.dashboard-layout #llm-analysis-panel #llm-selected-items{max-height:64px;overflow:auto;margin:0 0 6px}
  main.dashboard-layout #llm-analysis-panel #llm-question{height:clamp(64px,10dvh,128px);min-height:60px;max-height:180px}
  main.dashboard-layout #llm-analysis-panel .llm-report{display:flex;flex:1 1 auto;flex-direction:column;min-height:82px;margin-top:8px}
  main.dashboard-layout #llm-analysis-panel #llm-result{flex:1 1 auto;min-height:0;overflow:auto;overscroll-behavior:contain;padding:12px;font-size:14px}
  main.dashboard-layout #llm-analysis-panel .llm-report-header{flex:none;padding:8px 12px}
  main.dashboard-layout #llm-analysis-panel .llm-selected-item span{min-width:0}
}
@media(max-width:1079px){main.dashboard-layout>#llm-analysis-panel{grid-row:1}}
#llm-analysis-panel [hidden]{display:none!important}
#llm-analysis-panel .llm-controls{display:flex;flex-wrap:wrap;align-items:end;gap:10px;margin:12px 0}
#llm-analysis-panel .llm-mechanism-field{display:flex;flex-direction:column;gap:4px;min-width:0;max-width:100%;font-size:14px}
#llm-analysis-panel .llm-mechanism-tabs{display:flex;flex-wrap:wrap;gap:3px;padding:3px;border:1px solid #64748b;border-radius:9px;background:#16243a}
#llm-analysis-panel .llm-mechanism-tab{min-height:30px;border:0;border-radius:6px;padding:6px 10px;background:transparent;color:#cbd5e1;font-size:13px;white-space:nowrap}
#llm-analysis-panel .llm-mechanism-tab:hover{background:#263b58}
#llm-analysis-panel .llm-mechanism-tab[aria-checked="true"]{background:#246bd0;color:#fff;box-shadow:0 1px 4px #0003}
#llm-analysis-panel .llm-mechanism-tab:focus-visible{outline:2px solid #93c5fd;outline-offset:2px}
#llm-analysis-panel label{display:flex;flex-direction:column;gap:4px;font-size:14px}
#llm-analysis-panel #llm-model{max-width:min(100%,260px)}
#llm-analysis-panel #llm-effort{max-width:min(100%,260px)}
#llm-analysis-panel select,#llm-analysis-panel input,#llm-analysis-panel button,#llm-analysis-panel textarea{box-sizing:border-box;font:14px/1.4 "Noto Sans KR",sans-serif;padding:7px 9px;border:1px solid #64748b;border-radius:5px;background:#fff;color:#0f172a;max-width:100%}
#llm-analysis-panel input{width:150px}#llm-analysis-panel textarea{width:100%;min-height:180px;resize:vertical}
#llm-analysis-panel button{cursor:pointer}#llm-analysis-panel button:disabled{opacity:.5;cursor:default}
#llm-analysis-panel .llm-consent{flex-direction:row;align-items:center}#llm-analysis-panel .llm-consent input{width:auto}
#llm-analysis-panel .llm-selection-options{display:flex;align-items:center;flex-wrap:wrap;gap:10px;margin:10px 0}
#llm-analysis-panel #llm-selected-items{display:flex;flex-wrap:wrap;gap:6px;margin:8px 0 12px}
#llm-analysis-panel .llm-selected-item{display:flex;align-items:center;gap:7px;max-width:100%;padding:5px 9px;border:1px solid #475569;border-radius:8px;background:#16243a;font-size:12px}
#llm-analysis-panel .llm-selected-item span{overflow:hidden;text-overflow:ellipsis;white-space:nowrap;max-width:420px}
#llm-analysis-panel .llm-selected-item button{padding:0 4px;min-width:22px;font-size:15px}
#llm-analysis-panel .llm-state{color:#cbd5e1;font-size:14px;white-space:pre-wrap;overflow-wrap:anywhere}
#llm-analysis-panel .llm-status-details{margin:2px 0;font-size:12px;color:#cbd5e1}
#llm-analysis-panel .llm-status-details summary{cursor:pointer}
#llm-analysis-panel .llm-status-details[open]{max-height:120px;overflow:auto}
#llm-analysis-panel .llm-status-details[open] summary{margin-bottom:5px}
.llm-topic-button{font-size:12px;min-height:30px;margin-left:10px;padding:5px 12px}
</style>
<h2>항목별 LLM 분석</h2>
<div id="llm-connection" class="llm-state"></div>
''' + picker_markup() + r'''
<div class="llm-controls">
<label id="llm-model-label" hidden>모델<select id="llm-model"></select></label>
<label id="llm-effort-label" hidden>추론 강도<select id="llm-effort"><option value="low">낮음</option></select></label>
<label>분석 항목<select id="llm-topic"><option value="all">전체</option><option value="dpll">DPLL / Theory</option><option value="simplex">Simplex</option><option value="relu">ReLU 분기 이력</option><option value="tree">ReLU 탐색 트리</option><option value="mechanisms">Mechanisms</option></select></label>
<div id="llm-mechanism-label" class="llm-mechanism-field" hidden><span>기법</span><div class="llm-mechanism-tabs" role="radiogroup" aria-label="Mechanisms 기법 선택">
<button id="llm-mechanism-all" class="llm-mechanism-tab" type="button" role="radio" aria-checked="true" tabindex="0" data-mechanism="all">전체</button>
<button class="llm-mechanism-tab" type="button" role="radio" aria-checked="false" tabindex="-1" data-mechanism="bounds">경계 유도</button>
<button class="llm-mechanism-tab" type="button" role="radio" aria-checked="false" tabindex="-1" data-mechanism="backjump">충돌 분석·백점프</button>
<button class="llm-mechanism-tab" type="button" role="radio" aria-checked="false" tabindex="-1" data-mechanism="numeric">부동소수점</button>
<button class="llm-mechanism-tab" type="button" role="radio" aria-checked="false" tabindex="-1" data-mechanism="underapprox">과소근사</button>
</div></div>
<label>Round<select id="llm-round"><option value="">전체 Round</option></select></label>
<label id="llm-call-label" hidden>Simplex call<input id="llm-call" list="llm-call-list" placeholder="simplex-123"><datalist id="llm-call-list"></datalist></label>
<label id="llm-layer-label" hidden>Layer<select id="llm-layer"><option value="">전체 Layer</option></select></label>
<label id="llm-neuron-label" hidden>Neuron<select id="llm-neuron"><option value="">전체 Neuron</option></select></label>
</div>
<div class="llm-selection-options"><label class="llm-consent"><input type="checkbox" id="llm-multi-select">중복 선택</label><span id="llm-selected-count" class="llm-state"></span><button id="llm-clear-selection" class="llm-button" type="button" disabled>선택 비우기</button></div>
<div id="llm-selected-items" role="list" aria-label="선택한 분석 항목" hidden></div>
<label>질문<textarea id="llm-question" maxlength="2000" placeholder="비워두면 선택 항목의 반복·병목 패턴을 분석합니다."></textarea></label>
<div class="llm-controls llm-action-row"><label class="llm-consent"><input type="checkbox" id="llm-consent">선택 항목의 요약·로그 일부와 이번 세션의 질문·답변을 Google에 전송하는 데 동의합니다</label>
<div class="llm-buttons"><button id="llm-start" class="llm-button llm-primary" type="button" disabled>현재까지 분석</button><button id="llm-reload" class="llm-button" type="button">저장 결과 보기</button></div></div>
<details class="llm-status-details"><summary>실행·분석 정보</summary>
''' + analysis_usage_markup("llm") + r'''
<div id="llm-selection" class="llm-state"></div><div id="llm-live" class="llm-state"></div>
<div id="llm-snapshot" class="llm-state"></div><div id="llm-new-records" class="llm-state"></div>
</details>
<div id="llm-progress" role="status" class="llm-state"></div>
<div class="llm-report"><div class="llm-report-header"><span class="llm-report-title">분석 결과</span><span id="llm-result-tag" class="llm-result-tag">대기</span><button id="llm-copy" class="llm-button llm-copy" type="button" disabled>원문 복사</button></div><article id="llm-result" class="llm-result" aria-label="Markdown 분석 보고서">분석할 항목을 선택하세요.</article></div>
</section><script>''' + vocabulary_script + markdown + "\n" + selection + "\n" + script + "</script><!-- LLM_ANALYSIS_PANEL_END -->"


def install_panel(document):
    """Refresh the bridge UI in old dashboards without touching saved source files."""
    first, last = '<!-- LLM_ANALYSIS_PANEL_START -->', '<!-- LLM_ANALYSIS_PANEL_END -->'
    start = document.find(first)
    end = document.find(last, start) if start >= 0 else -1
    if start >= 0 and end >= 0:
        return document[:start] + panel_markup() + document[end + len(last):]
    start = document.find('<section id="llm-analysis-panel"')
    if start >= 0:
        # Before explicit markers, this section was immediately followed by its script.
        end = document.find('</script>', document.find('</section>', start))
        if end >= 0:
            return document[:start] + panel_markup() + document[end + len('</script>'):]
    return document.replace('</main>', panel_markup() + '</main>', 1)
