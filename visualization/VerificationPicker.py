"""ONNX/VNNLIB selector shared by the realtime dashboard and launcher page."""

from pathlib import Path

AI_STUDIO_QUOTA_URL = "https://aistudio.google.com/rate-limit"


def analysis_styles():
    css = Path(__file__).with_name("LogAnalysisPresentation.css").read_text(encoding="utf-8")
    return f"<style>{css}</style>"


def analysis_usage_markup(prefix):
    return f'''<div class="llm-usage-row"><a id="{prefix}-quota-link" class="llm-button llm-quota-link" href="{AI_STUDIO_QUOTA_URL}" target="_blank" rel="noopener noreferrer">AI Studio 한도</a><span data-quota-status role="status"></span></div>'''


def _catalog_combobox(name, label):
    return f'''
    <div class="verify-wide verify-lookup">
      <label for="verify-{name}-input">{label}</label>
      <div class="verify-combo">
        <input id="verify-{name}-input" type="text" role="combobox" aria-autocomplete="list"
          aria-expanded="false" aria-controls="verify-{name}-list" data-catalog-search="{name}"
          placeholder="{label} 선택" autocomplete="off" spellcheck="false" disabled>
        <button type="button" data-catalog-toggle aria-label="{label} 목록 열기" tabindex="-1" disabled>▾</button>
      </div>
      <div id="verify-{name}-list" class="verify-combo-list" role="listbox" aria-label="{label} 목록" hidden></div>
      <select name="{name}" hidden disabled></select>
    </div>'''


def picker_markup(*, standalone=False):
    attributes = 'open data-standalone="true"' if standalone else 'hidden'
    standalone_assets = ""
    if standalone:
        markdown = Path(__file__).with_name("LogAnalysisMarkdown.js").read_text(encoding="utf-8")
        standalone_assets = analysis_styles() + "<script>" + markdown + "</script>"
    return (standalone_assets + r'''
<details id="verify-picker" ATTRIBUTES>
<summary>새 검증 실행</summary>
<style>
#verify-picker{color:#e2e8f0;font:13px/1.45 system-ui,sans-serif;margin:4px 0 6px;min-width:0}
#verify-picker[hidden]{display:none}
#verify-picker>summary{cursor:pointer;font-weight:650}
#verify-picker form{display:grid;grid-template-columns:1fr 1fr;gap:7px 10px;margin-top:8px;padding:10px;border:1px solid #475569;border-radius:8px;background:#16243a}
#verify-picker label{display:grid;gap:3px;min-width:0;margin:0}
#verify-picker .verify-wide,#verify-picker [role="status"]{grid-column:1/-1}
#verify-picker .verify-lookup{position:relative;display:grid;gap:4px;min-width:0}
#verify-picker select,#verify-picker input,#verify-picker button,#verify-picker textarea{box-sizing:border-box;width:100%;min-width:0;font:inherit;padding:6px 8px;border:1px solid #64748b;border-radius:5px;background:white;color:#0f172a}
#verify-picker button{grid-column:1/-1;background:#1559bd;color:white;border-color:#70a8ee;cursor:pointer}
#verify-picker .verify-combo{display:grid;grid-template-columns:minmax(0,1fr) 34px;border:1px solid #64748b;border-radius:5px;background:#fff}
#verify-picker .verify-combo:focus-within{outline:2px solid #60a5fa;outline-offset:1px}
#verify-picker .verify-combo input{grid-column:1;border:0;outline:0;border-radius:4px 0 0 4px}
#verify-picker .verify-combo button{grid-column:2;border:0;border-left:1px solid #cbd5e1;border-radius:0 4px 4px 0;background:#f1f5f9;color:#334155}
#verify-picker .verify-combo button:hover:not(:disabled){background:#dbeafe}
#verify-picker .verify-combo-list{position:absolute;z-index:10;top:calc(100% + 3px);left:0;right:0;max-height:230px;overflow:auto;border:1px solid #64748b;border-radius:5px;background:#fff;color:#0f172a;box-shadow:0 8px 16px #0003}
#verify-picker .verify-combo-list>div{padding:6px 8px;overflow-wrap:anywhere}
#verify-picker .verify-combo-list [role="option"]{cursor:pointer}
#verify-picker .verify-combo-list [role="option"]:hover,#verify-picker .verify-combo-list .verify-combo-active{background:#dbeafe}
#verify-picker .verify-combo-list [aria-selected="true"]{font-weight:650}
#verify-picker :disabled{opacity:.6;cursor:default}
#verify-picker [role="status"]{color:#bfdbfe;overflow-wrap:anywhere}
#verify-picker .verify-runs{grid-column:1/-1;margin:0;padding-left:22px;max-height:225px;overflow:auto}
#verify-picker .verify-runs li{padding:5px 0;overflow-wrap:anywhere}
#verify-picker .verify-runs button,#verify-picker [data-single-dashboard]{width:auto;margin-left:8px;padding:2px 8px;background:#334155}
#verify-picker [data-single-dashboard]{justify-self:start;margin-left:0}
#verify-picker [data-queue]{background:#334155}
#verify-picker [data-stop]{background:#7f1d1d;border-color:#fca5a5}
#verify-picker [data-shutdown]{justify-self:end;width:auto;background:#334155}
#verify-picker .verify-settings{grid-column:1/-1;min-width:0;border:1px solid #475569;border-radius:8px;padding:8px 10px}
#verify-picker .verify-settings>summary{cursor:pointer;font-weight:650}
#verify-picker .verify-settings-grid{display:grid;grid-template-columns:repeat(2,minmax(0,1fr));gap:8px;margin-top:8px}
#verify-picker .verify-settings-grid fieldset{min-width:0;margin:0;padding:6px 8px;border:1px solid #475569;border-radius:6px}
#verify-picker .verify-checks{display:flex;flex-wrap:wrap;gap:5px 16px}
#verify-picker .verify-settings label.verify-check{display:flex;flex-direction:row;align-items:center;gap:6px}
#verify-picker .verify-check input{width:auto;margin:0;accent-color:#60a5fa}
#verify-picker [data-panels-required]{color:#fca5a5}
#verify-picker [hidden]{display:none!important}
#verify-picker .verify-history{min-width:0;margin-top:6px}
#verify-picker .verify-history h3{margin:0 0 6px;font-size:15px}
#verify-picker .verify-history-scroll{overflow-x:auto;border:1px solid #475569;border-radius:6px}
#verify-picker .verify-history table{width:100%;border-collapse:collapse;text-align:left;white-space:nowrap}
#verify-picker .verify-history th,#verify-picker .verify-history td{padding:6px 10px;border-bottom:1px solid #334155}
#verify-picker .verify-history th{background:#1e3049;font-weight:650}
#verify-picker .verify-history tr:last-child td{border-bottom:0}
#verify-picker .verify-history td:last-child{text-align:right;font-variant-numeric:tabular-nums}
#verify-picker .verify-history td[data-verdict="VERIFIED"]{color:#86efac}
#verify-picker .verify-history td[data-verdict="COUNTEREXAMPLE"],#verify-picker .verify-history td[data-verdict="오류"]{color:#fca5a5}
#verify-picker .verify-history td[data-verdict="TIMEOUT"],#verify-picker .verify-history td[data-verdict="UNKNOWN"]{color:#fcd34d}
@media(max-width:420px){#verify-picker .verify-settings-grid{grid-template-columns:minmax(0,1fr)}}
#verify-batch-analysis{grid-column:1/-1;min-width:0;border-top:1px solid #475569;margin-top:8px;padding-top:12px}
#verify-batch-analysis[hidden]{display:none}
#verify-batch-analysis>h3{margin:0 0 10px;font-size:17px}
#verify-batch-analysis .llm-batch-tabs{display:flex;gap:4px;margin:0 0 10px}
#verify-batch-analysis .llm-batch-tabs button{width:auto;background:#16243a;border-color:#64748b;color:#cbd5e1}
#verify-batch-analysis .llm-batch-tabs button[aria-selected="true"]{background:#246bd0;border-color:#70a8ee;color:#fff}
#verify-batch-analysis [data-new-analysis]{min-width:0;padding:0;margin:0 0 10px;border:0}
#verify-batch-analysis [data-batch-model]{width:min(100%,260px);margin-bottom:8px}
#verify-batch-analysis [data-batch-effort]{width:min(100%,260px);margin-bottom:8px}
#verify-batch-analysis .llm-status-details{margin:0 0 10px;font-size:12px;color:#cbd5e1}
#verify-batch-analysis .llm-status-details summary{cursor:pointer}
#verify-batch-analysis .llm-status-details[open] summary{margin-bottom:5px}
#verify-batch-analysis .llm-saved-row{display:grid;grid-template-columns:repeat(2,minmax(0,1fr));gap:8px;margin:8px 0}
#verify-batch-analysis .llm-saved-record{grid-column:1/-1}
#verify-batch-analysis .llm-saved-actions{grid-column:1/-1;display:flex;flex-wrap:wrap;align-items:center;gap:8px}
#verify-batch-analysis [data-saved-title]{margin:12px 0 4px;font-size:16px;overflow-wrap:anywhere}
#verify-batch-analysis .llm-question-field{display:grid;gap:4px;font-size:14px}
#verify-batch-analysis textarea{resize:vertical;min-height:128px;width:100%;font-size:14px}
#verify-batch-analysis .llm-action-row{display:flex;align-items:center;flex-wrap:wrap;justify-content:space-between;gap:10px;margin:12px 0 8px}
#verify-batch-analysis .llm-consent{display:flex;flex-direction:row;align-items:flex-start;gap:6px;font-size:14px}
#verify-batch-analysis .llm-consent input{width:auto;margin:3px 0 0}
#verify-batch-analysis .llm-button{width:auto}
#verify-batch-analysis .llm-state{color:#cbd5e1;font-size:14px;overflow-wrap:anywhere}
#verify-batch-analysis .llm-state:empty{display:none}
#verify-batch-analysis [data-report][data-empty="true"]{color:#475569}
</style>
<form>
''' + _catalog_combobox("group", "벤치마크") + _catalog_combobox("model", "신경망") + _catalog_combobox("property", "VNNLIB 속성") + r'''
  <label>검증 엔진<select name="engine" required></select></label>
  <label>제한 시간(초)<input name="timeout" type="number" min="1" max="43200" step="1" value="670" required></label>
  <details class="verify-settings" open>
    <summary>시각화 설정</summary>
    <div class="verify-settings-grid">
      <label class="verify-wide">모드<select name="visualization_mode">
        <option value="both" selected>실시간 + 피드백</option>
        <option value="realtime">실시간</option>
        <option value="feedback">피드백</option>
        <option value="off">시각화 없음</option>
      </select></label>
      <fieldset class="verify-wide" data-visualization-panels>
        <legend>표시할 패널</legend>
        <div class="verify-checks">
          <label class="verify-check"><input type="checkbox" value="dpll-theory" data-panel checked>DPLL / Theory</label>
          <label class="verify-check"><input type="checkbox" value="simplex" data-panel checked>Simplex</label>
          <label class="verify-check"><input type="checkbox" value="relu" data-panel checked>ReLU 분기 이력·탐색 트리</label>
          <label class="verify-check"><input type="checkbox" value="mechanisms" data-panel checked>Mechanisms</label>
        </div>
        <span data-panels-required hidden>시각화 패널을 하나 이상 선택해 주세요.</span>
      </fieldset>
    </div>
  </details>
  <details class="verify-settings">
    <summary>고급 검증 설정</summary>
    <div class="verify-settings-grid">
      <label>Simplex 호출당 최대 반복<input type="number" name="simplex_max_iter" min="1" max="2147483647" step="1" value="1000000" required></label>
      <label>최대 DPLL 라운드<input type="number" name="max_rounds" min="1" max="2147483647" step="1" value="10000" required></label>
      <label>최대 ReLU 재귀 깊이<input type="number" name="max_recursion" min="1" max="2147483647" step="1" value="500" required></label>
      <label>외부 엔진 LP 모드<select name="lp_mode"><option value="core" selected>Core</option><option value="oracle">Oracle</option></select></label>
      <fieldset class="verify-wide">
        <legend>Reluplex 기법</legend>
        <div class="verify-checks">
          <label class="verify-check"><input type="checkbox" value="bound" data-feature checked>경계 유도</label>
          <label class="verify-check"><input type="checkbox" value="backjump" data-feature>충돌 분석·백점프</label>
          <label class="verify-check"><input type="checkbox" value="numeric" data-feature checked>부동소수점 안전장치</label>
          <label class="verify-check"><input type="checkbox" value="underapprox" data-feature>과소근사</label>
        </div>
      </fieldset>
      <label class="verify-wide">과소근사 ε<input type="number" name="under_approx_eps" step="any" value="0.1" required></label>
    </div>
  </details>
  <button type="submit" disabled>검증 시작</button>
  <button type="button" data-queue disabled>대기열에 추가</button>
  <button type="button" data-stop disabled>검증 중단</button>
  <ol class="verify-runs" data-draft aria-label="검증 대기열" hidden></ol>
  <div role="status" aria-live="polite"></div>
  <button type="button" data-single-dashboard hidden>대시보드 열기</button>
  <div class="verify-wide" data-dashboard-error aria-live="polite" hidden></div>
  <ol class="verify-runs" data-runs aria-label="연속 검증 결과" hidden></ol>
  <section class="verify-wide verify-history" aria-labelledby="verify-history-title" HISTORY_VISIBILITY>
    <h3 id="verify-history-title">이번 세션 검증 내역</h3>
    <div class="verify-history-scroll">
      <table><thead><tr><th scope="col">날짜·시간</th><th scope="col">신경망 이름</th><th scope="col">속성</th><th scope="col">검증 결과</th><th scope="col">검증 시간</th></tr></thead>
        <tbody data-history><tr><td colspan="5">아직 실행한 검증이 없습니다.</td></tr></tbody>
      </table>
    </div>
  </section>
  <section id="verify-batch-analysis" aria-labelledby="verify-batch-analysis-title" hidden>
    <h3 id="verify-batch-analysis-title">통합 LLM 분석</h3>
    <div class="llm-batch-tabs" role="tablist" aria-label="통합 분석 보기">
      <button type="button" role="tab" id="verify-question-tab" aria-selected="true" aria-controls="verify-question-panel">질문</button>
      <button type="button" role="tab" id="verify-saved-tab" aria-selected="false" aria-controls="verify-saved-panel" tabindex="-1">저장 결과</button>
    </div>
    <div id="verify-question-panel" role="tabpanel" aria-labelledby="verify-question-tab">
      <fieldset data-new-analysis aria-label="새 통합 분석">
        <label>모델<select data-batch-model disabled><option value="">모델 선택</option></select></label>
        <label>추론 강도<select data-batch-effort disabled><option value="low">낮음</option></select></label>
        <details class="llm-status-details"><summary>모델 사용량</summary>
          ''' + analysis_usage_markup("llm-batch") + r'''
        </details>
        <p class="llm-state" data-followup-source hidden></p>
        <label class="llm-question-field">질문<textarea maxlength="2000" placeholder="비워두면 전체 실행의 공통 패턴과 성능 차이를 분석합니다."></textarea></label>
        <div class="llm-action-row">
          <label class="llm-consent"><input type="checkbox">실행 요약과 이번 세션의 질문·답변을 Google에 전송하는 데 동의합니다</label>
          <button class="llm-button llm-primary" type="button" data-analyze disabled>전체 실행 분석</button>
        </div>
      </fieldset>
      <div class="llm-state" data-analysis-status aria-live="polite">연속 검증을 실행하면 기록이 쌓이는 대로 분석할 수 있습니다.</div>
    </div>
    <div id="verify-saved-panel" role="tabpanel" aria-labelledby="verify-saved-tab" hidden>
      <div class="llm-saved-row">
        <label>신경망<select data-saved-network disabled><option value="">신경망 선택</option></select></label>
        <label>속성<select data-saved-property disabled><option value="">속성 선택</option></select></label>
        <label class="llm-saved-record">날짜·시간·설명<select data-saved-analyses disabled><option value="">저장 결과 선택</option></select></label>
        <div class="llm-saved-actions">
          <button class="llm-button" type="button" data-view-saved disabled>저장 결과 보기</button>
          FOLLOWUP_BUTTON
        </div>
      </div>
      <h4 data-saved-title hidden></h4>
      <div class="llm-state" data-saved-status aria-live="polite"></div>
    </div>
    <div class="llm-report">
      <div class="llm-report-header">
        <span class="llm-report-title">분석 결과</span>
        <span class="llm-result-tag" data-analysis-tag aria-live="polite">대기</span>
        <button class="llm-button llm-copy" type="button" data-copy disabled>원문 복사</button>
      </div>
      <article class="llm-result" data-report data-empty="true" aria-label="통합 분석 보고서">전체 실행 분석을 눌러 결과를 비교하세요.</article>
    </div>
  </section>
  SHUTDOWN_BUTTON
</form>
</details>
<script>
window.setAnalysisEffortOptions = (select, levels) => {
  const previous = select.value;
  const labels = {default:'모델 기본값', minimal:'최소', low:'낮음', medium:'보통', high:'높음', xhigh:'매우 높음'};
  levels ||= ['default'];
  select.replaceChildren(...levels.map(level => new Option(labels[level], level)));
  select.value = levels.includes(previous) ? previous : levels.includes('low') ? 'low' : levels[0];
};
window.bindAnalysisQuotaLink = (link, openQuota) => {
  if (!link || location.protocol === 'file:') return;
  const status = link.parentElement.querySelector('[data-quota-status]');
  let opening = false;
  const open = async event => {
    event.preventDefault();
    if (opening) return;
    opening = true;
    status.textContent = '';
    try {
      await openQuota();
      status.textContent = '기본 브라우저에서 열었습니다.';
    } catch (error) {
      status.textContent = error.message;
    } finally {
      opening = false;
    }
  };
  link.addEventListener('click', open);
  link.addEventListener('auxclick', event => { if (event.button === 1) open(event); });
};
(() => {
  const panel = document.getElementById('verify-picker');
  if (!panel || location.protocol === 'file:') return;
  const form = panel.querySelector('form');
  const fields = form.elements;
  const button = form.querySelector('[type="submit"]');
  const status = form.querySelector('[role="status"]');
  const queueButton = form.querySelector('[data-queue]');
  const stopButton = form.querySelector('[data-stop]');
  const shutdownButton = form.querySelector('[data-shutdown]');
  const draftList = form.querySelector('[data-draft]');
  const runList = form.querySelector('[data-runs]');
  const runHistory = form.querySelector('[data-history]');
  const singleDashboard = form.querySelector('[data-single-dashboard]');
  const dashboardError = form.querySelector('[data-dashboard-error]');
  const analysis = document.getElementById('verify-batch-analysis');
  const analysisTabs = [...analysis.querySelectorAll('[role="tab"]')];
  const analysisPanels = [...analysis.querySelectorAll('[role="tabpanel"]')];
  const batchModel = analysis.querySelector('[data-batch-model]');
  const batchEffort = analysis.querySelector('[data-batch-effort]');
  let batchLevels = {};
  const savedSelect = analysis.querySelector('[data-saved-analyses]');
  const savedNetwork = analysis.querySelector('[data-saved-network]');
  const savedProperty = analysis.querySelector('[data-saved-property]');
  const viewSaved = analysis.querySelector('[data-view-saved]');
  const followupButton = analysis.querySelector('[data-followup-saved]');
  const followupSource = analysis.querySelector('[data-followup-source]');
  const savedStatus = analysis.querySelector('[data-saved-status]');
  const savedTitle = analysis.querySelector('[data-saved-title]');
  const analyzeButton = analysis.querySelector('[data-analyze]');
  const analysisQuestion = analysis.querySelector('textarea');
  const defaultQuestionPlaceholder = analysisQuestion.placeholder;
  const consent = analysis.querySelector('input');
  const analysisStatus = analysis.querySelector('[data-analysis-status]');
  const analysisTag = analysis.querySelector('[data-analysis-tag]');
  const report = analysis.querySelector('[data-report]');
  const copyButton = analysis.querySelector('[data-copy]');
  const panelChecks = [...form.querySelectorAll('[data-panel]')];
  const featureChecks = [...form.querySelectorAll('[data-feature]')];
  const numberSettings = ['simplex_max_iter', 'max_rounds', 'max_recursion'];
  const checkedValues = checks => checks.filter(input => input.checked).map(input => input.value);
  const readSettings = () => ({
    visualization_mode:fields.visualization_mode.value, panels:checkedValues(panelChecks),
    lp_mode:fields.lp_mode.value,
    features:checkedValues(featureChecks),
    under_approx_eps:fields.under_approx_eps.valueAsNumber > 0 ? fields.under_approx_eps.valueAsNumber : 0.1,
    ...Object.fromEntries(numberSettings.map(name => [name, fields[name].valueAsNumber])),
  });
  const initialSettings = readSettings();
  const validSettings = settings => (settings.visualization_mode === 'off' || settings.panels.length > 0)
    && numberSettings.every(name => fields[name].validity.valid && settings[name] > 0)
    && (!settings.features.includes('underapprox') || fields.under_approx_eps.valueAsNumber > 0);
  const endpoint = new URL('api/', location.href);
  let csrf = null, loading = false, running = false, revision = 0;
  let batchId = null, batchComplete = false, analysisRunning = false, analysisAvailable = false;
  let analysisReady = false, analysisSubmitting = false, analysisRevision = 0;
  let batchPollTimer = null, pollingBatch = null;
  let reportSource = '';
  let modelLoading = false, modelLoaded = false, savedLoading = false;
  let followupPending = false;
  let activeAnalysisTab = 0, currentAnalysis = {status:'idle'}, savedAnalysis = null;
  let savedGroups = new Map();
  let singleId = null;
  let activeRunId = null, stopPending = false, stopping = false;
  let launcherClosed = false;
  let historyLoading = false, historyTimer = null;
  const historyRows = new Map();
  const historyDate = new Intl.DateTimeFormat('sv-SE', {dateStyle:'short', timeStyle:'medium'});
  let runListSignature = '';
  let maxRuns = 50;
  const draft = [];
  let saving = Promise.resolve();
  const options = (field, choices, placeholder) => {
    field.replaceChildren(...(placeholder ? [new Option(placeholder, '')] : []),
      ...choices.map(item => new Option(item.label, item.id)));
  };
  const enabled = () => {
    if (launcherClosed) return;
    const settings = readSettings();
    form.querySelector('[data-visualization-panels]').disabled = settings.visualization_mode === 'off';
    form.querySelector('[data-panels-required]').hidden = settings.visualization_mode === 'off' || settings.panels.length > 0;
    fields.lp_mode.disabled = fields.engine.value === 'native';
    fields.under_approx_eps.disabled = !settings.features.includes('underapprox');
    const valid = Boolean(fields.model.value && fields.property.value && fields.engine.value
      && fields.timeout.validity.valid && validSettings(settings));
    button.disabled = running || analysisRunning || followupPending || !csrf || (!draft.length && !valid);
    button.textContent = draft.length ? '연속 검증 시작 (' + draft.length + ')' : '검증 시작';
    queueButton.disabled = running || !csrf || !valid || draft.length >= maxRuns;
    stopButton.disabled = !activeRunId || !running || stopping || stopPending;
    batchModel.disabled = !analysisAvailable || analysisRunning;
    batchEffort.disabled = batchModel.disabled || !batchModel.value;
    analyzeButton.disabled = !analysisAvailable || !analysisReady || analysisRunning || analysisSubmitting || followupPending
      || !consent.checked || !batchModel.value;
    analyzeButton.textContent = !followupSource.hidden ? '이어서 질문하기' : batchComplete ? '전체 실행 분석' : '현재까지 분석';
    viewSaved.disabled = savedLoading || !savedSelect.value;
    if (followupButton) followupButton.disabled = !csrf || !savedSelect.value || running || analysisRunning || followupPending;
    copyButton.disabled = !reportSource || activeAnalysisTab === 0 && analysisRunning;
  };
  function setAnalysisTag(state) {
    const labels = {idle:'대기', running:'분석 중', complete:'완료', incomplete:'일부 답변', error:'오류'};
    analysisTag.dataset.status = state;
    analysisTag.textContent = labels[state] || labels.idle;
  }
  function renderAnalysis(result, emptyText = '전체 실행 분석을 눌러 결과를 비교하세요.') {
    setAnalysisTag(result.status);
    const source = result.report || '';
    report.dataset.empty = String(!source);
    if (reportSource !== source || !source && report.textContent !== emptyText) {
      reportSource = source;
      copyButton.textContent = '원문 복사';
      if (source && window.renderLogAnalysisMarkdown) window.renderLogAnalysisMarkdown(report, source);
      else report.textContent = source || emptyText;
    }
  }
  function renderAnalysisTab() {
    if (activeAnalysisTab === 1) {
      renderAnalysis(savedAnalysis || {status:'idle'}, '저장된 LLM 답변을 선택하세요.');
    } else {
      renderAnalysis(currentAnalysis, analysisReady ? '분석 버튼을 눌러 기록을 비교하세요.'
        : '분석 결과가 여기에 표시됩니다.');
    }
  }
  function selectAnalysisTab(index) {
    activeAnalysisTab = index;
    analysisTabs.forEach((tab, i) => {
      tab.setAttribute('aria-selected', String(i === index));
      tab.tabIndex = i === index ? 0 : -1;
      analysisPanels[i].hidden = i !== index;
    });
    if (index === 1) loadSavedAnalyses();
    renderAnalysisTab();
    enabled();
  }
  analysisTabs.forEach((tab, index) => {
    tab.addEventListener('click', () => selectAnalysisTab(index));
    tab.addEventListener('keydown', event => {
      if (!['ArrowLeft', 'ArrowRight', 'Home', 'End'].includes(event.key)) return;
      event.preventDefault();
      const next = event.key === 'Home' ? 0 : event.key === 'End' ? analysisTabs.length - 1 : 1 - index;
      selectAnalysisTab(next);
      analysisTabs[next].focus();
    });
  });
  function renderDraft() {
    draftList.hidden = !draft.length;
    draftList.replaceChildren(...draft.map((entry, index) => {
      const row = document.createElement('li');
      row.textContent = entry.label;
      const remove = document.createElement('button');
      remove.type = 'button';
      remove.textContent = '삭제';
      remove.setAttribute('aria-label', (index + 1) + '번 대기 항목 삭제');
      remove.addEventListener('click', () => {
        if (running) return;
        draft.splice(index, 1);
        renderDraft();
      });
      row.append(remove);
      return row;
    }));
    enabled();
  }
  const restore = (field, value) => {
    if ([...field.options].some(option => option.value === value)) field.value = value;
  };
  const normalizeSearch = value => value.normalize('NFKC').toLowerCase().replace(/[\s_.-]+/g, '');
  const catalogPickers = {
    group:{placeholder:'벤치마크 선택'},
    model:{placeholder:'신경망 선택'},
    property:{placeholder:'속성 선택'},
  };
  function closeCatalog(picker) {
    picker.list.hidden = true;
    picker.search.setAttribute('aria-expanded', 'false');
    picker.search.removeAttribute('aria-activedescendant');
    picker.search.value = picker.field.value ? picker.field.selectedOptions[0].text : '';
  }
  function showCatalog(picker, search = '') {
    for (const other of Object.values(catalogPickers)) {
      if (other !== picker && !other.list.hidden) closeCatalog(other);
    }
    const query = normalizeSearch(search);
    picker.filtered = Boolean(query);
    picker.matches = picker.choices.filter(item => item.searchKey.includes(query));
    picker.active = -1;
    picker.search.removeAttribute('aria-activedescendant');
    picker.list.replaceChildren(...picker.matches.map((item, index) => {
      const row = document.createElement('div');
      row.id = picker.list.id + '-option-' + index;
      row.dataset.index = index;
      row.setAttribute('role', 'option');
      row.setAttribute('aria-selected', String(item.id === picker.field.value));
      row.textContent = item.label;
      return row;
    }));
    if (!picker.matches.length) {
      const empty = document.createElement('div');
      empty.textContent = '검색 결과 없음';
      picker.list.append(empty);
    }
    picker.list.scrollTop = 0;
    picker.list.hidden = false;
    picker.search.setAttribute('aria-expanded', 'true');
  }
  function chooseCatalog(picker, index) {
    const item = picker.matches[index];
    if (!item) return;
    const previous = picker.field.value;
    picker.field.value = item.id;
    closeCatalog(picker);
    if (previous !== item.id) picker.field.dispatchEvent(new Event('change', {bubbles:true}));
  }
  function moveCatalogSelection(picker, direction) {
    if (picker.list.hidden) showCatalog(picker);
    const count = picker.matches.length;
    if (!count) return;
    picker.list.children[picker.active]?.classList.remove('verify-combo-active');
    const initial = direction > 0 ? 0 : count - 1;
    picker.active = picker.active < 0 ? initial : (picker.active + direction + count) % count;
    const row = picker.list.children[picker.active];
    row.classList.add('verify-combo-active');
    picker.search.setAttribute('aria-activedescendant', row.id);
    row.scrollIntoView({block:'nearest'});
  }
  function setCatalog(name, choices, selected = '') {
    const picker = catalogPickers[name];
    picker.choices = choices.map(item => ({...item, searchKey:normalizeSearch(item.label)}));
    picker.search.disabled = picker.toggle.disabled = !choices.length;
    options(picker.field, choices, picker.placeholder);
    restore(picker.field, selected);
    closeCatalog(picker);
  }
  for (const [name, picker] of Object.entries(catalogPickers)) {
    picker.field = fields[name];
    picker.search = form.querySelector('[data-catalog-search="' + name + '"]');
    const container = picker.search.closest('.verify-lookup');
    picker.list = container.querySelector('[role="listbox"]');
    picker.toggle = container.querySelector('[data-catalog-toggle]');
    picker.choices = [];
    picker.search.addEventListener('input', () => showCatalog(picker, picker.search.value));
    picker.search.addEventListener('click', () => {
      if (picker.list.hidden) showCatalog(picker);
    });
    picker.toggle.addEventListener('pointerdown', event => event.preventDefault());
    picker.toggle.addEventListener('click', () => {
      picker.search.focus({preventScroll:true});
      if (picker.list.hidden || picker.filtered) showCatalog(picker);
      else closeCatalog(picker);
    });
    picker.list.addEventListener('mousedown', event => event.preventDefault());
    picker.list.addEventListener('click', event => {
      const row = event.target.closest('[role="option"]');
      if (row) chooseCatalog(picker, Number(row.dataset.index));
    });
    container.addEventListener('focusout', event => {
      if (!container.contains(event.relatedTarget)) closeCatalog(picker);
    });
    document.addEventListener('pointerdown', event => {
      if (!picker.list.hidden && !container.contains(event.target)) closeCatalog(picker);
    });
    picker.search.addEventListener('keydown', event => {
      if (event.isComposing || event.keyCode === 229) return;
      if (event.key === 'Escape') {
        if (!picker.list.hidden) event.preventDefault();
        closeCatalog(picker);
      } else if (event.key === 'ArrowDown' || event.key === 'ArrowUp') {
        event.preventDefault();
        moveCatalogSelection(picker, event.key === 'ArrowDown' ? 1 : -1);
      } else if (event.key === 'Enter') {
        event.preventDefault();
        if (picker.list.hidden) showCatalog(picker);
        else chooseCatalog(picker, picker.active < 0 && picker.matches.length === 1 ? 0 : picker.active);
      }
    });
    picker.field.addEventListener('change', () => closeCatalog(picker));
  }
  function selection() {
    const settings = Object.fromEntries(Object.entries(readSettings()).filter(([key, value]) =>
      JSON.stringify(value) !== JSON.stringify(initialSettings[key])));
    return {group:fields.group.value, model:fields.model.value, property:fields.property.value,
      engine:fields.engine.value, timeout:fields.timeout.valueAsNumber,
      ...(Object.keys(settings).length ? {settings} : {})};
  }
  async function api(route, body) {
    const response = await fetch(new URL(route, endpoint), {cache:'no-store',
      signal: AbortSignal.timeout(10000),
      ...(body ? {method:'POST', headers:{'Content-Type':'application/json','X-Analysis-Token':csrf},
        body:JSON.stringify(body), keepalive:route==='verify-preferences'} : {})});
    const data = await response.json();
    if (!response.ok) throw Error(data.error || response.status);
    return data;
  }
  async function refreshRunHistory() {
    if (!runHistory || !panel.dataset.standalone || launcherClosed || historyLoading) return;
    clearTimeout(historyTimer);
    historyLoading = true;
    let active = running;
    try {
      const {runs} = await api('verify-history');
      if (launcherClosed) return;
      if (runs.length && !historyRows.size) runHistory.replaceChildren();
      active = runs.some(run => run.status === 'running');
      for (const run of runs) {
        let row = historyRows.get(run.number);
        if (!row) {
          row = runHistory.insertRow();
          for (let i = 0; i < 5; i++) row.insertCell();
          row.cells[0].textContent = historyDate.format(new Date(run.started_at));
          row.cells[1].textContent = run.model;
          row.cells[2].textContent = run.property;
          historyRows.set(run.number, row);
        }
        const result = run.result || ({running:'실행 중', cancelled:'중단', error:'오류'}[run.status] || '—');
        row.cells[3].textContent = result;
        row.cells[3].dataset.verdict = result;
        row.cells[4].textContent = run.elapsed_seconds.toFixed(1) + '초';
      }
    } catch {
      // A later status update or focus retries the local history request.
    } finally {
      historyLoading = false;
      if (!launcherClosed && (active || running)) historyTimer = setTimeout(refreshRunHistory, 1000);
    }
  }
  window.addEventListener('focus', () => { if (csrf) refreshRunHistory(); });
  window.bindAnalysisQuotaLink(document.getElementById('llm-batch-quota-link'), () => {
    if (!csrf) throw Error('검증기 연결이 준비되면 다시 눌러 주세요.');
    return api('open-ai-studio-quota', {});
  });
  async function loadModels() {
    if (!analysisAvailable || modelLoaded || modelLoading) return;
    modelLoading = true;
    options(batchModel, [], '모델 불러오는 중…');
    try {
      const {models, default_model:preferred, thinking_levels:levels} = await api('verify-models');
      batchLevels = levels || {};
      const hasPreferred = models.includes(preferred);
      options(batchModel, models.map(id => ({id,label:id})),
        hasPreferred ? null : models.length ? '모델 선택' : '사용 가능한 모델 없음');
      if (hasPreferred) batchModel.value = preferred;
      setAnalysisEffortOptions(batchEffort, batchLevels[batchModel.value]);
      modelLoaded = true;
    } catch {
      options(batchModel, [], '모델 조회 실패 · 클릭하여 재시도');
    } finally {
      modelLoading = false;
      enabled();
    }
  }
  batchModel.addEventListener('pointerdown', loadModels);
  batchModel.addEventListener('focus', loadModels);
  batchModel.addEventListener('change', () => {
    setAnalysisEffortOptions(batchEffort, batchLevels[batchModel.value]);
    enabled();
  });
  const savedDate = item => new Date(item.saved_at).toLocaleString('ko-KR');
  const sourceLabel = ({network, property}) => network && property ? network + ' (' + property + ')'
    : network || property || '';
  const batchSources = item => (item.sources || []).map(sourceLabel).filter(Boolean).join(', ');
  function addSavedEntry(groups, source, entry) {
    const network = source.network?.replace(/\.onnx\.gz$/i, '.onnx') || '신경망 정보 없음';
    const name = source.property?.replace(/\.vnnlib(?:\.gz)?$/i, '') || '';
    const number = /^prop[_ -]?(\d+)$/i.exec(name);
    const property = number ? 'prop_' + number[1] : name || '속성 정보 없음';
    if (!groups.has(network)) groups.set(network, new Map());
    const properties = groups.get(network);
    if (!properties.has(property)) properties.set(property, new Map());
    properties.get(property).set(entry.id, entry);
  }
  function setSavedOptions(field, entries, empty) {
    const selected = field.value;
    options(field, entries, entries.length ? null : empty);
    restore(field, selected);
    field.disabled = !entries.length;
  }
  const sortedChoices = names => [...names].sort((a, b) => a.localeCompare(b, 'ko', {numeric:true}))
    .map(name => ({id:name, label:name}));
  function renderSavedEntries() {
    const properties = savedGroups.get(savedNetwork.value);
    setSavedOptions(savedProperty, sortedChoices(properties?.keys() || []), '속성 없음');
    const entries = [...(properties?.get(savedProperty.value)?.values() || [])];
    entries.sort((a, b) => Date.parse(b.saved_at) - Date.parse(a.saved_at));
    setSavedOptions(savedSelect, entries.map(item => ({
      id:item.id, label:[savedDate(item), item.title].filter(Boolean).join(' · '),
    })), '저장된 결과 없음');
    enabled();
  }
  function clearSavedReport() {
    savedAnalysis = null;
    savedTitle.hidden = true;
    savedStatus.textContent = '';
    if (activeAnalysisTab === 1) renderAnalysisTab();
  }
  for (const selector of [savedNetwork, savedProperty]) {
    selector.addEventListener('change', () => {
      clearSavedReport();
      renderSavedEntries();
    });
  }
  savedSelect.addEventListener('change', clearSavedReport);
  async function loadSavedAnalyses() {
    try {
      const {analyses, run_analyses:individual = []} = await api('verify-saved-analyses');
      if (launcherClosed) return;
      savedGroups = new Map();
      for (const item of analyses) {
        for (const source of item.sources?.length ? item.sources : [{}]) {
          addSavedEntry(savedGroups, source, item);
        }
      }
      for (const item of individual) {
        addSavedEntry(savedGroups, item, {...item, id:'run:' + item.id});
      }
      setSavedOptions(savedNetwork, sortedChoices(savedGroups.keys()), '신경망 없음');
      renderSavedEntries();
      if (!savedAnalysis) savedStatus.textContent = '';
    } catch (error) {
      if (!launcherClosed) savedStatus.textContent = '저장 목록 조회 실패 · ' + error.message;
    }
    enabled();
  }
  viewSaved.addEventListener('click', async () => {
    if (viewSaved.disabled) return;
    savedLoading = true;
    enabled();
    try {
      const selection = savedSelect.value;
      const isIndividual = selection.startsWith('run:');
      const parameter = isIndividual ? 'run_id=' + encodeURIComponent(selection.slice(4))
        : 'id=' + encodeURIComponent(selection);
      const {analysis:result} = await api('verify-saved-analyses?' + parameter);
      if (launcherClosed || savedSelect.value !== selection) return;
      savedAnalysis = result;
      savedTitle.textContent = result.title || '';
      savedTitle.hidden = !result.title;
      savedStatus.textContent = [savedDate(result), isIndividual ? sourceLabel(result) : batchSources(result),
        result.scope, result.model].filter(Boolean).join(' · ');
      if (activeAnalysisTab === 1) renderAnalysisTab();
    } catch (error) {
      if (!launcherClosed) savedStatus.textContent = error.message;
    } finally {
      savedLoading = false;
      enabled();
    }
  });
  followupButton?.addEventListener('click', async () => {
    if (followupButton.disabled) return;
    followupPending = true;
    enabled();
    try {
      const {id} = await api('verify-followup', {id:savedSelect.value});
      if (launcherClosed) return;
      clearTimeout(batchPollTimer);
      batchId = activeRunId = id;
      analysisRevision++;
      analysisReady = false;
      currentAnalysis = {status:'idle'};
      analysisQuestion.value = '';
      selectAnalysisTab(0);
      analysisQuestion.focus();
      await pollBatch(id);
    } catch (error) {
      if (!launcherClosed) savedStatus.textContent = error.message;
    } finally {
      followupPending = false;
      enabled();
    }
  });
  async function openDashboard(id, number) {
    dashboardError.hidden = true;
    try {
      await api('verify-open-dashboard', {id, number});
    } catch (error) {
      dashboardError.textContent = error.message;
      dashboardError.hidden = false;
    }
  }
  singleDashboard.addEventListener('click', () => openDashboard(singleId));
  stopButton.addEventListener('click', async () => {
    if (stopButton.disabled) return;
    const id = activeRunId;
    stopPending = true;
    enabled();
    dashboardError.hidden = true;
    try {
      const response = await api('verify-stop', {id});
      if (response.status === 'stopping' && activeRunId === id && running) {
        stopping = true;
        status.textContent = '검증을 중단하고 있습니다.';
      }
    } catch (error) {
      if (activeRunId === id) {
        dashboardError.textContent = error.message;
        dashboardError.hidden = false;
      }
    } finally {
      stopPending = false;
      enabled();
    }
  });
  function saveSelection() {
    if (!csrf || !fields.timeout.validity.valid || !fields.engine.value || !validSettings(readSettings())) return;
    const data = selection();
    saving = saving.then(() => api('verify-preferences', data)).catch(error => {
      if (!running) status.textContent = '선택을 저장하지 못했습니다. ' + error.message;
    });
  }
  async function setup() {
    if (csrf || loading || panel.hidden || !panel.open) return;
    loading = true;
    try {
      const data = await api('verify-catalog');
      csrf = data.csrf;
      refreshRunHistory();
      if (shutdownButton) shutdownButton.disabled = false;
      analysisAvailable = Boolean(data.batch_analysis_available);
      analysis.hidden = false;
      renderAnalysisTab();
      maxRuns = data.max_batch_runs || maxRuns;
      setCatalog('group', data.groups, data.preferences?.group);
      const labels = {glpk:'GLPK',native:'자체 Simplex',z3:'Z3'};
      options(fields.engine, data.engines.map(id => ({id,label:labels[id]})));
      status.textContent = '';
      const previous = data.preferences || {};
      restore(fields.engine, previous.engine);
      if (Number.isInteger(previous.timeout)) fields.timeout.value = previous.timeout;
      const settings = previous.settings || {};
      restore(fields.visualization_mode, settings.visualization_mode);
      restore(fields.lp_mode, settings.lp_mode);
      if (Array.isArray(settings.panels)) panelChecks.forEach(input => {input.checked = settings.panels.includes(input.value);});
      if (Array.isArray(settings.features)) featureChecks.forEach(input => {input.checked = settings.features.includes(input.value);});
      [...numberSettings, 'under_approx_eps'].forEach(name => {
        if (Number.isFinite(settings[name])) fields[name].value = settings[name];
      });
      enabled();
      if (fields.group.value) await loadGroup(previous);
      if (data.single_id) {
        singleId = activeRunId = data.single_id;
        running = true;
        enabled();
        pollSingle(singleId);
      } else if (data.batch_id) {
        batchId = data.batch_id;
        activeRunId = batchId;
        running = true;
        enabled();
        pollBatch(batchId);
      }
    } catch (error) { status.textContent = error.message; }
    finally { loading = false; }
  }
  panel.addEventListener('toggle', setup);
  if (panel.dataset.standalone) setup();
  async function loadGroup(previous = null) {
    const version = ++revision;
    const group = fields.group.value;
    fields.model.disabled = fields.property.disabled = true;
    setCatalog('model', []);
    setCatalog('property', []);
    enabled();
    if (!previous) saveSelection();
    if (!group) return;
    try {
      const data = await api('verify-catalog?group=' + encodeURIComponent(group));
      if (version !== revision) return;
      setCatalog('model', data.models, previous?.model);
      setCatalog('property', data.properties, previous?.property);
      fields.model.disabled = !data.models.length;
      fields.property.disabled = !data.properties.length;
      status.textContent = data.models.length && data.properties.length ? '' : '이 목록에 사용할 ONNX 또는 VNNLIB 파일이 없습니다.';
      enabled();
    } catch (error) { if (version === revision) status.textContent = error.message; }
  }
  fields.group.addEventListener('change', () => loadGroup());
  form.addEventListener('change', event => {
    enabled();
    if (event.target !== fields.group && !event.target.hasAttribute('data-catalog-search')
        && !analysis.contains(event.target)) saveSelection();
  });
  queueButton.addEventListener('click', () => {
    if (queueButton.disabled) return;
    const choice = selection();
    draft.push({choice, label: [fields.model.selectedOptions[0].text, fields.property.selectedOptions[0].text,
      fields.engine.selectedOptions[0].text, choice.timeout + '초',
      fields.visualization_mode.selectedOptions[0].text].join(' · ')});
    renderDraft();
  });
  const runLabels = {waiting:'대기', running:'실행 중', complete:'완료', error:'실행 실패', cancelled:'중단됨'};
  async function pollBatch(id) {
    if (launcherClosed || pollingBatch === id) return;
    clearTimeout(batchPollTimer);
    pollingBatch = id;
    const revisionAtRequest = analysisRevision;
    const updatingAnalysis = !analysisSubmitting;
    let retry = false;
    try {
      const state = await api('verify-batch?id=' + encodeURIComponent(id));
      if (batchId !== id || launcherClosed) return;
      runList.hidden = !state.runs.length;
      followupSource.hidden = !state.followup_source;
      followupSource.textContent = state.followup_source ? '이어서 질문 · ' + state.followup_source.title : '';
      analysisQuestion.placeholder = state.followup_source ? '이전 답변에서 이어서 궁금한 점을 입력하세요.' : defaultQuestionPlaceholder;
      const signature = JSON.stringify([id, state.status, state.runs.map(run =>
        [run.number, run.status, run.phase, run.verdict, run.error, run.dashboard_ready])]);
      if (signature !== runListSignature) {
        runListSignature = signature;
        runList.replaceChildren(...state.runs.map(run => {
          const row = document.createElement('li');
          const phase = run.status === 'running' && state.status === 'stopping' ? '중단 중'
            : run.status === 'running' && run.phase !== 'running' ? '실행 준비 중' : runLabels[run.status];
          row.textContent = [run.model, run.property, run.engine,
            run.verdict || phase, run.error].filter(Boolean).join(' · ');
          if (run.dashboard_ready) {
            const open = document.createElement('button');
            open.type = 'button';
            open.textContent = '대시보드 열기';
            open.addEventListener('click', () => openDashboard(id, run.number));
            row.append(open);
          }
          return row;
        }));
      }
      stopping = state.status === 'stopping';
      running = state.status === 'running' || stopping;
      batchComplete = state.status === 'complete' || state.status === 'cancelled';
      analysisReady = state.analysis_ready;
      status.textContent = stopping ? '검증을 중단하고 있습니다.' : running
        ? '대기열 순서대로 검증 중입니다.'
        : !state.runs.length && state.followup_source ? ''
        : (state.status === 'cancelled' ? '연속 검증 중단' : '연속 검증 종료')
          + ' · ' + state.runs.length + '건' + (state.save_error ? ' · ' + state.save_error : '');
      if (analysisReady) loadModels();
      if (updatingAnalysis && !analysisSubmitting && revisionAtRequest === analysisRevision) {
        analysisRunning = state.analysis.status === 'running';
        currentAnalysis = state.analysis;
        if (!state.followup_source && state.status === 'cancelled' && state.analysis.question && !analysisQuestion.value) {
          analysisQuestion.value = state.analysis.question;
        }
        analysisStatus.textContent = !analysisAvailable ? 'Gemini API 키를 설정하고 이 화면을 다시 열어 주세요.'
          : analysisRunning ? '요청 시점까지의 기록을 분석하고 있습니다.'
          : state.analysis.error || [
            state.analysis.model, state.analysis.partial && state.analysis.snapshot_at
              ? '부분 기록 · ' + savedDate({saved_at:state.analysis.snapshot_at}) : '',
            state.status === 'cancelled' && state.analysis.status === 'complete' && !state.analysis.snapshot_at
              ? '이전 답변 · 다시 분석하면 저장된 중단 로그를 읽습니다.' : '',
            !analysisReady ? (running ? '분석할 검증 기록을 기다리고 있습니다.' : '분석할 검증 기록이 없습니다.') : '',
          ].filter(Boolean).join(' · ');
        if (activeAnalysisTab === 0) renderAnalysisTab();
        if (activeAnalysisTab === 1 && state.analysis.status === 'complete') loadSavedAnalyses();
      }
      retry = running || analysisRunning;
    } catch (error) {
      if (batchId !== id || launcherClosed) return;
      status.textContent = '상태 조회 실패 · ' + error.message;
      retry = running || analysisRunning;
    } finally {
      if (pollingBatch === id) pollingBatch = null;
    }
    enabled();
    if (retry) batchPollTimer = setTimeout(() => pollBatch(id), 1500);
  }
  analyzeButton.addEventListener('click', async () => {
    if (analyzeButton.disabled) return;
    analysisSubmitting = true;
    analysisRevision++;
    analysisRunning = true;
    currentAnalysis = {status:'running'};
    enabled();
    renderAnalysisTab();
    analysisStatus.textContent = '요청 시점까지의 기록을 분석하고 있습니다.';
    const model = batchModel.value;
    try {
      await api('verify-batch-analysis', {id:batchId, consent:consent.checked,
        question:analysisQuestion.value, model, thinking_level:batchEffort.value});
    } catch (error) {
      currentAnalysis = {status:'error'};
      if (activeAnalysisTab === 0) renderAnalysisTab();
      analysisStatus.textContent = error.message;
      analysisRunning = false;
    } finally {
      analysisSubmitting = false;
      enabled();
      if (analysisRunning) pollBatch(batchId);
    }
  });
  copyButton.addEventListener('click', async () => {
    if (!reportSource) return;
    try {
      await navigator.clipboard.writeText(reportSource);
      copyButton.textContent = '복사됨';
    } catch {
      copyButton.textContent = '복사 실패';
    }
  });
  async function pollSingle(id, note = '') {
    if (launcherClosed) return;
    try {
      const state = await api('verify-job?id=' + encodeURIComponent(id));
      if (launcherClosed) return;
      singleDashboard.hidden = !state.dashboard_ready;
      stopping = state.status === 'stopping';
      running = state.status === 'running' || stopping;
      status.textContent = running ? (stopping ? '검증을 중단하고 있습니다.'
        : state.phase === 'running' ? '검증 실행 중입니다.' : '검증을 준비하고 있습니다.') + note
        : state.status === 'cancelled' ? '검증 중단됨'
        : state.status === 'complete' ? '검증 종료 · ' + state.verdict : (state.error || '검증 실행 실패');
    } catch (error) {
      if (launcherClosed) return;
      status.textContent = '상태 조회 실패 · ' + error.message;
    }
    enabled();
    if (running) setTimeout(() => pollSingle(id, note), 1500);
  }
  form.addEventListener('submit', async event => {
    event.preventDefault();
    if (button.disabled) return;
    running = true;
    activeRunId = null;
    stopping = false;
    singleDashboard.hidden = true;
    dashboardError.hidden = true;
    enabled();
    status.textContent = '검증을 시작하는 중입니다.';
    try {
      const data = selection();
      await saving;
      if (launcherClosed) return;
      if (draft.length) {
        const job = await api('verify-batch', {runs:draft.map(entry => entry.choice)});
        batchId = job.id;
        activeRunId = batchId;
        batchComplete = false;
        analysisReady = false;
        currentAnalysis = {status:'idle'};
        analysisStatus.textContent = '분석할 검증 기록을 기다리고 있습니다.';
        selectAnalysisTab(0);
        draft.length = 0;
        renderDraft();
        pollBatch(batchId);
        return;
      }
      const job = await api('verify-start', data);
      singleId = job.id;
      activeRunId = singleId;
      enabled();
      const note = job.preferences_saved === false ? ' 마지막 선택은 저장되지 않았습니다.' : '';
      pollSingle(singleId, note);
    } catch (error) {
      if (launcherClosed) return;
      status.textContent = error.message;
      running = false;
      enabled();
    } finally {
      refreshRunHistory();
    }
  });
  shutdownButton?.addEventListener('click', async () => {
    shutdownButton.disabled = true;
    shutdownButton.textContent = '종료 중…';
    dashboardError.hidden = true;
    try {
      await api('verify-shutdown', {});
      launcherClosed = true;
      clearTimeout(historyTimer);
      for (const field of fields) field.disabled = true;
      shutdownButton.textContent = '검증기 종료';
      status.textContent = '검증기가 종료되었습니다.';
    } catch (error) {
      dashboardError.textContent = error.message;
      dashboardError.hidden = false;
      shutdownButton.textContent = '검증기 종료';
      shutdownButton.disabled = false;
    }
  });
  form.noValidate = true;
})();
</script>
''').replace('ATTRIBUTES', attributes, 1).replace('HISTORY_VISIBILITY', '' if standalone else 'hidden', 1).replace(
        'FOLLOWUP_BUTTON',
        '<button class="llm-button" type="button" data-followup-saved disabled>해당 내용 질문하기</button>'
        if standalone else '',
        1,
    ).replace(
        'SHUTDOWN_BUTTON',
        '<button type="button" data-shutdown disabled>검증기 종료</button>' if standalone else '',
        1,
    )
