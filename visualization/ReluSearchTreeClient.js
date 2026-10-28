(() => {
  'use strict';
  const INITIAL_DATA = __DATA__;
  const SNAPSHOT_TYPE = 'relu-search-tree-snapshot';
  if (new URLSearchParams(location.search).get('tree-snapshot') === '1') {
    window.parent.postMessage({type: SNAPSHOT_TYPE, data: INITIAL_DATA}, '*');
    return;
  }

  const STATE_PREFIX='relu-search-tree-ui:';
  const $ = selector => document.querySelector(selector);
  const stateKey = location.pathname;
  let savedState = {};
  try {
    if (window.name.startsWith(STATE_PREFIX)) {
      const value = JSON.parse(window.name.slice(STATE_PREFIX.length));
      if (!value.path || value.path === stateKey) savedState = value;
    }
  } catch (_error) {}
  const legacyFilters = {SAT_PROPAGATED: 'CLOSED_SAT', UNSAT_PROPAGATED: 'CLOSED_UNSAT', UNSAT_SKIPPED: 'CLOSED_UNSAT', SAT: 'FINAL', UNSAT: 'FINAL', UNKNOWN: 'FINAL', TIMEOUT: 'FINAL'};
  savedState.status = legacyFilters[savedState.status] || savedState.status;
  const statusOptions = [...$('#statusFilter').options].map(option => option.value);
  if (statusOptions.includes(savedState.status)) $('#statusFilter').value = savedState.status;
  if (['completion', 'visit', 'duration'].includes(savedState.timelineOrder)) $('#timelineOrder').value = savedState.timelineOrder;
  $('#layerFilter').value = savedState.layer || '';
  $('#neuronFilter').value = savedState.neuron || '';
  if (['additional', 'all', 'initial'].includes(savedState.stableScope)) $('#stableScope').value = savedState.stableScope;
  const savedStableOrder = savedState.stableOrder === 'source' ? 'node' : savedState.stableOrder;
  if (['newest', 'neuron', 'node'].includes(savedStableOrder)) $('#stableOrder').value = savedStableOrder;
  const reportPanels = [
    {key: 'timeline', element: $('#timelinePanel'), list: $('#timeline'), render: renderTimeline, scroll: Number(savedState.timelineScroll) || 0},
    {key: 'stable', element: $('#stablePanel'), list: $('#stableList'), render: renderStableRecords, scroll: Number(savedState.stableScroll) || 0},
  ];

  const expandedIds = new Set(savedState.expandedIds || []);
  const views = new Map();
  const rootViews = new Map();
  const timelineViews = new Map();
  const stableViews = new Map();
  const TIMELINE_PAGE_SIZE = 50;
  let timelinePage = Math.max(0, Number(savedState.timelinePage) || 0);
  let stablePage = Math.max(0, Number(savedState.stablePage) || 0);
  let DATA;
  let ALL = [];
  let NODE_BY_ID = new Map();
  const BRANCH_GROWTH = new Map();
  let latestNodeId = null;
  const finalPathIds = new Set();
  const recentPathIds = new Set();
  let recentTargetId = null;
  let trackLatestPath = savedState.trackLatestPath === true;
  let MATCHED = new Map();
  let renderedFlat = null;
  let TIMELINE = [];
  let snapshotFrame = null;
  let snapshotPending = false;
  let standaloneTimer = null;
  let lastStatus = null;

  function statusClass(status) {
    return ['SAT', 'UNSAT', 'CLOSED', 'UNKNOWN', 'TIMEOUT', 'RUNNING'].includes(status) ? status : 'OTHER';
  }
  function branchRoleLabel(node) {
    if (node.satRole) return node.satRole === 'found' ? 'SAT 발견' : 'SAT 전달';
    if (node.unsatRole) return {found: 'UNSAT 발견', propagated: 'UNSAT 전달', skipped: 'UNSAT 생략'}[node.unsatRole];
    return '';
  }
  function isFinalVerdict(node) { return node.result === 'SAT' || node.result === 'UNSAT'; }
  function isClosedResult(node) {
    return node.result === 'CLOSED' && ((node.branchResult === 'SAT' && node.satRole !== 'found') || (node.branchResult === 'UNSAT' && node.unsatRole !== 'found'));
  }
  function resultLabel(node) {
    if (isFinalVerdict(node)) return node.result;
    if (isClosedResult(node)) return `CLOSED (${node.branchResult})`;
    return branchRoleLabel(node) || (node.result === 'CLOSED' ? `CLOSED (${node.branchResult})` : node.result);
  }
  function nodeStatusClass(node) {
    if (isFinalVerdict(node)) return statusClass(node.result);
    if (isClosedResult(node)) return 'CLOSED';
    if (node.satRole === 'found') return 'SAT_FOUND';
    if (node.unsatRole === 'found') return 'UNSAT_FOUND';
    return statusClass(node.result === 'CLOSED' ? node.branchResult : node.result);
  }
  function neuronLabel(node) { return node.layer == null ? node.variable : `L${node.layer}:N${node.neuron}`; }
  function sourceLabel(node) { return `${neuronLabel(node)} (${node.phase}) · 방문 #${node.order}`; }
  function tip(node) {
    const lines = [
      `노드 #${node.id} · Split #${node.splitNumber ?? '?'}`, `${neuronLabel(node)} (${node.phase})`,
      `결과: ${resultLabel(node)}${node.reason ? ' / ' + node.reason : ''}`,
    ];
    if (isClosedResult(node) || isFinalVerdict(node)) lines.push(`분기 역할: ${branchRoleLabel(node)}`);
    const source = NODE_BY_ID.get(node.satSourceId);
    if (source) lines.push(`해 발견 지점: ${sourceLabel(source)}`);
    const sources = (node.unsatSourceIds || []).map(id => NODE_BY_ID.get(id)).filter(Boolean);
    if (node.unsatRole === 'skipped') lines.push('기존 충돌로 탐색 생략 (BACKJUMP)');
    if (sources.length) lines.push(`모순 발견 지점: ${sources.slice(0, 8).map(sourceLabel).join(', ')}${sources.length > 8 ? ` 외 ${sources.length - 8}개` : ''}`);
    if (node.stableAddedCount) lines.push(`이 가지에서 추가 STABLE ${node.stableAddedCount}개`);
    lines.push(`현재 깊이: ${node.depth}`, `하위 탐색 최대 깊이: ${node.maxDepth}`, `bound: ${node.bound || '-'}`, `repair: ${node.repairs}`, `Simplex 호출: ${node.simplexCalls}`, `소요시간: ${node.duration.toFixed(6)}초`);
    return lines.join('\n');
  }

  function setText(element, value) {
    const text = String(value);
    if (element.textContent !== text) element.textContent = text;
  }
  function setClass(element, value) { if (element.className !== value) element.className = value; }
  function span(className, parent) {
    const element = document.createElement('span');
    element.className = className;
    parent.appendChild(element);
    return element;
  }
  function patchGrowth(view) {
    const growth = BRANCH_GROWTH.get(view.id) || 0;
    setText(view.growth, growth ? `+${growth.toLocaleString()}` : '');
    if (view.growth.hidden !== !growth) view.growth.hidden = !growth;
    const title = growth ? `아직 확인하지 않은 하위 가지 ${growth.toLocaleString()}개` : '';
    if (view.growth.title !== title) view.growth.title = title;
  }
  function acknowledgeGrowth(node) {
    for (; node; node = NODE_BY_ID.get(node.parentId)) {
      if (BRANCH_GROWTH.delete(node.id)) {
        const view = views.get(node.id);
        if (view) patchGrowth(view);
      }
    }
  }
  function matchesStatus(node, selected) {
    if (selected === 'FINAL') return finalPathIds.has(node.id);
    if (selected === 'STABLE') return (node.stableAddedCount || 0) > 0;
    if (selected === 'CLOSED') return isClosedResult(node);
    if (selected === 'CLOSED_SAT' || selected === 'CLOSED_UNSAT') return isClosedResult(node) && selected === `CLOSED_${node.branchResult}`;
    return selected === 'ALL' || node.result === selected || node.branchResult === selected || nodeStatusClass(node) === selected || (node.result === 'CLOSED' && selected === `CLOSED_${node.branchResult}`);
  }
  function matches(node) {
    const layer = $('#layerFilter').value.trim(), neuron = $('#neuronFilter').value.trim();
    return matchesStatus(node, $('#statusFilter').value) && (!layer || String(node.layer) === layer) && (!neuron || String(node.neuron) === neuron);
  }
  function flatFilter() {
    return $('#statusFilter').value !== 'FINAL' && ($('#statusFilter').value !== 'ALL' || $('#layerFilter').value.trim() !== '' || $('#neuronFilter').value.trim() !== '');
  }
  function indexData(data) {
    const previous = NODE_BY_ID;
    const hadSnapshot = DATA !== undefined;
    DATA = data;
    ALL = [];
    const pending = [...DATA.roots].reverse();
    while (pending.length) {
      const node = pending.pop();
      ALL.push(node);
      for (let i = node.children.length - 1; i >= 0; --i) pending.push(node.children[i]);
    }
    NODE_BY_ID = new Map(ALL.map(node => [node.id, node]));
    const latest = ALL.reduce((last, node) => !last || node.order > last.order ? node : last, null);
    latestNodeId = latest?.id ?? null;
    finalPathIds.clear();
    for (let node = latest; node; node = NODE_BY_ID.get(node.parentId)) finalPathIds.add(node.id);
    for (const id of BRANCH_GROWTH.keys()) {
      if (!NODE_BY_ID.has(id)) BRANCH_GROWTH.delete(id);
    }
    if (hadSnapshot) {
      for (const node of ALL) {
        const before = previous.get(node.id);
        const increase = Math.max(0, node.subtreeSize - 1 - (before ? before.subtreeSize - 1 : 0));
        if (increase) {
          BRANCH_GROWTH.set(node.id, (BRANCH_GROWTH.get(node.id) || 0) + increase);
        }
      }
    }
    updateFilterIndex();
  }
  function updateFilterIndex() {
    MATCHED = new Map();
    for (let i = ALL.length - 1; i >= 0; --i) {
      const node = ALL[i];
      MATCHED.set(node.id, matches(node) || (!flatFilter() && node.children.some(child => MATCHED.get(child.id))));
    }
    TIMELINE = [...ALL].sort((a, b) => (a.completedAt ?? Number.MAX_VALUE) - (b.completedAt ?? Number.MAX_VALUE) || a.order - b.order);
    TIMELINE.forEach((node, index) => { node.completionOrder = index + 1; });
    const DURATION_TIMELINE = [...ALL].sort((a, b) => b.duration - a.duration || a.order - b.order);
    DURATION_TIMELINE.forEach((node, index) => { node.durationOrder = index + 1; });
    if ($('#statusFilter').value === 'FINAL') expandMatchingTree();
  }

  function dispose(view) {
    view.disposed = true;
    for (const child of view.childViews.values()) dispose(child);
    view.childViews.clear();
    views.delete(view.id);
    view.li.remove();
  }
  function removeChildren(view) {
    for (const child of view.childViews.values()) dispose(child);
    view.childViews.clear();
    if (view.children) view.children.remove();
    view.children = null;
  }
  function createView(node) {
    const li = document.createElement('li'), details = document.createElement('details'), summary = document.createElement('summary');
    li.appendChild(details);
    details.appendChild(summary);
    details.dataset.id = node.id;
    const view = {id: node.id, li, details, summary, childViews: new Map(), children: null, disposed: false};
    view.label = span('label', summary);
    view.phase = span('phase', summary);
    view.nodeId = span('meta', summary);
    view.split = span('meta', summary);
    view.badge = span('badge', summary);
    view.stable = span('badge STABLE', summary);
    view.duration = span('meta', summary);
    const countGroup = span('branch-count', summary);
    view.growth = span('branch-growth', countGroup);
    view.count = span('collapsed', countGroup);
    // Tooltips are derived only when a visible row is inspected.
    const showTip = () => { const current = NODE_BY_ID.get(view.id); if (current) summary.title = tip(current); };
    summary.addEventListener('mouseenter', showTip);
    summary.addEventListener('focus', showTip);
    summary.addEventListener('click', event => {
      event.preventDefault();
      const node = NODE_BY_ID.get(view.id);
      if (!node) return;
      acknowledgeGrowth(node);
      // Record the click synchronously, before an arriving snapshot can patch
      // this row. The native toggle event is otherwise delivered later.
      if (!flatFilter() && node.children.length) {
        if (details.open) expandedIds.delete(view.id);
        else expandedIds.add(view.id);
      }
      patchView(view, node);
      saveState();
    });
    details.addEventListener('toggle', () => {
      if (view.disposed || flatFilter() || details.open === expandedIds.has(view.id)) return;
      acknowledgeGrowth(NODE_BY_ID.get(view.id));
      if (details.open) expandedIds.add(view.id);
      else expandedIds.delete(view.id);
      // Keep descendants' expansion choices, but discard their DOM on collapse.
      patchView(view, NODE_BY_ID.get(view.id));
      saveState();
    });
    views.set(node.id, view);
    return view;
  }
  function patchView(view, node) {
    const color = nodeStatusClass(node);
    setClass(view.summary, 's-' + color);
    const weight = `${Math.min(10, 3 + Math.log10(1 + node.duration * 1000) * 1.6)}px`;
    if (view.summary.style.getPropertyValue('--weight') !== weight) view.summary.style.setProperty('--weight', weight);
    setText(view.label, neuronLabel(node));
    setClass(view.phase, 'phase ' + (node.phase === '+' ? 'plus' : 'minus'));
    setText(view.phase, node.phase);
    setText(view.nodeId, `노드 #${node.id}`);
    setText(view.split, `Split #${node.splitNumber ?? '?'}`);
    setClass(view.badge, 'badge ' + color);
    setText(view.badge, resultLabel(node));
    setText(view.stable, node.stableAddedCount ? `STABLE ${node.stableAddedCount.toLocaleString()}개` : '');
    view.stable.hidden = !node.stableAddedCount;
    setText(view.duration, `${node.duration.toFixed(3)}s`);
    patchGrowth(view);
    setText(view.count, `하위 가지 ${(node.subtreeSize - 1).toLocaleString()}개`);
    if (view.summary.matches(':hover, :focus')) view.summary.title = tip(node);
    const hasChildren = !flatFilter() && node.children.length > 0;
    view.details.classList.toggle('leaf', !hasChildren);
    view.details.classList.toggle('recent-path', recentPathIds.has(node.id));
    view.details.classList.toggle('recent-target', node.id === recentTargetId);
    const open = hasChildren && expandedIds.has(node.id);
    if (view.details.open !== open) view.details.open = open;
    if (open) {
      if (!view.children) {
        view.children = document.createElement('ul');
        view.children.className = 'children';
        view.details.appendChild(view.children);
      }
      reconcile(view.children, view.childViews, node.children);
    } else removeChildren(view);
  }
  function reconcile(container, currentViews, nodes) {
    const desired = nodes.filter(node => MATCHED.get(node.id));
    const wanted = new Set(desired.map(node => node.id));
    for (const [id, view] of currentViews) {
      if (!wanted.has(id)) { dispose(view); currentViews.delete(id); }
    }
    let cursor = container.firstElementChild;
    for (const node of desired) {
      let view = currentViews.get(node.id);
      if (!view) { view = createView(node); currentViews.set(node.id, view); }
      patchView(view, node);
      if (cursor === view.li) cursor = cursor.nextElementSibling;
      else container.insertBefore(view.li, cursor);
    }
  }
  function renderTree() {
    $('#treeEmpty')?.remove();
    const flat = flatFilter();
    if (renderedFlat !== flat) {
      for (const view of rootViews.values()) dispose(view);
      rootViews.clear();
      renderedFlat = flat;
    }
    reconcile($('#tree'), rootViews, flat ? ALL : DATA.roots);
    if (!rootViews.size) {
      const empty = document.createElement('li');
      empty.id = 'treeEmpty'; empty.className = 'empty';
      empty.textContent = ALL.length ? '조건에 맞는 탐색 기록이 없습니다.' : '기록된 ReLU 분기가 없습니다.';
      $('#tree').appendChild(empty);
    }
  }

  function renderTimeline() {
    const order = $('#timelineOrder').value;
    const source = order === 'completion' ? TIMELINE : order === 'duration' ? [...ALL].sort((a, b) => a.durationOrder - b.durationOrder) : ALL;
    setText($('#timelineTitle'), order === 'completion' ? '탐색 완료 순서 타임라인' : order === 'duration' ? '탐색 소요시간순 타임라인' : '탐색 방문 순서 타임라인');
    if (!$('#timelinePanel').open) return;
    const matching = source.filter(matches);
    timelinePage = Math.min(timelinePage, Math.max(0, Math.ceil(matching.length / TIMELINE_PAGE_SIZE) - 1));
    const first = timelinePage * TIMELINE_PAGE_SIZE;
    const page = matching.slice(first, first + TIMELINE_PAGE_SIZE);
    const wanted = new Set(page.map(node => node.id));
    for (const [id, view] of timelineViews) {
      if (!wanted.has(id)) { view.row.remove(); timelineViews.delete(id); }
    }
    $('#timelineEmpty')?.remove();
    let cursor = $('#timeline').firstElementChild;
    for (const node of page) {
      let view = timelineViews.get(node.id);
      if (!view) {
        const row = document.createElement('div'); row.className = 'row'; row.dataset.id = node.id;
        view = {row, number: span('', row), label: span('', row), badge: span('badge', row), duration: span('duration', row)};
        row.addEventListener('mouseenter', () => { const current = NODE_BY_ID.get(node.id); if (current) row.title = tip(current); });
        timelineViews.set(node.id, view);
      }
      const number = order === 'completion' ? node.completionOrder : order === 'duration' ? node.durationOrder : node.order;
      setText(view.number, `#${number}`);
      setText(view.label, `${neuronLabel(node)} ${node.phase} · 노드 #${node.id} · Split #${node.splitNumber ?? '?'} · 방문 #${node.order} · 완료 #${node.completionOrder}`);
      setClass(view.badge, 'badge ' + nodeStatusClass(node));
      setText(view.badge, resultLabel(node));
      setText(view.duration, `${node.duration.toFixed(3)}s`);
      if (cursor === view.row) cursor = cursor.nextElementSibling;
      else $('#timeline').insertBefore(view.row, cursor);
    }
    if (!page.length) {
      const empty = document.createElement('div'); empty.id = 'timelineEmpty'; empty.className = 'empty'; empty.textContent = '표시할 탐색 기록이 없습니다.';
      $('#timeline').appendChild(empty);
    }
    setText($('#timelinePage'), matching.length ? `${first + 1}–${first + page.length} / ${matching.length.toLocaleString()}개` : '0개');
    $('#timelinePrev').disabled = timelinePage === 0;
    $('#timelineNext').disabled = first + page.length >= matching.length;
  }

  function stableSourceLabel(source) {
    return {initial: '초기 경계 유도', search: '탐색 중 경계 유도', pivot: '피벗 경계 강화'}[source] || source;
  }
  function boundsLabel(bounds) {
    if (!Array.isArray(bounds)) return '-';
    return `[${bounds.map((value, index) => value == null ? (index ? '∞' : '−∞') : Number(value.toPrecision(5))).join(', ')}]`;
  }
  function renderStableRecords() {
    if (!$('#stablePanel').open) return;
    const scope = $('#stableScope').value, order = $('#stableOrder').value;
    const layer = $('#layerFilter').value.trim(), neuron = $('#neuronFilter').value.trim();
    const records = (DATA.stableRecords || []).filter(record =>
      (scope === 'all' || (record.source === 'initial') === (scope === 'initial')) &&
      (!layer || String(record.layer) === layer) && (!neuron || String(record.neuron) === neuron));
    const compareNeuron = (a, b) => (a.layer ?? Number.MAX_VALUE) - (b.layer ?? Number.MAX_VALUE) || (a.neuron ?? Number.MAX_VALUE) - (b.neuron ?? Number.MAX_VALUE);
    const compareNode = (a, b) => (a.roundIndex ?? 0) - (b.roundIndex ?? 0) || (a.nodeId ?? a.branchId ?? 0) - (b.nodeId ?? b.branchId ?? 0);
    records.sort((a, b) => (order === 'neuron' ? compareNeuron(a, b) : order === 'node' ? compareNode(a, b) : 0) || b.id - a.id);
    stablePage = Math.min(stablePage, Math.max(0, Math.ceil(records.length / TIMELINE_PAGE_SIZE) - 1));
    const first = stablePage * TIMELINE_PAGE_SIZE, page = records.slice(first, first + TIMELINE_PAGE_SIZE);
    const wanted = new Set(page.map(record => record.id));
    for (const [id, view] of stableViews) {
      if (!wanted.has(id)) { view.row.remove(); stableViews.delete(id); }
    }
    $('#stableEmpty')?.remove();
    let cursor = $('#stableRows').firstElementChild;
    for (const record of page) {
      let view = stableViews.get(record.id);
      if (!view) {
        const row = document.createElement('tr'); row.dataset.id = record.id;
        view = {row};
        for (const name of ['label', 'badge', 'state', 'source', 'range', 'scope']) {
          view[name] = document.createElement('td'); row.appendChild(view[name]);
        }
        view.badge.appendChild(document.createElement('span')).className = 'badge STABLE';
        view.range.className = 'range';
        view.location = document.createElement('button'); view.scope.appendChild(view.location);
        view.location.onclick = () => {
          const target = NODE_BY_ID.get(view.record.nodeId ?? view.record.branchId);
          if (target) { expandedIds.clear(); revealNodePath(target); }
        };
        stableViews.set(record.id, view);
      }
      view.record = record;
      setText(view.label, neuronLabel(record));
      setText(view.badge.firstChild, 'STABLE');
      setText(view.state, record.state === 'active' ? '활성(+)' : '비활성(-)');
      setClass(view.state, 'stable-state ' + (record.state === 'active' ? 'plus' : 'minus'));
      setText(view.source, stableSourceLabel(record.source));
      setText(view.range, `${boundsLabel(record.before)} → ${boundsLabel(record.after)}`);
      const round = record.roundIndex == null ? '' : `D${record.roundIndex} · `;
      const nodeId = record.nodeId ?? record.branchId;
      setText(view.location, round + (nodeId == null ? (record.depth ? '노드 미기록' : 'ROOT') : `노드 #${nodeId}`));
      view.location.disabled = nodeId == null || !NODE_BY_ID.has(nodeId);
      view.location.title = nodeId == null ? '이론 호출의 루트 또는 기록된 확정 위치 확인' : '이 탐색 노드에서 STABLE 확정 · 클릭하여 경로 보기';
      if (cursor === view.row) cursor = cursor.nextElementSibling;
      else $('#stableRows').insertBefore(view.row, cursor);
    }
    if (!page.length) {
      const row = document.createElement('tr'); row.id = 'stableEmpty';
      const cell = document.createElement('td'); cell.colSpan = 6; cell.className = 'empty';
      cell.textContent = '조건에 맞는 STABLE 기록이 없습니다.'; row.appendChild(cell); $('#stableRows').appendChild(row);
    }
    setText($('#stableCount'), `${records.length.toLocaleString()}건 / 전체 ${(DATA.stableRecords || []).length.toLocaleString()}건`);
    setText($('#stablePage'), records.length ? `${first + 1}–${first + page.length} / ${records.length.toLocaleString()}건` : '0건');
    $('#stablePrev').disabled = stablePage === 0;
    $('#stableNext').disabled = first + page.length >= records.length;
  }

  function renderMetrics() {
    $('#recentPath').disabled = latestNodeId === null;
    renderTrackingButton();
    setText($('#title'), DATA.title);
    setText($('#overall'), DATA.status);
    setClass($('#overall'), 'pill ' + statusClass(DATA.status));
    setText($('#count'), `분기 노드 ${DATA.splitCount.toLocaleString()}개 · 가지 ${DATA.nodeCount.toLocaleString()}개`);
    setText($('#reason'), DATA.reason || '');
    const bd = DATA.boundDerivation || {}, ps = DATA.prunedSpace || {};
    const hasBound = Number.isInteger(bd.total) && Number.isInteger(bd.fixed) && Number.isInteger(bd.remaining);
    const hasSpace = Number.isInteger(ps.totalExponent), percent = hasSpace ? ps.ratio * 100 : 0;
    const percentText = value => value === 0 ? '0.0' : value >= 0.01 ? value.toFixed(2) : value.toExponential(2);
    const terms = (ps.terms || []).map(term => `${term.count > 1 ? term.count + ' × ' : ''}2^${term.exponent}`).join(' + ') || '0';
    setText($('#closedMetric'), hasSpace ? `(${terms}) / 2^${ps.totalExponent} = ${percentText(percent)}%` : '계산 중');
    setText($('#closedDetail'), hasSpace ? `탐색하지 않아도 되는 공간 ${percentText(percent)}% · 남은 공간 ${percentText(Math.max(0, 100 - percent))}%` : '경계 유도 결과를 기다리는 중');
    $('#closedBar').style.width = `${Math.min(100, percent)}%`;
    setText($('#boundMetric'), hasBound ? `${bd.total.toLocaleString()} → ${bd.fixed.toLocaleString()} 고정 → ${bd.remaining.toLocaleString()} 불안정` : '계산 중');
    setText($('#boundDetail'), hasBound ? `전체 ${bd.total.toLocaleString()}개 중 경계로 ${bd.fixed.toLocaleString()}개 고정` : '경계 유도 결과를 기다리는 중');
    $('#boundBar').style.width = hasBound && bd.total ? `${bd.fixed / bd.total * 100}%` : '0%';
  }
  function saveState() {
    window.name = STATE_PREFIX + JSON.stringify({
      path: stateKey, expandedIds: [...expandedIds], timelinePage, trackLatestPath,
      stablePage, stableScope: $('#stableScope').value, stableOrder: $('#stableOrder').value,
      reportPanels: Object.fromEntries(reportPanels.map(panel => [panel.key, panel.element.open])),
      timelineOrder:$('#timelineOrder').value, status: $('#statusFilter').value,
      layer: $('#layerFilter').value, neuron: $('#neuronFilter').value,
      scrollX: window.scrollX, scrollY: window.scrollY, timelineScroll: readPanelScroll('timeline'),
      treeScrollLeft: $('#treeViewport').scrollLeft, treeScrollTop: $('#treeViewport').scrollTop,
      stableScroll: readPanelScroll('stable'),
    });
  }
  function announceStatus() {
    if (DATA.status === lastStatus) return;
    lastStatus = DATA.status;
    if (window.parent !== window) window.parent.postMessage({type: 'relu-search-tree-status', status: DATA.status}, '*');
    if (DATA.status !== 'RUNNING' && standaloneTimer !== null) { clearInterval(standaloneTimer); standaloneTimer = null; }
  }
  function applySnapshot(data) {
    if (!data || !Array.isArray(data.roots)) return;
    const scrollX = window.scrollX, scrollY = window.scrollY, timelineScroll = readPanelScroll('timeline');
    const stableScroll = readPanelScroll('stable');
    const treeScrollLeft = $('#treeViewport').scrollLeft, treeScrollTop = $('#treeViewport').scrollTop;
    const previousLatest = latestNodeId;
    indexData(data);
    const follow = trackLatestPath && latestNodeId !== null && latestNodeId !== previousLatest;
    if (DATA.status !== 'RUNNING') trackLatestPath = false;
    renderMetrics(); renderTree(); renderTimeline(); renderStableRecords();
    sizeTreeViewport();
    $('#treeViewport').scrollLeft = treeScrollLeft;
    $('#treeViewport').scrollTop = treeScrollTop;
    if (follow) revealRecentPath(false);
    window.scrollTo(scrollX, scrollY);
    restorePanelScroll('timeline', timelineScroll);
    restorePanelScroll('stable', stableScroll);
    saveState(); announceStatus();
  }
  function requestSnapshot() {
    if (snapshotPending || DATA.status !== 'RUNNING') return;
    if (!snapshotFrame) {
      snapshotFrame = document.createElement('iframe');
      snapshotFrame.hidden = true; snapshotFrame.id = 'treeSnapshotLoader';
      snapshotFrame.setAttribute('aria-hidden', 'true'); snapshotFrame.tabIndex = -1;
      snapshotFrame.addEventListener('load', () => { snapshotPending = false; });
      document.body.appendChild(snapshotFrame);
    }
    snapshotPending = true;
    const url = new URL(location.href);
    url.searchParams.set('tree-snapshot', '1'); url.searchParams.set('t', Date.now());
    snapshotFrame.src = url.href;
  }
  window.addEventListener('message', event => {
    const message = event.data || {};
    if (message.type === SNAPSHOT_TYPE && snapshotFrame && event.source === snapshotFrame.contentWindow) {
      snapshotPending = false;
      applySnapshot(message.data);
    } else if (message.type === 'relu-search-tree-refresh' && window.parent !== window && event.source === window.parent) requestSnapshot();
  });

  function renderTrackingButton() {
    const button = $('#trackRecentPath');
    button.disabled = DATA.status !== 'RUNNING';
    button.setAttribute('aria-pressed', String(trackLatestPath));
    setText(button, trackLatestPath ? '최근 경로 추적 중' : '최근 경로 추적');
  }
  function revealRecentPath(focus = true) {
    revealNodePath(NODE_BY_ID.get(latestNodeId), focus);
  }
  function scrollTreeNode(summary, center) {
    const viewport = $('#treeViewport');
    if (center) {
      viewport.style.scrollMarginTop = `${$('header').offsetHeight + 12}px`;
      viewport.scrollIntoView({block: 'nearest', inline: 'nearest'});
    }
    const bounds = viewport.getBoundingClientRect(), row = summary.getBoundingClientRect();
    const padding = 4;
    const left = row.left - bounds.left - viewport.clientLeft + viewport.scrollLeft;
    const top = row.top - bounds.top - viewport.clientTop + viewport.scrollTop;
    // Oversized rows only need their node identity and verdict in view.
    const leadingWidth = summary.querySelector('.badge').getBoundingClientRect().right - row.left;
    const width = Math.min(row.width > viewport.clientWidth - padding * 2 ? leadingWidth : row.width, viewport.clientWidth - padding * 2);
    let x = viewport.scrollLeft, y = viewport.scrollTop;
    if (left < x + padding) x = left - padding;
    else if (left + width > x + viewport.clientWidth - padding) x = left + width - viewport.clientWidth + padding;
    if (center) y = top - (viewport.clientHeight - row.height) / 2;
    else if (top < y + padding) y = top - padding;
    else if (top + row.height > y + viewport.clientHeight - padding) y = top + row.height - viewport.clientHeight + padding;
    if (Math.abs(x - viewport.scrollLeft) > 1 || Math.abs(y - viewport.scrollTop) > 1) viewport.scrollTo({left: x, top: y, behavior: 'instant'});
  }
  function revealNodePath(target, focus = true) {
    if (!target) return;
    if (flatFilter() || !MATCHED.get(target.id)) {
      $('#statusFilter').value = 'ALL';
      $('#layerFilter').value = '';
      $('#neuronFilter').value = '';
      timelinePage = 0;
      updateFilterIndex(); renderTimeline(); renderStableRecords();
    }
    recentPathIds.clear();
    recentTargetId = target.id;
    for (let node = target; node && !recentPathIds.has(node.id); node = NODE_BY_ID.get(node.parentId)) {
      recentPathIds.add(node.id);
      if (node.id !== target.id) expandedIds.add(node.id);
    }
    acknowledgeGrowth(target);
    renderTree();
    const summary = views.get(target.id)?.summary;
    if (summary) {
      if (focus) summary.focus({preventScroll: true});
      scrollTreeNode(summary, focus);
    }
    saveState();
  }
  function readPanelScroll(key) {
    const panel = reportPanels.find(item => item.key === key);
    if (panel.element.open) panel.scroll = panel.list.scrollTop;
    return panel.scroll;
  }
  function restorePanelScroll(key, scroll) {
    const panel = reportPanels.find(item => item.key === key);
    panel.scroll = scroll;
    if (panel.element.open) panel.list.scrollTop = scroll;
  }
  function renderPanelToggle() {
    const anyOpen = reportPanels.some(panel => panel.element.open);
    setText($('#toggleReportPanels'), anyOpen ? '하단 패널 접기' : '하단 패널 펼치기');
    $('#toggleReportPanels').setAttribute('aria-expanded', String(anyOpen));
  }
  for (const panel of reportPanels) {
    panel.element.open = savedState.reportPanels?.[panel.key] !== false;
    panel.element.querySelector(':scope > summary').addEventListener('click', () => readPanelScroll(panel.key));
    panel.element.addEventListener('toggle', () => {
      if (panel.element.open) {
        panel.render(); panel.list.scrollTop = panel.scroll;
      }
      renderPanelToggle(); saveState();
    });
  }
  $('#toggleReportPanels').onclick = () => {
    const open = !reportPanels.some(panel => panel.element.open);
    for (const panel of reportPanels) {
      readPanelScroll(panel.key);
      panel.element.open = open;
      if (open) { panel.render(); panel.list.scrollTop = panel.scroll; }
    }
    renderPanelToggle(); saveState();
  };
  $('#defaultView').onclick = () => { expandedIds.clear(); renderTree(); $('#treeViewport').scrollTo(0, 0); saveState(); };
  $('#recentPath').onclick = () => {
    expandedIds.clear();
    revealRecentPath();
  };
  $('#trackRecentPath').onclick = () => {
    trackLatestPath = !trackLatestPath && DATA.status === 'RUNNING';
    renderTrackingButton();
    if (trackLatestPath) revealRecentPath(false);
    saveState();
  };
  function expandMatchingTree() {
    for (const node of ALL) {
      if (MATCHED.get(node.id) && node.children.length) expandedIds.add(node.id);
    }
  }
  $('#expandAll').onclick = () => {
    expandMatchingTree();
    renderTree(); saveState();
  };
  $('#collapseAll').onclick = () => { for (const root of DATA.roots) expandedIds.delete(root.id); renderTree(); saveState(); };
  const filterChanged = () => { timelinePage = 0; stablePage = 0; updateFilterIndex(); expandMatchingTree(); renderTree(); $('#treeViewport').scrollTo(0, 0); renderTimeline(); renderStableRecords(); saveState(); };
  $('#statusFilter').onchange = filterChanged;
  $('#layerFilter').oninput = filterChanged;
  $('#neuronFilter').oninput = filterChanged;
  $('#timelineOrder').onchange = () => { timelinePage = 0; renderTimeline(); saveState(); };
  $('#timelinePrev').onclick = () => { timelinePage--; renderTimeline(); $('#timeline').scrollTop = 0; saveState(); };
  $('#timelineNext').onclick = () => { timelinePage++; renderTimeline(); $('#timeline').scrollTop = 0; saveState(); };
  const stableFilterChanged = () => { stablePage = 0; renderStableRecords(); $('#stableList').scrollTop = 0; saveState(); };
  $('#stableScope').onchange = stableFilterChanged;
  $('#stableOrder').onchange = stableFilterChanged;
  $('#stablePrev').onclick = () => { stablePage--; renderStableRecords(); $('#stableList').scrollTop = 0; saveState(); };
  $('#stableNext').onclick = () => { stablePage++; renderStableRecords(); $('#stableList').scrollTop = 0; saveState(); };
  window.addEventListener('pagehide', saveState);

  const sizeTreeViewport = () => $('#treeViewport').style.setProperty('--tree-header-height', `${$('header').offsetHeight}px`);
  new ResizeObserver(sizeTreeViewport).observe($('header'));
  indexData(INITIAL_DATA);
  if (DATA.status !== 'RUNNING') trackLatestPath = false;
  renderMetrics(); renderTree(); renderTimeline(); renderStableRecords(); renderPanelToggle();
  sizeTreeViewport();
  $('#treeViewport').scrollLeft = Number(savedState.treeScrollLeft) || 0;
  $('#treeViewport').scrollTop = Number(savedState.treeScrollTop) || 0;
  window.scrollTo(Number(savedState.scrollX) || 0, Number(savedState.scrollY) || 0);
  restorePanelScroll('timeline', Number(savedState.timelineScroll) || 0);
  restorePanelScroll('stable', Number(savedState.stableScroll) || 0);
  if (trackLatestPath) revealRecentPath(false);
  if (window.parent === window && DATA.status === 'RUNNING') standaloneTimer = setInterval(requestSnapshot, 2000);
  announceStatus();
})();
