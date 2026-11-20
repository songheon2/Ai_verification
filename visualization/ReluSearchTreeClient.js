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
  const treeOrders = ['node', 'layer', 'duration', 'completion'];
  if (treeOrders.includes(savedState.treeOrder)) $('#treeOrder').value = savedState.treeOrder;
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
  let SEARCH_CATALOG = new Map();
  const searchCombos=[];
  for (const [inputId, listId] of [['layerFilter','layerOptions'],['neuronFilter','neuronOptions']]) {
    const input=$('#'+inputId), list=document.createElement('datalist');
    list.id=listId;document.body.append(list);
    input.setAttribute('autocomplete','off');
    input.setAttribute('aria-label',inputId==='layerFilter'?'Layer 번호 입력 또는 선택':'Neuron 번호 입력 또는 선택');
    const wrapper=document.createElement('div'),button=document.createElement('button'),popup=document.createElement('div');
    wrapper.className='neuron-combo';button.type='button';button.textContent='▾';
    button.setAttribute('aria-label',inputId==='layerFilter'?'전체 레이어 목록':'해당 레이어 기록 있는 뉴런 목록');
    popup.id=inputId+'Choices';popup.className='neuron-choices';popup.hidden=true;popup.setAttribute('role','listbox');
    input.setAttribute('role','combobox');input.setAttribute('aria-autocomplete','none');
    input.setAttribute('aria-controls',popup.id);input.setAttribute('aria-expanded','false');
    input.before(wrapper);wrapper.append(input,button,popup);
    const close=()=>{popup.hidden=true;input.setAttribute('aria-expanded','false');};
    const render=()=>{
      if(popup.hidden)return;
      popup.replaceChildren(...[...list.options].map(option=>{
        const choice=document.createElement('button');choice.type='button';choice.textContent=option.label;
        choice.dataset.value=option.value;choice.setAttribute('role','option');
        choice.setAttribute('aria-selected',String(option.value===input.value));
        choice.onclick=()=>{input.value=option.value;input.dispatchEvent(new Event('input',{bubbles:true}));close();input.focus({preventScroll:true});};
        return choice;
      }));
      if(!popup.children.length){const empty=document.createElement('div');empty.className='meta';empty.textContent=inputId==='neuronFilter'?($('#layerFilter').value.trim()?'기록 있는 뉴런 없음':'레이어를 먼저 선택하세요'):'레이어 정보 없음';popup.append(empty);}
    };
    const open=()=>{for(const combo of searchCombos)combo.close();popup.hidden=false;input.setAttribute('aria-expanded','true');render();popup.scrollTop=0;};
    button.onclick=()=>popup.hidden?open():close();input.addEventListener('click',open);
    input.addEventListener('input',render);
    input.addEventListener('keydown',event=>{if(event.key==='ArrowDown'){event.preventDefault();open();popup.querySelector('button')?.focus();}else if(event.key==='Escape')close();});
    popup.addEventListener('keydown',event=>{
      if(event.key==='Escape'){close();input.focus();return;}
      if(!['ArrowDown','ArrowUp'].includes(event.key))return;
      event.preventDefault();const choices=[...popup.querySelectorAll('button')],index=choices.indexOf(document.activeElement);
      choices[Math.max(0,Math.min(choices.length-1,index+(event.key==='ArrowDown'?1:-1)))]?.focus();
    });
    searchCombos.push({wrapper,close,render});
  }
  document.addEventListener('click',event=>{for(const combo of searchCombos)if(!combo.wrapper.contains(event.target))combo.close();});
  function syncSearchChoices() {
    const replace=(id,entries)=>{
      const list=$('#'+id),key=JSON.stringify(entries);
      if(list.dataset.key===key)return;
      list.dataset.key=key;
      list.replaceChildren(...entries.map(([value,label])=>{
        const option=document.createElement('option');option.value=String(value);option.label=label;return option;
      }));
    };
    const layers=[...SEARCH_CATALOG.keys()].sort((a,b)=>a-b);
    replace('layerOptions',layers.map(layer=>[layer,`Layer ${layer} · ${SEARCH_CATALOG.get(layer).size} neurons`]));
    const selected=$('#layerFilter').value.trim();
    const neurons=selected===''?new Set():
      SEARCH_CATALOG.get(Number(selected)) || new Set();
    replace('neuronOptions',[...neurons].sort((a,b)=>a-b).map(neuron=>[neuron,`Neuron ${neuron}`]));
    for(const combo of searchCombos)combo.render();
  }
  let ALL = [];
  let SCOPED = [];
  let scopeIds = new Set();
  let scopeRootId = savedState.scopeRootId ?? null;
  let deferredScopeRootId = null;
  const scopeViews = new Map(Object.entries(savedState.scopeViews || {}));
  let subtreeSearch = savedState.subtreeSearch === true;
  let subtreeSearchRoots = [];
  const subtreeTargetIds = new Set();
  let autoExpandSubtree = savedState.autoExpandSubtree ?? (scopeRootId !== null);
  const subtreeCollapsedIds = new Set(savedState.subtreeCollapsedIds || []);
  let NODE_BY_ID = new Map();
  const BRANCH_GROWTH = new Map();
  let latestNodeId = null;
  const finalPathIds = new Set();
  const recentPathIds = new Set(scopeViews.get(scopeViewKey())?.recentPathIds || []);
  let recentTargetId = scopeViews.get(scopeViewKey())?.recentTargetId ?? null;
  let trackLatestPath = savedState.trackLatestPath === true;
  let ancestorSearch = savedState.ancestorSearch === true;
  let ancestorFeedback = '';
  const ancestorPathIds = new Set();
  const ancestorTargetIds = new Set();
  let MATCHED = new Map();
  let renderedFlat = null;
  let renderedScopeRootId;
  let TIMELINE = [];
  let snapshotFrame = null;
  let snapshotPending = false;
  let inputBusy=false;
  let snapshotTimeout=null,snapshotRequest=0;
  let finalSnapshotQueued = false;
  const snapshotAsset = {};
  let requestedSnapshotVersion = null;
  const SCROLL_IDLE_MS = 220;
  let lastScrollInput = -Infinity, scrollIdleTimer = null;
  let queuedSnapshot = null, queuedSnapshotRequest = false, scrollbarDragging = false;
  const scrollActive = () => performance.now() - lastScrollInput < SCROLL_IDLE_MS;
  function settleScroll() {
    if (scrollIdleTimer !== null) clearTimeout(scrollIdleTimer);
    scrollIdleTimer = setTimeout(() => {
      scrollIdleTimer = null;
      if (scrollActive() || scrollbarDragging) {settleScroll();return;}
      if (queuedSnapshot) {const latest=queuedSnapshot;queuedSnapshot=null;applySnapshot(latest,false);}
      if (queuedSnapshotRequest) {queuedSnapshotRequest=false;const force=finalSnapshotQueued;finalSnapshotQueued=false;requestSnapshot(force);}
    }, SCROLL_IDLE_MS);
  }
  function noteScrollInput() {lastScrollInput=performance.now();settleScroll();}
  document.addEventListener('wheel',noteScrollInput,{passive:true,capture:true});
  document.addEventListener('touchmove',noteScrollInput,{passive:true,capture:true});
  document.addEventListener('pointerdown',event=>{
    const viewport=$('#treeViewport'),box=viewport.getBoundingClientRect();
    scrollbarDragging=event.target===viewport &&
      (event.clientX>=box.left+viewport.clientWidth || event.clientY>=box.top+viewport.clientHeight);
    if(scrollbarDragging)noteScrollInput();
  },{passive:true,capture:true});
  document.addEventListener('pointermove',event=>{if(scrollbarDragging){if(event.buttons===0)endScrollbarDrag();else noteScrollInput();}},{passive:true,capture:true});
  const endScrollbarDrag=()=>{if(scrollbarDragging){scrollbarDragging=false;noteScrollInput();}};
  document.addEventListener('pointerup',endScrollbarDrag,{passive:true,capture:true});
  document.addEventListener('pointercancel',endScrollbarDrag,{passive:true,capture:true});
  window.addEventListener('blur',endScrollbarDrag);
  document.addEventListener('keydown',event=>{
    if(!event.target.closest?.('input,select,textarea,[contenteditable="true"]') &&
       ['ArrowUp','ArrowDown','ArrowLeft','ArrowRight','PageUp','PageDown','Home','End',' '].includes(event.key))noteScrollInput();
  },{capture:true});
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
  function selectForAnalysis(node, label='ReLU 탐색 트리') {
    if(!node || window.parent===window)return;
    const scope={topic:'tree'};
    if(Number.isInteger(node.roundIndex))scope.round=node.roundIndex;
    if(Number.isInteger(node.layer)&&Number.isInteger(node.neuron)){scope.layer=node.layer;scope.neuron=node.neuron;}
    if(node.branchId)scope.branch_id=node.branchId;
    const caption=[`트리 · 노드 #${node.id}`,node.roundIndex==null?'':`R${node.roundIndex}`,`${neuronLabel(node)} (${node.phase})`].filter(Boolean).join(' · ');
    window.parent.postMessage({type:'log-analysis-select',id:`tree-node-${node.id}`,label:caption,scope,text:`${label}\n${node.roundIndex==null?'':'DPLL Round: '+node.roundIndex+'\n'}${tip(node)}`},'*');
  }
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
    if (ancestorSearch) return ancestorPathIds.has(node.id);
    if (subtreeSearch) return scopeIds.has(node.id);
    const layer = $('#layerFilter').value.trim(), neuron = $('#neuronFilter').value.trim();
    return scopeIds.has(node.id) && matchesStatus(node, $('#statusFilter').value) && (!layer || String(node.layer) === layer) && (!neuron || String(node.neuron) === neuron);
  }
  function flatFilter() {
    return $('#treeOrder').value !== 'node' || (!ancestorSearch && !subtreeSearch && $('#statusFilter').value !== 'FINAL' && ($('#statusFilter').value !== 'ALL' || $('#layerFilter').value.trim() !== '' || $('#neuronFilter').value.trim() !== ''));
  }
  function compareCompletion(a, b) {
    if (a.completionSequence != null && b.completionSequence != null) return a.completionSequence - b.completionSequence;
    return (a.completedAt ?? Infinity) - (b.completedAt ?? Infinity) || a.id - b.id;
  }
  function compareTreeNodes(a, b) {
    const order = $('#treeOrder').value;
    if (order === 'layer') return (a.layer ?? Infinity) - (b.layer ?? Infinity) || (a.neuron ?? Infinity) - (b.neuron ?? Infinity) || a.id - b.id;
    if (order === 'duration') return b.duration - a.duration || a.id - b.id;
    if (order === 'completion') return compareCompletion(a, b);
    return a.id - b.id;
  }
  function indexData(data) {
    const previous = NODE_BY_ID;
    const hadSnapshot = DATA !== undefined;
    DATA = {...data,modelLayerSizes:data.modelLayerSizes?.length?data.modelLayerSizes:DATA?.modelLayerSizes || []};
    document.body.setAttribute('aria-busy', String(Boolean(data.deferredData)));
    if (!data.deferredData && deferredScopeRootId !== null) {
      scopeRootId = deferredScopeRootId; deferredScopeRootId = null;
    }
    ALL = [];
    const pending = [...DATA.roots].reverse();
    while (pending.length) {
      const node = pending.pop();
      ALL.push(node);
      for (let i = node.children.length - 1; i >= 0; --i) pending.push(node.children[i]);
    }
    NODE_BY_ID = new Map(ALL.map(node => [node.id, node]));
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
    SEARCH_CATALOG = new Map();
    const sizes=DATA.modelLayerSizes;
    for(let layer=1;layer<sizes.length-1;layer++) {
      SEARCH_CATALOG.set(layer,new Set());
    }
    for(const node of [...ALL,...(DATA.stableRecords || [])]) {
      const layer=node.layer,neuron=node.neuron;
      if(!Number.isInteger(layer)||!Number.isInteger(neuron))continue;
      if(!SEARCH_CATALOG.has(layer))SEARCH_CATALOG.set(layer,new Set());
      SEARCH_CATALOG.get(layer).add(neuron);
    }
    syncSearchChoices();
    updateScopeIndex();
    updateFilterIndex();
  }
  function scopeViewKey() { return subtreeSearch ? 'subtree-search' : String(scopeRootId); }
  function scopeRoots() {
    if (subtreeSearch) return subtreeSearchRoots;
    return scopeRootId === null ? DATA.roots : [NODE_BY_ID.get(scopeRootId)];
  }
  function updateScopeIndex() {
    if (scopeRootId !== null && !NODE_BY_ID.has(scopeRootId)) {
      if (DATA.deferredData) deferredScopeRootId = scopeRootId;
      scopeRootId = null;
    }
    subtreeSearchRoots = []; subtreeTargetIds.clear();
    if (subtreeSearch) {
      const layer = $('#layerFilter').value.trim(), neuron = $('#neuronFilter').value.trim();
      const included = new Set();
      for (const node of ALL) {
        const target = String(node.layer) === layer && String(node.neuron) === neuron;
        if (target) {
          subtreeTargetIds.add(node.id);
          if (!included.has(node.parentId)) subtreeSearchRoots.push(node);
        }
        if (target || included.has(node.parentId)) included.add(node.id);
      }
    }
    SCOPED = [];
    const pending = [...scopeRoots()].reverse();
    while (pending.length) {
      const node = pending.pop();
      SCOPED.push(node);
      for (let i = node.children.length - 1; i >= 0; --i) pending.push(node.children[i]);
    }
    scopeIds = new Set(SCOPED.map(node => node.id));
    const latest = SCOPED.reduce((last, node) => !last || node.order > last.order ? node : last, null);
    latestNodeId = latest?.id ?? null;
    finalPathIds.clear();
    for (let node = latest; node && scopeIds.has(node.id); node = NODE_BY_ID.get(node.parentId)) finalPathIds.add(node.id);
  }
  function updateFilterIndex() {
    ancestorPathIds.clear(); ancestorTargetIds.clear();
    if (ancestorSearch) {
      const layer = $('#layerFilter').value.trim(), neuron = $('#neuronFilter').value.trim();
      for (const target of ALL) {
        if (String(target.layer) !== layer || String(target.neuron) !== neuron) continue;
        ancestorTargetIds.add(target.id);
        for (let node = target; node && !ancestorPathIds.has(node.id); node = NODE_BY_ID.get(node.parentId)) ancestorPathIds.add(node.id);
      }
    }
    MATCHED = new Map();
    for (let i = SCOPED.length - 1; i >= 0; --i) {
      const node = SCOPED[i];
      MATCHED.set(node.id, matches(node) || (!flatFilter() && node.children.some(child => MATCHED.get(child.id))));
    }
    TIMELINE = [...ALL].sort(compareCompletion);
    TIMELINE.forEach((node, index) => { node.completionOrder = index + 1; });
    const DURATION_TIMELINE = [...ALL].sort((a, b) => b.duration - a.duration || a.order - b.order);
    DURATION_TIMELINE.forEach((node, index) => { node.durationOrder = index + 1; });
    if (ancestorSearch || $('#statusFilter').value === 'FINAL') expandMatchingTree();
    else if (autoExpandSubtree && (scopeRootId !== null || subtreeSearch) && !flatFilter()) {
      expandMatchingTree();
      for (const id of subtreeCollapsedIds) expandedIds.delete(id);
    }
  }

  function setNodeExpanded(id, open) {
    if (open) { expandedIds.add(id); subtreeCollapsedIds.delete(id); }
    else {
      expandedIds.delete(id);
      if (autoExpandSubtree) subtreeCollapsedIds.add(id);
    }
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
    view.focus = document.createElement('button');
    view.focus.type = 'button'; view.focus.className = 'subtree-focus';
    view.focus.textContent = '이 가지 보기';
    view.focus.addEventListener('click', event => {
      event.preventDefault(); event.stopPropagation();
      selectForAnalysis(NODE_BY_ID.get(view.id));
      changeScope(view.id, true);
    });
    summary.appendChild(view.focus);
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
      selectForAnalysis(node);
      acknowledgeGrowth(node);
      // Record the click synchronously, before an arriving snapshot can patch
      // this row. The native toggle event is otherwise delivered later.
      if (!flatFilter() && node.children.some(child => MATCHED.get(child.id))) {
        setNodeExpanded(view.id, !details.open);
      }
      patchView(view, node);
      saveState();
    });
    details.addEventListener('toggle', () => {
      if (view.disposed || flatFilter() || details.open === expandedIds.has(view.id)) return;
      acknowledgeGrowth(NODE_BY_ID.get(view.id));
      setNodeExpanded(view.id, details.open);
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
    view.focus.disabled = scopeRootId === node.id;
    view.focus.setAttribute('aria-label', `노드 #${node.id} 이 가지 보기`);
    setText(view.split, `Split #${node.splitNumber ?? '?'}`);
    setClass(view.badge, 'badge ' + color);
    setText(view.badge, resultLabel(node));
    setText(view.stable, node.stableAddedCount ? `STABLE ${node.stableAddedCount.toLocaleString()}개` : '');
    view.stable.hidden = !node.stableAddedCount;
    setText(view.duration, `${node.duration.toFixed(3)}s`);
    patchGrowth(view);
    setText(view.count, `하위 가지 ${(node.subtreeSize - 1).toLocaleString()}개`);
    if (view.summary.matches(':hover, :focus')) view.summary.title = tip(node);
    const hasChildren = !flatFilter() && node.children.some(child => MATCHED.get(child.id));
    view.details.classList.toggle('leaf', !hasChildren);
    view.details.classList.toggle('recent-path', recentPathIds.has(node.id));
    view.details.classList.toggle('recent-target', node.id === recentTargetId);
    view.details.classList.toggle('ancestor-target', ancestorSearch && ancestorTargetIds.has(node.id));
    view.details.classList.toggle('subtree-target', subtreeSearch && subtreeTargetIds.has(node.id));
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
    const desired = nodes.filter(node => MATCHED.get(node.id)).sort(compareTreeNodes);
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
  function renderScopeBar() {
    const root = NODE_BY_ID.get(scopeRootId);
    setText($('#scopeLabel'), root ? `현재 보는 가지: 노드 #${root.id} · ${neuronLabel(root)} (${root.phase})` : '전체 트리');
    setText($('#scopeCount'), root ? `하위 가지 ${(root.subtreeSize - 1).toLocaleString()}개` : `가지 ${DATA.nodeCount.toLocaleString()}개`);
    if (subtreeSearch) {
      setText($('#scopeLabel'), `서브 트리 검색 · L${$('#layerFilter').value}:N${$('#neuronFilter').value}`);
      setText($('#scopeCount'), `시작 노드 ${subtreeSearchRoots.length.toLocaleString()}개 · 가지 ${SCOPED.length.toLocaleString()}개`);
    }
    $('#scopeParent').disabled = !root;
    $('#scopeAll').disabled = !root && !subtreeSearch;
    $('#recentPath').title = root ? '선택한 가지 안에서 가장 최근에 추가된 노드까지 펼칩니다' : '루트부터 가장 최근에 추가된 가지까지 펼칩니다';
    $('#findAncestors').setAttribute('aria-pressed', String(ancestorSearch));
    $('#findSubtrees').setAttribute('aria-pressed', String(subtreeSearch));
    const targets = subtreeSearch ? subtreeTargetIds : ancestorTargetIds;
    setText($('#ancestorInfo'), ancestorSearch || subtreeSearch ? (targets.size ? `일치 ${targets.size.toLocaleString()}개 · ${subtreeSearch ? '하위 트리' : '상위 경로'} 표시 중` : '일치하는 탐색 노드가 없습니다.') : ancestorFeedback);
  }
  function renderTree() {
    $('#treeEmpty')?.remove();
    const flat = flatFilter();
    if (renderedFlat !== flat || renderedScopeRootId !== scopeRootId) {
      for (const view of rootViews.values()) dispose(view);
      rootViews.clear();
      renderedFlat = flat;
      renderedScopeRootId = scopeRootId;
    }
    renderScopeBar();
    reconcile($('#tree'), rootViews, flat ? SCOPED : scopeRoots());
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
        row.addEventListener('click',()=>selectForAnalysis(NODE_BY_ID.get(node.id),'ReLU 탐색 타임라인'));
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
    const layer = subtreeSearch ? '' : $('#layerFilter').value.trim(), neuron = subtreeSearch ? '' : $('#neuronFilter').value.trim();
    const records = (DATA.stableRecords || []).filter(record =>
      ((scopeRootId === null && !subtreeSearch) || scopeIds.has(record.nodeId ?? record.branchId)) &&
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
        row.addEventListener('click',event=>{
          if(event.target.closest('button'))return;
          const record=view.record,scope={topic:'tree'};
          if(Number.isInteger(record.roundIndex))scope.round=record.roundIndex;
          if(Number.isInteger(record.layer)&&Number.isInteger(record.neuron)){scope.layer=record.layer;scope.neuron=record.neuron;}
          if(window.parent!==window)window.parent.postMessage({type:'log-analysis-select',scope,text:`ReLU 탐색 트리 · STABLE\n${neuronLabel(record)}\n${stableSourceLabel(record.source)}\n${boundsLabel(record.before)} → ${boundsLabel(record.after)}`},'*');
        });
        view.range.className = 'range';
        view.location = document.createElement('button'); view.scope.appendChild(view.location);
        view.location.onclick = () => {
          const target = NODE_BY_ID.get(view.record.nodeId ?? view.record.branchId);
          if (target) { autoExpandSubtree = false; subtreeCollapsedIds.clear(); expandedIds.clear(); revealNodePath(target); }
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
  function captureScopeView() {
    return {
      expandedIds: [...expandedIds], recentPathIds: [...recentPathIds], recentTargetId,
      ancestorSearch, subtreeSearch,
      autoExpandSubtree, subtreeCollapsedIds: [...subtreeCollapsedIds],
      status: $('#statusFilter').value, treeOrder: $('#treeOrder').value, layer: $('#layerFilter').value, neuron: $('#neuronFilter').value,
      treeScrollLeft: $('#treeViewport').scrollLeft, treeScrollTop: $('#treeViewport').scrollTop,
      timelinePage, stablePage, timelineScroll: readPanelScroll('timeline'), stableScroll: readPanelScroll('stable'),
      pageScrollX: window.scrollX, pageScrollY: window.scrollY,
    };
  }
  function changeScope(id, expandSubtree = false) {
    if ((id === scopeRootId && !subtreeSearch) || (id !== null && !NODE_BY_ID.has(id))) return;
    const previous = captureScopeView();
    scopeViews.set(scopeViewKey(), previous);
    scopeRootId = id;
    const restored = expandSubtree ? {} : scopeViews.get(String(id)) || {};
    subtreeSearch = restored.subtreeSearch === true;
    autoExpandSubtree = expandSubtree || restored.autoExpandSubtree === true;
    subtreeCollapsedIds.clear();
    for (const nodeId of restored.subtreeCollapsedIds || []) subtreeCollapsedIds.add(nodeId);
    ancestorSearch = restored.ancestorSearch === true;
    ancestorFeedback = '';
    expandedIds.clear();
    recentPathIds.clear();
    for (const nodeId of restored.recentPathIds || []) recentPathIds.add(nodeId);
    recentTargetId = restored.recentTargetId ?? null;
    $('#statusFilter').value = statusOptions.includes(restored.status) ? restored.status : 'ALL';
    $('#treeOrder').value = treeOrders.includes(restored.treeOrder) ? restored.treeOrder : 'node';
    $('#layerFilter').value = restored.layer || '';
    $('#neuronFilter').value = restored.neuron || '';
    timelinePage = restored.timelinePage || 0; stablePage = restored.stablePage || 0;
    updateScopeIndex();
    for (const nodeId of restored.expandedIds || []) if (scopeIds.has(nodeId)) expandedIds.add(nodeId);
    if (expandSubtree) for (const node of SCOPED) if (node.children.length) expandedIds.add(node.id);
    updateFilterIndex();
    if (id !== null) acknowledgeGrowth(NODE_BY_ID.get(id));
    renderMetrics(); renderTree(); renderTimeline(); renderStableRecords();
    sizeTreeViewport();
    $('#treeViewport').scrollLeft = restored.treeScrollLeft || 0;
    $('#treeViewport').scrollTop = restored.treeScrollTop || 0;
    restorePanelScroll('timeline', restored.timelineScroll || 0);
    restorePanelScroll('stable', restored.stableScroll || 0);
    $('#treeViewport').focus({preventScroll: true});
    window.scrollTo(restored.pageScrollX ?? previous.pageScrollX, restored.pageScrollY ?? previous.pageScrollY);
    saveState();
  }
  function saveState() {
    scopeViews.set(scopeViewKey(), captureScopeView());
    window.name = STATE_PREFIX + JSON.stringify({
      path: stateKey, expandedIds: [...expandedIds], timelinePage, trackLatestPath,
      scopeRootId, scopeViews: Object.fromEntries(scopeViews),
      autoExpandSubtree, subtreeCollapsedIds: [...subtreeCollapsedIds],
      ancestorSearch, subtreeSearch,
      stablePage, stableScope: $('#stableScope').value, stableOrder: $('#stableOrder').value,
      reportPanels: Object.fromEntries(reportPanels.map(panel => [panel.key, panel.element.open])),
      timelineOrder:$('#timelineOrder').value, status: $('#statusFilter').value, treeOrder: $('#treeOrder').value,
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
  function applySnapshot(data, allowFollow = true) {
    if (!data || !Array.isArray(data.roots)) return;
    if (data.snapshotVersion != null && data.snapshotVersion === DATA.snapshotVersion) return;
    if(inputBusy){queuedSnapshot=data;return;}
    if (scrollActive() || scrollbarDragging) {queuedSnapshot=data;settleScroll();return;}
    const scrollX = window.scrollX, scrollY = window.scrollY, timelineScroll = readPanelScroll('timeline');
    const stableScroll = readPanelScroll('stable');
    const treeScrollLeft = $('#treeViewport').scrollLeft, treeScrollTop = $('#treeViewport').scrollTop;
    const previousLatest = latestNodeId;
    indexData(data);
    const follow = allowFollow && trackLatestPath && latestNodeId !== null && latestNodeId !== previousLatest;
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
  function finishSnapshotRequest(){
    if(snapshotTimeout!==null){clearTimeout(snapshotTimeout);snapshotTimeout=null;}
    snapshotPending=false;
    if(finalSnapshotQueued){finalSnapshotQueued=false;requestSnapshot(true);}
  }
  async function requestSnapshot(force=false) {
    if(inputBusy){queuedSnapshotRequest=true;finalSnapshotQueued=finalSnapshotQueued||force;return;}
    if(snapshotPending){finalSnapshotQueued=finalSnapshotQueued||force;return;}
    if(DATA.status!=='RUNNING'&&!force)return;
    if(scrollActive() || scrollbarDragging){queuedSnapshotRequest=true;finalSnapshotQueued=finalSnapshotQueued||force;settleScroll();return;}
    snapshotPending = true;
    const request=++snapshotRequest;
    snapshotTimeout=setTimeout(()=>{
      if(request!==snapshotRequest)return;
      snapshotRequest++;snapshotAsset.version=null;finishSnapshotRequest();
      if(DATA.status==='RUNNING')requestSnapshot();
    },6000);
    let update;
    try { update = await visualizationAssetUpdate(DATA.snapshotFile || location.href, snapshotAsset,force); }
    catch (_error) { if(request===snapshotRequest)finishSnapshotRequest();return; }
    if(request!==snapshotRequest)return;
    if (!update) { finishSnapshotRequest(); return; }
    if (!snapshotFrame) {
      snapshotFrame = document.createElement('iframe');
      snapshotFrame.hidden = true; snapshotFrame.id = 'treeSnapshotLoader';
      snapshotFrame.setAttribute('aria-hidden', 'true'); snapshotFrame.tabIndex = -1;
      snapshotFrame.addEventListener('load', finishSnapshotRequest);
      document.body.appendChild(snapshotFrame);
    }
    const url = new URL(update.url);
    url.searchParams.set('tree-snapshot', '1');
    requestedSnapshotVersion = update.version;
    snapshotFrame.src = url.href;
  }
  window.addEventListener('message', event => {
    const message = event.data || {};
    if(event.source===window.parent&&message.type==='visualization-input-activity'){
      inputBusy=message.active===true;
      if(!inputBusy){
        if(queuedSnapshot){const latest=queuedSnapshot;queuedSnapshot=null;applySnapshot(latest,false);}
        if(queuedSnapshotRequest){queuedSnapshotRequest=false;const force=finalSnapshotQueued;finalSnapshotQueued=false;requestSnapshot(force);}
      }
      return;
    }
    if (message.type === SNAPSHOT_TYPE && snapshotFrame && event.source === snapshotFrame.contentWindow) {
      if(message.data && Array.isArray(message.data.roots))snapshotAsset.version=requestedSnapshotVersion;
      applySnapshot(message.data);
    } else if (message.type === 'relu-search-tree-refresh' && window.parent !== window && event.source === window.parent) requestSnapshot(message.final===true);
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
  function findAncestorPaths() { findNeuronTrees(false); }
  function findSubtrees() { findNeuronTrees(true); }
  function findNeuronTrees(subtrees) {
    const layer = $('#layerFilter').value.trim(), neuron = $('#neuronFilter').value.trim();
    if (!layer || !neuron || ![layer, neuron].every(value => /^\d+$/.test(value) && Number.isSafeInteger(Number(value)))) {
      ancestorFeedback = !layer || !neuron ? '레이어와 뉴런 번호를 입력하세요.' : '0 이상의 정수 번호를 입력하세요.';
      renderScopeBar();
      return;
    }
    scopeViews.set(scopeViewKey(), captureScopeView());
    scopeRootId = null;
    $('#statusFilter').value = 'ALL';
    $('#layerFilter').value = String(Number(layer));
    $('#neuronFilter').value = String(Number(neuron));
    ancestorSearch = !subtrees; subtreeSearch = subtrees;
    ancestorFeedback = ''; trackLatestPath = false;
    autoExpandSubtree = subtrees; subtreeCollapsedIds.clear();
    timelinePage = 0; stablePage = 0;
    expandedIds.clear(); recentPathIds.clear(); recentTargetId = null;
    updateScopeIndex(); updateFilterIndex();
    for (const id of subtrees ? scopeIds : ancestorPathIds) BRANCH_GROWTH.delete(id);
    renderMetrics(); renderTree(); renderTimeline(); renderStableRecords();
    sizeTreeViewport();
    $('#treeViewport').scrollTo(0, 0);
    let target = null;
    const targets = subtrees ? subtreeTargetIds : ancestorTargetIds;
    for (const node of ALL) if (targets.has(node.id) && (!target || node.order > target.order)) target = node;
    const summary = views.get(target?.id)?.summary;
    if (summary) { summary.focus({preventScroll: true}); scrollTreeNode(summary, true); }
    else $('#treeViewport').focus({preventScroll: true});
    restorePanelScroll('timeline', 0); restorePanelScroll('stable', 0);
    saveState();
  }
  function scrollTreeNode(summary, center) {
    sizeTreeViewport();
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
    if (!target || !scopeIds.has(target.id)) return;
    if (ancestorSearch || flatFilter() || !MATCHED.get(target.id)) {
      ancestorSearch = false; ancestorFeedback = '';
      $('#statusFilter').value = 'ALL';
      $('#layerFilter').value = '';
      $('#neuronFilter').value = '';
      timelinePage = 0;
      updateFilterIndex(); renderTimeline(); renderStableRecords();
    }
    recentPathIds.clear();
    recentTargetId = target.id;
    for (let node = target; node && scopeIds.has(node.id) && !recentPathIds.has(node.id); node = NODE_BY_ID.get(node.parentId)) {
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
    panel.appliedOpen = panel.element.open;
    panel.element.querySelector(':scope > summary').addEventListener('click', event => {
      event.preventDefault();
      readPanelScroll(panel.key);
      panel.element.open = !panel.element.open;
      panel.appliedOpen = panel.element.open;
      if (panel.element.open) { panel.render(); panel.list.scrollTop = panel.scroll; }
      renderPanelToggle(); saveState();
    });
    panel.element.addEventListener('toggle', () => {
      if (panel.appliedOpen === panel.element.open) return;
      panel.appliedOpen = panel.element.open;
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
      panel.appliedOpen = open;
      if (open) { panel.render(); panel.list.scrollTop = panel.scroll; }
    }
    renderPanelToggle(); saveState();
  };
  $('#defaultView').onclick = () => {
    autoExpandSubtree = false; subtreeCollapsedIds.clear();
    if (ancestorSearch) {
      ancestorSearch = false; ancestorFeedback = '';
      $('#layerFilter').value = ''; $('#neuronFilter').value = '';
      updateFilterIndex(); renderTimeline(); renderStableRecords();
    }
    expandedIds.clear(); renderTree(); $('#treeViewport').scrollTo(0, 0); saveState();
  };
  $('#scopeAll').onclick = () => changeScope(null);
  $('#scopeParent').onclick = () => changeScope(NODE_BY_ID.get(scopeRootId)?.parentId ?? null);
  $('#recentPath').onclick = () => {
    autoExpandSubtree = false; subtreeCollapsedIds.clear();
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
    for (const node of SCOPED) {
      if (MATCHED.get(node.id) && node.children.some(child => MATCHED.get(child.id))) expandedIds.add(node.id);
    }
  }
  $('#expandAll').onclick = () => {
    autoExpandSubtree = scopeRootId !== null || subtreeSearch; subtreeCollapsedIds.clear();
    expandMatchingTree();
    renderTree(); saveState();
  };
  $('#collapseAll').onclick = () => { for (const root of scopeRoots()) setNodeExpanded(root.id, false); renderTree(); saveState(); };
  const filterChanged = () => { ancestorSearch = false; subtreeSearch = false; ancestorFeedback = ''; subtreeCollapsedIds.clear(); timelinePage = 0; stablePage = 0; updateScopeIndex(); updateFilterIndex(); expandMatchingTree(); renderTree(); $('#treeViewport').scrollTo(0, 0); renderTimeline(); renderStableRecords(); saveState(); };
  $('#statusFilter').onchange = filterChanged;
  $('#treeOrder').onchange = () => {
    updateFilterIndex(); renderTree();
    $('#treeViewport').scrollTo(0, 0); saveState();
  };
  $('#layerFilter').oninput = () => {syncSearchChoices();filterChanged();};
  $('#neuronFilter').oninput = filterChanged;
  $('#findAncestors').onclick = findAncestorPaths;
  $('#findSubtrees').onclick = findSubtrees;
  for (const input of [$('#layerFilter'), $('#neuronFilter')]) {
    input.addEventListener('keydown', event => {
      if (event.key === 'Enter') { event.preventDefault(); findAncestorPaths(); }
    });
  }
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
  window.addEventListener('resize', sizeTreeViewport);
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
  if (DATA.snapshotFile) setTimeout(requestSnapshot, 0);
})();
