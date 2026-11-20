(() => {
  'use strict';
  const INITIAL_DATA = __DATA__;
  const SNAPSHOT = 'solver-mechanisms-snapshot';
  if (new URLSearchParams(location.search).get('mechanism-snapshot') === '1') {
    window.parent.postMessage({type: SNAPSHOT, data: INITIAL_DATA}, '*');
    return;
  }
  const $ = selector => document.querySelector(selector);
  const panelNames = ['bounds', 'backjump', 'numerical', 'underapprox'];
  const enabled = new Set(INITIAL_DATA.panels || panelNames);
  if (enabled.size === 1) $('.cards').style.gridTemplateColumns = '1fr';
  $('.cards').style.gridTemplateRows=`repeat(${Math.ceil(enabled.size/2)},minmax(0,1fr))`;
  document.querySelectorAll('.card').forEach(card=>{
    const content=document.createElement('div'),inner=document.createElement('div');
    content.className='card-content';inner.className='card-content-inner';
    inner.append(...card.childNodes);content.append(inner);card.append(content);
  });
  let fitTask=null;
  const fitCards=()=>{
    if(fitTask!==null)return;
    fitTask=setTimeout(()=>{
      fitTask=null;
      const headerHeight=`${$('header').getBoundingClientRect().height}px`;
      const heights=[...document.querySelectorAll('.card:not([hidden]) .card-content-inner')].map(inner=>inner.getBoundingClientRect().height);
      const controlsHeight=`${innerWidth>850?Math.ceil(Math.max(0,...heights)):0}px`;
      const style=document.documentElement.style;
      if(style.getPropertyValue('--report-header-height')!==headerHeight)style.setProperty('--report-header-height',headerHeight);
      if(style.getPropertyValue('--mechanism-controls-height')!==controlsHeight)style.setProperty('--mechanism-controls-height',controlsHeight);
    },0);
  };
  new ResizeObserver(fitCards).observe($('header'));
  new ResizeObserver(fitCards).observe($('.cards'));
  window.addEventListener('resize',fitCards);
  $('#expandCharts').onclick=()=>{
    if(document.fullscreenElement){document.exitFullscreen();return;}
    if(document.documentElement.requestFullscreen){
      document.documentElement.requestFullscreen().catch(()=>window.open(location.href,'_blank'));
    }else window.open(location.href,'_blank');
  };
  document.addEventListener('fullscreenchange',()=>{
    $('#expandCharts').textContent=document.fullscreenElement?'원래 크기':'그림 크게 보기';
    fitCards();
  });
  document.querySelectorAll('.card').forEach((card, index) => { card.hidden = !enabled.has(panelNames[index]); });
  for (const option of [...$('#mechanismFilter').options]) {
    if (option.value !== 'all' && !enabled.has(option.value)) option.remove();
  }
  const STATE_PREFIX = 'solver-mechanisms-ui:';
  let saved = {};
  try { if (window.name.startsWith(STATE_PREFIX)) saved = JSON.parse(window.name.slice(STATE_PREFIX.length)); } catch (_error) {}
  if (saved.path !== location.pathname) saved = {};
  const recordsPanel=$('#occurrenceRecords');
  recordsPanel.open=saved.recordsOpen===true;
  if (saved.filter === 'all' || enabled.has(saved.filter)) $('#mechanismFilter').value = saved.filter;
  if (['drift','residual','pivot'].includes(saved.metric)) $('#numericalMetric').value = saved.metric;
  if (Number.isFinite(saved.epsilon) && saved.epsilon >= 0) $('#epsilon').value = saved.epsilon;
  if(typeof saved.layerSearch==='string')$('#boundLayerSearch').value=saved.layerSearch;
  if(typeof saved.neuronSearch==='string')$('#boundNeuronSearch').value=saved.neuronSearch;
  const previousNeuron=String(saved.neuron || '').match(/^([zh])(\d+)_(\d+)(?:_|$)/);
  if(previousNeuron && typeof saved.layerSearch!=='string'){
    $('#boundLayerSearch').value=previousNeuron[2];$('#boundNeuronSearch').value=`${Number(previousNeuron[3])} (${previousNeuron[1]})`;
  }
  let boundLayerChoices=[],boundNeuronChoices=[];
  const searchPickers=[];
  if (['stable','all'].includes(saved.boundScope)) $('#boundScope').value=saved.boundScope;
  let data = INITIAL_DATA, loader = null, pending = false, timer = null, finalQueued = false;
  let inputBusy=false,queuedData=null,queuedRefresh=false,resumeRecords=null;
  const snapshotAsset = {};
  let requestedSnapshotVersion = null;
  let recordsReport = null, recordsFilter = null, recordsGeneration = 0;
  const BOUND_INTERVAL_MS = 3000;
  let boundIndex = Number.isInteger(saved.boundIndex) ? saved.boundIndex : -1;
  let boundPlaying = saved.boundPlaying !== false, boundFollow = saved.boundFollow !== false, boundTimer = null;
  const searchSteps = Object.assign(Object.create(null), saved.searchSteps && typeof saved.searchSteps==='object' ? saved.searchSteps : {});
  const text = (selector, value) => { const element=$(selector);if(element.textContent!==String(value))element.textContent=value; };
  const reportCaches=new WeakMap();
  function reportCache(report){
    if(!reportCaches.has(report))reportCaches.set(report,{});
    return reportCaches.get(report);
  }
  const finite = value => typeof value === 'number' && Number.isFinite(value);
  const number = value => finite(value) ? Number(value.toPrecision(6)).toString() : '?';
  const limit = (value, lower) => value == null ? (lower ? '−∞' : '+∞') : number(value);
  const interval = values => `[${limit(values[0],true)}, ${limit(values[1],false)}]`;
  const sourceLabel = source => ({initial:'탐색 전 경계 유도',search:'탐색 중 경계 유도',pivot:'피벗 경계 강화','lp-pivot':'Simplex 피벗','lp-full':'Simplex 전체 경계 강화',restore:'복원 경계 되돌림'}[source] || source);
  const roleExplanation = item => (item.role || String(item.variable || '').charAt(0))==='h'?'ReLU 출력':'ReLU 적용 전';
  const neuronLabel = item => {
    const name=typeof item==='string'?item:item.variable || item.neuron || '';
    const match=String(name).match(/^([zh])(\d+)_(\d+)(?:_|$)/);
    if(match)return `L${Number(match[2])}:N${Number(match[3])} (${match[1]})`;
    const index=item?.index ?? (typeof item?.neuron==='number'?item.neuron:null);
    return item?.layer!=null && index!=null ? `L${item.layer}:N${index}${item.role?' ('+item.role+')':''}` : String(name || '?');
  };
  const svgElement = (name, attributes = {}, content) => {
    const element = document.createElementNS('http://www.w3.org/2000/svg', name);
    for (const [key, value] of Object.entries(attributes)) element.setAttribute(key, value);
    if (content != null) element.textContent = content;
    return element;
  };
  function selectForAnalysis(mechanism, record, label, detail='') {
    if(window.parent===window)return;
    const scope={topic:'mechanisms',mechanism};
    if(Number.isInteger(record?.round_index))scope.round=record.round_index;
    if(mechanism==='bounds' && record){
      const match=String(record.variable||record.neuron||'').match(/^[zh](\d+)_(\d+)(?:_|$)/);
      const layer=match?Number(match[1]):record.layer,neuron=match?Number(match[2]):record.index??record.neuron;
      if(Number.isInteger(layer)&&Number.isInteger(neuron)){scope.layer=layer;scope.neuron=neuron;}
    }
    const context=[label,record?.sequence!=null?'기록 #'+record.sequence:'',record?.id!=null?'변경 #'+record.id:'',
      record?.round_index!=null?'DPLL Round: '+record.round_index:'',record?.depth!=null?'탐색 깊이: '+record.depth:'',detail].filter(Boolean).join('\n');
    window.parent.postMessage({type:'log-analysis-select',scope,text:context.slice(0,1800)},'*');
  }
  function selectable(element, mechanism, record, label, detail) {
    element.style.cursor='pointer';element.setAttribute('tabindex','0');element.setAttribute('role','button');
    element.setAttribute('aria-label',label+' · 질문에 선택 정보 넣기');
    const select=event=>{event.stopPropagation();selectForAnalysis(mechanism,record,label,detail);};
    element.addEventListener('click',select);
    element.addEventListener('keydown',event=>{if(event.key==='Enter'||event.key===' '){event.preventDefault();select(event);}});
  }
  function chart(index, title) {
    const card = document.querySelectorAll('.card')[index];
    let host = card.querySelector('.chart');
    if (!host) { host = document.createElement('div'); host.className='chart'; card.append(host); }
    host.replaceChildren();
    const heading = document.createElement('div'); heading.className='chart-title'; heading.textContent=title;
    const svg=svgElement('svg',{viewBox:'0 0 560 230',role:'img','aria-label':title});
    host.append(heading,svg);
    const view={host,svg};
    host.style.cursor='pointer';
    host.onclick=()=>selectForAnalysis(['bounds','backjump','numeric','underapprox'][index],view.record,title,view.detail||card.querySelector('.value').textContent);
    return view;
  }
  function emptyChart(view, message) {
    view.svg.remove();
    const empty=document.createElement('div'); empty.className='chart-empty'; empty.textContent=message; view.host.append(empty);
  }
  function options(selector, entries, preferred) {
    const element=$(selector), previous=element.value || preferred;
    if (JSON.stringify(entries)!==element.dataset.entries) {
      element.replaceChildren();
      for (const [value,label] of entries) { const option=document.createElement('option');option.value=value;option.textContent=label;element.append(option); }
      element.dataset.entries=JSON.stringify(entries);
    }
    if (entries.some(([key])=>key===previous)) element.value=previous;
    element.disabled=!entries.length;
  }
  function boundRounds(report) {
    const cache=reportCache(report);if(cache.rounds)return cache.rounds;
    const rounds=[...(report.boundRounds || [])], known=new Set(rounds.map(item=>item.round_index));
    for(const item of [...(report.boundHistory || []),...(report.eventsByKind?.bounds || [])])if(!known.has(item.round_index)){known.add(item.round_index);rounds.push({round_index:item.round_index,ranges:[]});}
    return cache.rounds=rounds;
  }
  function displayBoundHistory(report) {
    if ($('#boundScope').value==='all') return report.boundHistory || [];
    const cache=reportCache(report);if(cache.stableHistory)return cache.stableHistory;
    return cache.stableHistory=(report.eventsByKind?.bounds || []).filter(record=>record.kind==='stable').map(record=>({
      ...record,id:record.sequence,role:'z',neuron:record.variable,
    }));
  }
  function selectedRanges(report) {
    const rounds=boundRounds(report), history=displayBoundHistory(report);
    options('#boundRound',rounds.map((item,index)=>[String(index),`R${item.round_index ?? '?'}`]),saved.round);
    const change=history[Math.max(0,Math.min(boundIndex,history.length-1))];
    if(boundFollow && change){const index=rounds.findIndex(item=>item.round_index===change.round_index);if(index>=0)$('#boundRound').value=String(index);}
    const cache=reportCache(report);
    const ranges=rounds[Number($('#boundRound').value)]?.ranges || (cache.emptyRanges ||= []);
    const round=rounds[Number($('#boundRound').value)]?.round_index;
    const catalogKey=JSON.stringify([$('#boundScope').value,round??null]);
    cache.catalogs ||= new Map();
    const parts=item=>{
      cache.neuronParts ||= new Map();
      if(cache.neuronParts.has(item.variable))return cache.neuronParts.get(item.variable);
      const match=String(item.variable).match(/^([zh])(\d+)_(\d+)(?:_|$)/);
      const value=match?{layer:Number(match[2]),index:Number(match[3]),role:match[1]}:null;
      cache.neuronParts.set(item.variable,value);return value;
    };
    if(!cache.catalogs.has(catalogKey)){
      const variables=new Map($('#boundScope').value==='all'?ranges.map(item=>[item.variable,item]):[]);
      for(const item of history)if(item.round_index===round)variables.set(item.variable,item);
      const records=[...variables.values()].filter(item=>parts(item)),byLayer=new Map();
      for(const item of records){const layer=parts(item).layer;if(!byLayer.has(layer))byLayer.set(layer,[]);byLayer.get(layer).push(item);}
      cache.catalogs.set(catalogKey,{records,byLayer,layers:[...byLayer.keys()].sort((a,b)=>a-b)});
    }
    const {records,byLayer,layers}=cache.catalogs.get(catalogKey);
    boundLayerChoices=layers.map(layer=>[String(layer),`Layer ${layer}`]);
    if(boundFollow && change){
      const current=parts(change);
      if(current){$('#boundLayerSearch').value=String(current.layer);$('#boundNeuronSearch').value=`${current.index} (${current.role})`;}
    }
    if(!$('#boundLayerSearch').value && layers.length)$('#boundLayerSearch').value=String(layers[0]);
    const layerText=$('#boundLayerSearch').value.trim();
    const candidates=layerText!==''?byLayer.get(Number(layerText))||[]:[];
    boundNeuronChoices=candidates.map(item=>[`${parts(item).index} (${parts(item).role})`,`N${parts(item).index} (${parts(item).role})`]);
    boundNeuronChoices=[...new Map(boundNeuronChoices.map(entry=>[entry[0],entry])).values()].sort((a,b)=>parseInt(a[0])-parseInt(b[0])||a[0].localeCompare(b[0]));
    const entries=(boundFollow?records:candidates).map(item=>[item.variable,neuronLabel(item)]);
    options('#boundNeuron',entries,saved.neuron);
    if(boundFollow && change)$('#boundNeuron').value=change.variable;
    else {
      const query=$('#boundNeuronSearch').value.trim().match(/^(\d+)(?:\s*\(([zh])\))?$/);
      if(query){const selected=candidates.find(item=>parts(item).index===Number(query[1])&&(!query[2]||parts(item).role===query[2]));$('#boundNeuron').value=selected?.variable || '';}
      else if(!$('#boundNeuronSearch').value.trim()){
        const selected=candidates.find(item=>item.variable===$('#boundNeuron').value);
        if(selected)$('#boundNeuronSearch').value=`${parts(selected).index} (${parts(selected).role})`;
      }
    }
    for(const picker of searchPickers)picker.render();
    options('#underNeuron',ranges.map(item=>[item.variable,neuronLabel(item)]),saved.underNeuron);
    return ranges;
  }
  function rangePlot(view, rows) {
    const values=rows.flatMap(row=>row.values).filter(finite);
    if (!values.length) { emptyChart(view,'유한한 경계 없음'); return; }
    const low=Math.min(...values), high=Math.max(...values), span=high-low || Math.max(1,Math.abs(low)*0.1);
    const min=low-span*0.12,max=high+span*0.12;
    const x=value=>155+365*(value-min)/(max-min);
    if (min<=0 && max>=0) view.svg.append(svgElement('line',{x1:x(0),x2:x(0),y1:12,y2:180,stroke:'#94a3b8','stroke-dasharray':'4 4'}),svgElement('text',{x:x(0),y:215,'text-anchor':'middle',fill:'#64748b'},'0'));
    rows.forEach((row,index)=>{
      const y=45+index*86, [lo,hi]=row.values;
      view.svg.append(svgElement('text',{x:0,y:y+5,fill:'#334155'},row.label));
      const line=svgElement('line',{x1:lo==null?145:x(lo),x2:hi==null?535:x(hi),y1:y,y2:y,stroke:row.color,'stroke-width':10,'stroke-linecap':'round'});
      line.append(svgElement('title',{},`${row.label} ${interval(row.values)}`));view.svg.append(line);
      for (const [value,isLower] of [[lo,true],[hi,false]]) {
        const cx=value==null?(isLower?145:535):x(value);
        view.svg.append(svgElement('circle',{cx,cy:y,r:6,fill:row.color}));
      }
      view.svg.append(svgElement('text',{x:340,y:y+34,'text-anchor':'middle',fill:'#475569'},interval(row.values)));
    });
  }
  function boundChart(report, ranges) {
    const history=displayBoundHistory(report);
    if (history.length) boundIndex=Math.max(0,Math.min(boundIndex,history.length-1));
    const change=boundFollow?history[boundIndex]:null;
    $('#boundFollowMode').setAttribute('aria-pressed',String(boundFollow));$('#boundSearchMode').setAttribute('aria-pressed',String(!boundFollow));
    $('#boundSearchControls').hidden=boundFollow;$('.bound-playback').hidden=!boundFollow;
    $('#boundTimeline').hidden=!boundFollow;
    $('#boundSearchPlayback').hidden=boundFollow;
    $('#boundStep').max=Math.max(1,history.length);$('#boundStep').value=Math.max(1,boundIndex+1);$('#boundStep').disabled=!history.length;
    $('#boundPrev').disabled=!history.length || boundIndex<=0;$('#boundNext').disabled=!history.length || boundIndex>=history.length-1;
    $('#boundPlay').disabled=!history.length;
    const complete=history.length && boundIndex===history.length-1 && data.status!=='RUNNING';
    text('#boundPending',`+${Math.max(0,history.length-boundIndex-1).toLocaleString()}개 재생 대기`);
    text('#boundPlay',complete && boundFollow?'다시 재생':boundPlaying && boundFollow?'일시정지':'재생');
    text('#boundPosition',history.length?`${boundIndex+1} / ${history.length} · ${!boundFollow?'수동 보기':complete?'재생 완료':'3초 간격'}`:'변경 기록 대기');
    if (change) {
      text('#boundsValue',`R${change.round_index ?? '?'} · 깊이 ${change.depth} · ${change.role || 'z'} (${roleExplanation(change)}) · 변경 #${change.id}`);
      const view=chart(0,`${neuronLabel(change)} · ${change.kind==='stable'?'STABLE · ':''}${sourceLabel(change.source)}`);
      view.record=change;view.detail=`경계 유도 · ${neuronLabel(change)}\n변경 전: ${interval(change.before)}\n변경 후: ${interval(change.after)}`;
      rangePlot(view,[{label:'변경 전',values:change.before,color:'#94a3b8'},{label:'변경 후',values:change.after,color:'#0284c7'}]);
      return;
    }
    if(!boundFollow){
      const {key,records}=searchBoundHistory(report);
      const stored=Number.isInteger(searchSteps[key])?searchSteps[key]:0;
      const index=Math.max(0,Math.min(stored,records.length-1));
      if(records.length)searchSteps[key]=index;
      $('#boundSearchStep').max=Math.max(1,records.length);$('#boundSearchStep').value=index+1;$('#boundSearchStep').disabled=records.length<=1;
      $('#boundSearchPrev').disabled=!records.length || index===0;$('#boundSearchNext').disabled=!records.length || index>=records.length-1;
      text('#boundSearchPosition',records.length?`${index+1} / ${records.length} · 변경 #${records[index].id}`:'0 / 0');
      const selectedChange=records[index];
      if(selectedChange){
        text('#boundsValue',`R${selectedChange.round_index ?? '?'} · ${neuronLabel(selectedChange)} (${roleExplanation(selectedChange)}) · 깊이 ${selectedChange.depth} · 변경 #${selectedChange.id}`);
        const view=chart(0,`${neuronLabel(selectedChange)} · ${selectedChange.kind==='stable'?'STABLE · ':''}${sourceLabel(selectedChange.source)}`);
        view.record=selectedChange;view.detail=`경계 유도 · ${neuronLabel(selectedChange)}\n변경 전: ${interval(selectedChange.before)}\n변경 후: ${interval(selectedChange.after)}`;
        rangePlot(view,[{label:'변경 전',values:selectedChange.before,color:'#94a3b8'},{label:'변경 후',values:selectedChange.after,color:'#0284c7'}]);return;
      }
    }
    if($('#boundScope').value==='stable') {emptyChart(chart(0,'STABLE 확정'),!boundFollow && $('#boundNeuronSearch').value.trim()?'일치하는 STABLE 뉴런 없음':'STABLE 확정 기록 없음');return;}
    const selected=ranges.find(item=>item.variable===$('#boundNeuron').value);
    const view=chart(0,selected ? `${neuronLabel(selected)} · ${{active:'활성 확정',inactive:'비활성 확정',unstable:'분기 가능'}[selected.state] || selected.state}` : '뉴런별 최솟값 · 최댓값');
    if (!selected) { emptyChart(view,!boundFollow && $('#boundNeuronSearch').value.trim()?'일치하는 뉴런 없음':'탐색 전 경계 유도 기록 없음'); return; }
    view.record=selected;view.detail=`경계 유도 · ${neuronLabel(selected)}\n유도 전: ${interval(selected.before)}\n유도 후: ${interval(selected.after)}`;
    rangePlot(view,[{label:'유도 전',values:selected.before,color:'#94a3b8'},{label:'유도 후',values:selected.after,color:'#0284c7'}]);
  }
  function searchBoundHistory(report) {
    const round=boundRounds(report)[Number($('#boundRound').value)]?.round_index, variable=$('#boundNeuron').value;
    const mode=$('#boundScope').value,cache=reportCache(report);
    cache.searchHistories ||= {};
    if(!cache.searchHistories[mode]){
      const histories=new Map();
      for(const record of displayBoundHistory(report)){
        const key=JSON.stringify([record.round_index??null,record.variable]);
        if(!histories.has(key))histories.set(key,[]);histories.get(key).push(record);
      }
      cache.searchHistories[mode]=histories;
    }
    return {key:JSON.stringify([mode,round ?? null,variable]),records:cache.searchHistories[mode].get(JSON.stringify([round??null,variable]))||[]};
  }
  function seekSearchBound(index) {
    const {key,records}=searchBoundHistory(data.report || {});
    if(!records.length)return;
    searchSteps[key]=Math.max(0,Math.min(index,records.length-1));render();save();
  }
  function scheduleBoundPlayback() {
    if (inputBusy || boundTimer!==null || !boundPlaying || !boundFollow || !enabled.has('bounds')) return;
    const history=displayBoundHistory(data.report || {});
    if (!history.length || (boundIndex>=history.length-1 && data.status!=='RUNNING')) return;
    boundTimer=setTimeout(()=>{
      boundTimer=null;
      const latest=displayBoundHistory(data.report || {});
      if (!inputBusy && boundPlaying && boundFollow && boundIndex<latest.length-1) {boundIndex++;render();save();}
      else scheduleBoundPlayback();
    },BOUND_INTERVAL_MS);
  }
  function pauseBounds() {
    boundPlaying=false;
    if (boundTimer!==null) {clearTimeout(boundTimer);boundTimer=null;}
  }
  function seekBound(index) {
    pauseBounds();boundFollow=true;boundIndex=index;render();save();
  }
  function conflictChart(report, totals) {
    const jumps=(report.eventsByKind?.backjump || []).filter(item=>item.kind==='backjump').slice(-10);
    const view=chart(1,jumps.length ? '백점프 깊이 · ● 출발 → ◆ 복귀' : '모순 발생 원인');
    if (!jumps.length) {
      const counts=[['LP (선형 제약 문제)',totals.conflict_LP || 0],['경계 유도',totals.conflict_DERIVE || 0],['변수 경계',totals.conflict_BOUNDS || 0]];
      if (!counts.some(item=>item[1])) {emptyChart(view,'충돌·백점프 발생 기록 없음');return;}
      const max=Math.max(...counts.map(item=>item[1]));
      counts.forEach(([label,count],index)=>{const y=24+index*62,group=svgElement('g');group.append(svgElement('text',{x:0,y:y+22,'font-size':14},label),svgElement('rect',{x:140,y,width:330*count/max,height:30,rx:4,fill:'#7c3aed'}),svgElement('text',{x:550,y:y+22,'text-anchor':'end'},count));selectable(group,'backjump',null,'모순 발생 원인',`${label}: ${count}건`);view.svg.append(group);});
      return;
    }
    const max=Math.max(1,...jumps.flatMap(item=>[item.from_depth,item.target_depth]));
    const y=depth=>32+145*depth/max;
    for (const depth of [...new Set([0,Math.round(max/2),max])]) view.svg.append(svgElement('line',{x1:50,x2:545,y1:y(depth),y2:y(depth),stroke:'#e2e8f0'}),svgElement('text',{x:40,y:y(depth)+6,'text-anchor':'end',fill:'#64748b'},depth));
    view.svg.append(svgElement('text',{x:0,y:16,fill:'#64748b'},'깊이 ↓'));
    jumps.forEach((item,index)=>{
      const x=jumps.length===1?300:75+index*440/(jumps.length-1), from=y(item.from_depth), to=y(item.target_depth);
      const group=svgElement('g');
      selectable(group,'backjump',item,'충돌 분석·백점프',`출발 깊이: ${item.from_depth}\n복귀 깊이: ${item.target_depth}\n원인 깊이: ${(item.cause_depths||[]).join(', ')}`);
      group.append(svgElement('title',{},`D${item.round_index ?? '?'} · #${item.sequence}: 깊이 ${item.from_depth} → ${item.target_depth} · 원인 [${(item.cause_depths || []).join(', ')}]`),svgElement('line',{x1:x,x2:x,y1:from,y2:to,stroke:'#7c3aed','stroke-width':4}),svgElement('circle',{cx:x,cy:from,r:6,fill:'#dc2626'}),svgElement('path',{d:`M ${x} ${to-8} l 7 8 l -7 8 l -7 -8 Z`,fill:'#7c3aed'}));
      group.append(svgElement('text',{x,y:218,'text-anchor':'middle',fill:'#475569'},'#'+item.sequence));view.svg.append(group);
    });
  }
  function logPlot(view, records, series, threshold) {
    const values=records.flatMap(record=>series.map(item=>record[item.key])).filter(finite);
    for (const item of series) if (['drift','residual'].includes(item.key)) values.push(...records.map(record=>record['after_'+item.key]).filter(finite));
    if (finite(threshold) && threshold>0) values.push(threshold);
    if (!records.length || !values.length) {emptyChart(view,records.some(record=>record.drift_nonfinite || record.residual_nonfinite)?'비유한 값 감지 · 복원 기록 확인':'측정 기록 없음');return;}
    const positive=values.filter(value=>value>0);
    const floor=Math.max(1e-300,Math.min(1e-16,positive.length?Math.min(...positive)/10:1e-16));
    const log=value=>Math.log10(Math.max(floor,value));
    const lower=Math.floor(Math.min(...values.map(log)))-1, upper=Math.max(lower+2,Math.ceil(Math.max(...values.map(log)))+1);
    const y=value=>180-145*(log(value)-lower)/(upper-lower), x=index=>records.length===1?305:85+index*445/(records.length-1);
    for (const exponent of [lower,Math.round((lower+upper)/2),upper]) {
      const cy=180-145*(exponent-lower)/(upper-lower);
      view.svg.append(svgElement('line',{x1:80,x2:540,y1:cy,y2:cy,stroke:'#e2e8f0'}),svgElement('text',{x:72,y:cy+5,'text-anchor':'end',fill:'#64748b'},exponent===lower?'≤1e'+exponent:'1e'+exponent));
    }
    if (finite(threshold)) view.svg.append(svgElement('line',{x1:80,x2:540,y1:y(threshold),y2:y(threshold),stroke:'#dc2626','stroke-dasharray':'6 4'}),svgElement('text',{x:535,y:y(threshold)-8,'text-anchor':'end',fill:'#dc2626'},'M='+number(threshold)));
    series.forEach((item,seriesIndex)=>{
      const points=records.map((record,index)=>finite(record[item.key])?`${x(index)},${y(record[item.key])}`:null);
      let run=[];
      const flush=()=>{if(run.length)view.svg.append(svgElement('polyline',{points:run.join(' '),fill:'none',stroke:item.color,'stroke-width':2}));run=[];};
      points.forEach(point=>{if(point)run.push(point);else flush();});flush();
      records.forEach((record,index)=>{
        if (!finite(record[item.key])) {
          if (record[item.key+'_nonfinite']) {
            const cx=x(index), marker=svgElement('path',{d:`M ${cx} 28 l 7 12 l -14 0 Z`,fill:'#dc2626'});
            marker.append(svgElement('title',{},`#${record.sequence} · 비유한 ${item.label}`));view.svg.append(marker);
            selectable(marker,'numeric',record,'부동소수점 연산',`${item.label}: 비유한 값`);
          }
          return;
        }
        const cx=x(index),cy=y(record[item.key]);
        const point=svgElement('circle',{cx,cy,r:4,fill:record.outcome==='restored'?'#16a34a':item.color});
        point.append(svgElement('title',{},`#${record.sequence} · ${item.label}: ${number(record[item.key])}${record.outcome==='restored'?' · 타블로 복원':''}`));view.svg.append(point);
        selectable(point,'numeric',record,'부동소수점 연산',`${item.label}: ${number(record[item.key])}\n${describe(record)}`);
        const after=record['after_'+item.key];
        if (record.outcome==='restored' && seriesIndex===0 && finite(after)) {
          const restored=svgElement('g');restored.append(svgElement('line',{x1:cx,x2:cx,y1:cy,y2:y(after),stroke:'#16a34a','stroke-width':3}),svgElement('path',{d:`M ${cx} ${y(after)-6} l 6 6 l -6 6 l -6 -6 Z`,fill:'#16a34a'}));
          selectable(restored,'numeric',record,'부동소수점 연산 · 타블로 복원',`${item.label}: ${number(record[item.key])} → ${number(after)}\n${describe(record)}`);view.svg.append(restored);
        }
      });
      view.svg.append(svgElement('text',{x:85+seriesIndex*225,y:18,fill:item.color},item.label));
    });
    view.svg.append(svgElement('text',{x:85,y:218,fill:'#64748b'},'#'+records[0].sequence),svgElement('text',{x:535,y:218,'text-anchor':'end',fill:'#64748b'},'기록 #'+records[records.length-1].sequence));
  }
  function numericalChart(report) {
    const metric=$('#numericalMetric').value, checks=report.numericalChecks || [];
    const settings=(report.settings || []).at(-1) || {};
    const view=chart(2,metric==='pivot'?'선택 계수 · 제외 기준 (log₁₀)':metric==='residual'?'초기 등식 잔차 합 · ◆ 복원 후 (log₁₀)':'상대 계수 오차 · ◆ 복원 후 (log₁₀)');
    if (metric==='pivot') logPlot(view,report.numericalPivots || [],[{key:'coefficient',label:'선택 |a|',color:'#0284c7'},{key:'zero_threshold',label:'제외 기준',color:'#d97706'}]);
    else logPlot(view,checks,[{key:metric==='residual'?'residual':'drift',label:metric==='residual'?'Σ |원래 등식 잔차|':'상대 계수 오차',color:'#0284c7'}],metric==='drift' ? settings.drift_threshold : undefined);
  }
  function underChart(ranges) {
    const selected=ranges.find(item=>item.variable===$('#underNeuron').value);
    const view=chart(3,`${selected?neuronLabel(selected)+' · ':''}[l, u] → [l + ε, u − ε]`);
    const epsilon=Number($('#epsilon').value);
    view.record=selected;view.detail=`과소근사 구간 미리보기 (실제 솔버 적용 결과가 아닙니다.)\n${selected?neuronLabel(selected)+'\n원래 구간: '+interval(selected.after):'선택한 뉴런 없음'}\nε: ${epsilon}`;
    if (!Number.isFinite(epsilon) || epsilon<0 || $('#epsilon').value==='') {emptyChart(view,'ε은 0 이상의 유한한 값이어야 합니다.');return;}
    if (!selected) {emptyChart(view,'비교할 뉴런 경계 없음 · 솔버 미적용');return;}
    const [lo,hi]=selected.after;
    if (!finite(lo) || !finite(hi)) {emptyChart(view,'유한한 양쪽 경계가 필요합니다.');return;}
    if (lo+epsilon>hi-epsilon) {rangePlot(view,[{label:'원래 구간',values:[lo,hi],color:'#94a3b8'}]);view.svg.append(svgElement('text',{x:0,y:150,fill:'#dc2626'},'축소 후 공집합 (l + ε > u − ε)'));return;}
    rangePlot(view,[{label:'원래 구간',values:[lo,hi],color:'#94a3b8'},{label:'ε 축소',values:[lo+epsilon,hi-epsilon],color:'#16a34a'}]);
  }
  function describe(record) {
    if (record.kind==='stable') return `STABLE · ${neuronLabel(record)} · ${record.state==='active'?'활성 확정':record.state==='inactive'?'비활성 확정':record.state} · ${sourceLabel(record.source)}${record.before && record.after?' · '+interval(record.before)+' → '+interval(record.after):''}`;
    if (record.kind==='bounds') return `${sourceLabel(record.source)}: 경계 변경 ${record.changed || 0}개 · STABLE ${record.fixed || 0}개${record.conflict?' · 모순':''}`;
    if (record.kind==='simplex') return `LP 경계 강화 ${record.pivot_tightened || 0}회 · 전체 강화 ${record.full_tighten || 0}회 · 경계 모순 ${record.bound_conflicts || 0}회`;
    if (record.kind==='backjump') return `BACKJUMP: 깊이 ${record.from_depth} → ${record.target_depth} · 원인 깊이 [${(record.cause_depths || []).join(', ')}] · 원인 노드 [${(record.causeNodeIds || []).map(id=>'#'+id).join(', ')}]${record.targetNodeId!=null?' · 복귀 노드 #'+record.targetNodeId:''}`;
    if (record.kind==='conflict') return `모순 발견: ${{LP:'Simplex',DERIVE:'경계 유도',BOUNDS:'변수 경계'}[record.source] || record.source} · 원인 깊이 [${(record.cause_depths || []).join(', ')}]`;
    if (record.kind==='numerical') return record.action==='pivot'
      ? `피벗 ${record.pivot} · 선택 |a|=${number(record.coefficient)} · 제외 기준=${number(record.zero_threshold)} · 작은 계수 제외 ${record.excluded_small || 0}개 · ${record.mode}`
      : `오차 검사 · 계보 피벗 ${record.lineage_pivots} · 상대 오차 ${number(record.drift)} · 잔차 합 ${number(record.residual)} · ${{restored:'타블로 복원',within_tolerance:'허용 범위',restore_failed:'복원 실패'}[record.outcome] || record.outcome}${record.outcome==='restored'?' · 복원 후 '+number(record.after_drift):''}`;
    return record.kind;
  }
  function save() {
    window.name=STATE_PREFIX+JSON.stringify({path:location.pathname,filter:$('#mechanismFilter').value,scroll:$('#events').scrollTop,pageY:window.scrollY,recordsOpen:recordsPanel.open,
      boundIndex,boundPlaying,boundFollow,searchSteps,layerSearch:$('#boundLayerSearch').value,neuronSearch:$('#boundNeuronSearch').value,boundScope:$('#boundScope').value,
      round:$('#boundRound').value,neuron:$('#boundNeuron').value,underNeuron:$('#underNeuron').value,metric:$('#numericalMetric').value,epsilon:Number($('#epsilon').value)});
  }
  function renderRecords(report) {
    if(!recordsPanel.open){recordsGeneration++;recordsReport=null;return;}
    const filter=$('#mechanismFilter').value;
    if(recordsReport===report && recordsFilter===filter)return;
    recordsReport=report;recordsFilter=filter;
    const generation=++recordsGeneration;
    const byKind=Object.fromEntries(Object.entries(report.eventsByKind || {}).filter(([key])=>enabled.has(key)));
    const records=filter==='all'?[...new Map(Object.values(byKind).flat().map(record=>[record.sequence,record])).values()].sort((a,b)=>a.sequence-b.sequence):byKind[filter] || [];
    const container=$('#events'),scroll=container.scrollTop,pageY=window.scrollY;container.replaceChildren();
    let offset=0,restoringScroll=true,lastRestoredScroll=0,restoringPage=true,lastPageY=window.scrollY;
    function appendBatch() {
      if(generation!==recordsGeneration)return;
      if(inputBusy){resumeRecords=appendBatch;return;}
      const fragment=document.createDocumentFragment(),end=Math.min(offset+100,records.length);
      for(;offset<end;offset++) {
        const record=records[offset],row=document.createElement('div');row.className='event';
        const position=document.createElement('strong');
        position.textContent=`기록 #${record.sequence} · D${record.round_index ?? '?'} · ${record.nodeId!=null?'노드 #'+record.nodeId:'루트 처리'} · 깊이 ${record.depth}`;
        const content=document.createElement('div');content.textContent=describe(record);row.append(position,content);fragment.append(row);
        const mechanism=['stable','bounds','bound_updates','simplex'].includes(record.kind)?'bounds':['conflict','backjump'].includes(record.kind)?'backjump':record.kind==='underapprox'?'underapprox':'numeric';
        selectable(row,mechanism,record,'Mechanisms · 발생 기록',describe(record));
      }
      container.append(fragment);
      if(restoringScroll && container.scrollTop!==lastRestoredScroll)restoringScroll=false;
      if(restoringScroll && scroll){container.scrollTop=scroll;lastRestoredScroll=container.scrollTop;}
      if(restoringPage && window.scrollY!==lastPageY)restoringPage=false;
      if(restoringPage && pageY){window.scrollTo(window.scrollX,pageY);lastPageY=window.scrollY;}
      if(offset<records.length)setTimeout(appendBatch,0);
    }
    appendBatch();
    if(!records.length)container.textContent=filter==='underapprox'?'과소근사는 솔버에 적용되어 있지 않습니다.':'해당 발생 기록이 없습니다.';
    text('#eventCount',`표시 ${records.length}건`);
  }
  let conflictReport=null,numericReport=null,numericMetric=null,underRanges=null,underNeuron=null,underEpsilon=null;
  function render() {
    document.body.setAttribute('aria-busy',String(Boolean(data.deferredData)));
    const report=data.report || {}, totals=report.totals || {}, ranges=selectedRanges(report);
    const states={active:0,inactive:0,unstable:0};ranges.forEach(item=>{if(item.state in states)states[item.state]++;});
    text('#title',data.title);text('#status',`실행 상태: ${data.status}`);
    text('#boundsValue',$('#boundScope').value==='stable'?`STABLE 확정 ${displayBoundHistory(report).length}건`:ranges.length?`뉴런 ${ranges.length}개 · 활성 ${states.active} · 비활성 ${states.inactive} · 불안정 ${states.unstable}`:'탐색 전 경계 기록 없음');
    text('#backjumpValue',(report.settings || []).length?`${report.settings.some(item=>item.backjump)?'백점프 활성':'백점프 비활성'} · 충돌 ${totals.conflict || 0}건 · 백점프 ${totals.backjump || 0}회`:'설정 기록 대기');
    text('#numericalValue',report.available?`오차 검사 ${totals.drift_checks || 0}회 · 복원 ${totals.restores || 0}회${totals.restore_failures?' · 복원 실패 '+totals.restore_failures+'회':''}`:'측정 기록 대기');
    const underapproxEnabled=(report.settings || []).some(item=>item.underapprox_enabled);
    text('#underapproxValue',underapproxEnabled?'솔버 과소근사 활성 · 아래 그림은 구간 미리보기':'솔버 미적용 · 구간 미리보기');
    boundChart(report,ranges);
    if(conflictReport!==report){conflictChart(report,totals);conflictReport=report;}
    const metric=$('#numericalMetric').value;
    if(numericReport!==report||numericMetric!==metric){numericalChart(report);numericReport=report;numericMetric=metric;}
    const neuron=$('#underNeuron').value,epsilon=$('#epsilon').value;
    if(underRanges!==ranges||underNeuron!==neuron||underEpsilon!==epsilon){underChart(ranges);underRanges=ranges;underNeuron=neuron;underEpsilon=epsilon;}
    fitCards();
    renderRecords(report);
    if(data.status!=='RUNNING' && timer!==null){clearInterval(timer);timer=null;}
    scheduleBoundPlayback();
    if(window.parent!==window)window.parent.postMessage({type:'relu-search-tree-status',status:data.status},'*');
  }
  async function refresh(force=false) {
    if(inputBusy){queuedRefresh=true;finalQueued=finalQueued||force;return;}
    if(pending){finalQueued=finalQueued || force;return;}if(data.status!=='RUNNING' && !force)return;
    pending=true;
    let update;
    try{update=await visualizationAssetUpdate(data.snapshotFile || location.href,snapshotAsset,force);}
    catch(_error){pending=false;if(finalQueued){finalQueued=false;refresh(true);}return;}
    if(!update){pending=false;if(finalQueued){finalQueued=false;refresh(true);}return;}
    if(!loader){loader=document.createElement('iframe');loader.id='snapshotLoader';loader.hidden=true;
      loader.onload=()=>{pending=false;if(finalQueued){finalQueued=false;refresh(true);}};document.body.append(loader);}
    const url=new URL(update.url);url.searchParams.set('mechanism-snapshot','1');requestedSnapshotVersion=update.version;loader.src=url.href;
  }
  function applySnapshot(payload){
    if(inputBusy){queuedData=payload;return;}
    if(payload.snapshotVersion!=null && payload.snapshotVersion===data.snapshotVersion)return;
    const pageY=window.scrollY;data=payload;render();window.scrollTo(window.scrollX,pageY);save();
  }
  window.addEventListener('message',event=>{
    if(event.source===window.parent&&event.data?.type==='visualization-input-activity'){
      inputBusy=event.data.active===true;
      if(!inputBusy){
        if(queuedData){const latest=queuedData;queuedData=null;applySnapshot(latest);}
        if(resumeRecords){const append=resumeRecords;resumeRecords=null;setTimeout(append,0);}
        scheduleBoundPlayback();
        if(queuedRefresh){queuedRefresh=false;const force=finalQueued;finalQueued=false;refresh(force);}
      }
      return;
    }
    if(loader && event.source===loader.contentWindow && event.data?.type===SNAPSHOT){
      pending=false;if(!event.data.data || typeof event.data.data.report!=='object')return;
      snapshotAsset.version=requestedSnapshotVersion;
      applySnapshot(event.data.data);
    }else if(event.source===window.parent && event.data?.type==='relu-search-tree-refresh')refresh(event.data.final===true);
  });
  $('#mechanismFilter').onchange=()=>{$('#events').scrollTop=0;render();save();};
  recordsPanel.querySelector(':scope > summary').addEventListener('click',event=>{
    event.preventDefault();recordsPanel.open=!recordsPanel.open;
    renderRecords(data.report || {});save();
  });
  recordsPanel.addEventListener('toggle',()=>{renderRecords(data.report || {});save();});
  for(const selector of ['#boundRound','#boundNeuron','#underNeuron','#numericalMetric','#epsilon']) $(selector).addEventListener('change',()=>{
    if(selector==='#boundRound' || selector==='#boundNeuron'){pauseBounds();boundFollow=false;}
    render();save();
  });
  $('#boundPrev').onclick=()=>seekBound(boundIndex-1);
  $('#boundNext').onclick=()=>{
    if(boundTimer!==null){clearTimeout(boundTimer);boundTimer=null;}
    boundFollow=true;boundIndex=Math.min(boundIndex+1,Math.max(0,displayBoundHistory(data.report || {}).length-1));
    render();save();
  };
  $('#boundStep').oninput=()=>seekBound(Number($('#boundStep').value)-1);
  $('#boundPlay').onclick=()=>{
    const count=displayBoundHistory(data.report || {}).length;
    if(boundIndex===count-1 && data.status!=='RUNNING'){boundIndex=0;boundPlaying=true;boundFollow=true;}
    else if(boundPlaying && boundFollow)pauseBounds();
    else {boundPlaying=true;boundFollow=true;}
    render();save();
  };
  $('#boundFollowMode').onclick=()=>{boundFollow=true;boundPlaying=true;render();save();};
  $('#boundSearchMode').onclick=()=>{pauseBounds();boundFollow=false;render();save();};
  $('#boundLayerSearch').oninput=()=>{pauseBounds();boundFollow=false;$('#boundNeuronSearch').value='';render();save();};
  $('#boundNeuronSearch').oninput=()=>{pauseBounds();boundFollow=false;render();save();};
  $('#boundScope').onchange=()=>{
    if(boundTimer!==null){clearTimeout(boundTimer);boundTimer=null;}
    boundIndex=-1;render();save();
  };
  $('#boundSearchStep').oninput=()=>seekSearchBound(Number($('#boundSearchStep').value)-1);
  $('#boundSearchPrev').onclick=()=>{const {key}=searchBoundHistory(data.report || {});seekSearchBound((searchSteps[key] || 0)-1);};
  $('#boundSearchNext').onclick=()=>{const {key}=searchBoundHistory(data.report || {});seekSearchBound((searchSteps[key] || 0)+1);};
  $('#events').addEventListener('scroll',save,{passive:true});window.addEventListener('scroll',save,{passive:true});
  for(const [id,choices] of [['boundLayerSearch',()=>boundLayerChoices],['boundNeuronSearch',()=>boundNeuronChoices]]){
    const input=$('#'+id),wrapper=document.createElement('span'),button=document.createElement('button'),popup=document.createElement('div');
    wrapper.className='mechanism-combo';button.type='button';button.textContent='▾';button.setAttribute('aria-label',id==='boundLayerSearch'?'전체 레이어 목록':'해당 레이어 기록 있는 뉴런 목록');
    popup.id=id+'Choices';popup.className='mechanism-choices';popup.hidden=true;popup.setAttribute('role','listbox');
    input.setAttribute('role','combobox');input.setAttribute('aria-controls',popup.id);input.setAttribute('aria-expanded','false');input.setAttribute('autocomplete','off');
    input.before(wrapper);wrapper.append(input,button,popup);
    const close=()=>{popup.hidden=true;input.setAttribute('aria-expanded','false');};
    const renderChoices=()=>{
      if(popup.hidden)return;
      popup.replaceChildren(...choices().map(([value,label])=>{
        const option=document.createElement('button');option.type='button';option.textContent=label;option.dataset.value=value;option.setAttribute('role','option');
        option.onclick=()=>{input.value=value;input.dispatchEvent(new Event('input',{bubbles:true}));close();input.focus({preventScroll:true});};return option;
      }));
      if(!popup.children.length){const empty=document.createElement('span');empty.className='muted';empty.textContent='기록 없음';popup.append(empty);}
    };
    const open=()=>{for(const picker of searchPickers)picker.close();popup.hidden=false;input.setAttribute('aria-expanded','true');renderChoices();popup.scrollTop=0;};
    input.addEventListener('click',open);button.onclick=()=>popup.hidden?open():close();
    input.addEventListener('keydown',event=>{if(event.key==='ArrowDown'){event.preventDefault();open();popup.querySelector('button')?.focus();}else if(event.key==='Escape')close();});
    popup.addEventListener('keydown',event=>{
      if(event.key==='Escape'){close();input.focus();return;}
      if(!['ArrowDown','ArrowUp'].includes(event.key))return;
      event.preventDefault();const options=[...popup.querySelectorAll('button')],index=options.indexOf(document.activeElement);
      options[Math.max(0,Math.min(options.length-1,index+(event.key==='ArrowDown'?1:-1)))]?.focus();
    });
    searchPickers.push({wrapper,close,render:renderChoices});
  }
  document.addEventListener('click',event=>{for(const picker of searchPickers)if(!picker.wrapper.contains(event.target))picker.close();});
  render();fitCards();$('#events').scrollTop=saved.scroll || 0;window.scrollTo(0,saved.pageY || 0);
  if(window.parent===window && data.status==='RUNNING')timer=setInterval(refresh,3000);
  if(data.snapshotFile)setTimeout(()=>refresh(true),0);
})();
