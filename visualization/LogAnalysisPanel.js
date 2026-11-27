(() => {
  const panel=document.getElementById('llm-analysis-panel');
  if(!panel || panel.dataset.ready) return;
  panel.dataset.ready='1';
  const $=id=>document.getElementById(id);
  const main=panel.parentElement;
  if(main?.tagName==='MAIN'){
    let graphs=main.querySelector(':scope > .dashboard-graphs');
    if(!graphs){
      // Saved dashboards have their graph sections directly in <main>.
      const sections=[...main.querySelectorAll(':scope > section')].filter(section=>section!==panel);
      if(sections.some(section=>section.querySelector('img[id$="-dashboard"], iframe[id$="-dashboard"]'))){
        graphs=document.createElement('div');
        graphs.className='dashboard-graphs';
        main.prepend(graphs);
        graphs.append(...sections);
      }
    }
    if(graphs){
      main.classList.add('dashboard-layout');
      for(const image of graphs.querySelectorAll('img[id$="-dashboard"]')){
        const updateAspect=()=>{
          if(image.naturalWidth&&image.naturalHeight){
            image.style.setProperty('--graph-aspect', image.naturalWidth/image.naturalHeight);
          }
        };
        image.addEventListener('load',updateAspect);
        updateAspect();
      }
      const header=document.querySelector('body > header');
      if(header){
        const syncHeader=()=>main.style.setProperty('--dashboard-header-height',
          (getComputedStyle(header).position==='sticky'?Math.ceil(header.getBoundingClientRect().height):0)+'px');
        syncHeader();
        new ResizeObserver(syncHeader).observe(header);
      }
    }
  }
  let inputBusy=false,inputComposing=false,inputTimer=null;
  const editingTarget=element=>element?.matches?.('#llm-question,#llm-call');
  window.logAnalysisInputBusy=()=>inputBusy;
  function announceInput(active){
    if(inputBusy===active)return;
    inputBusy=active;
    window.dispatchEvent(new CustomEvent('log-analysis-input-activity',{detail:{active}}));
    for(const id of ['relu-tree-dashboard','solver-mechanisms-dashboard']){
      const frame=$(id);if(frame?.tagName==='IFRAME')frame.contentWindow.postMessage({type:'visualization-input-activity',active},'*');
    }
  }
  function finishInput(){
    inputTimer=null;if(inputComposing)return;
    announceInput(false);
  }
  function noteInput(event){
    if(!editingTarget(event.target))return;
    if(inputTimer!==null)clearTimeout(inputTimer);
    announceInput(true);
    inputTimer=inputComposing?null:setTimeout(finishInput,700);
  }
  for(const name of ['focusin','pointerdown','beforeinput','input','keydown','paste'])panel.addEventListener(name,noteInput,{capture:true});
  panel.addEventListener('compositionstart',event=>{if(editingTarget(event.target)){inputComposing=true;noteInput(event);}},true);
  panel.addEventListener('compositionend',event=>{if(editingTarget(event.target)){inputComposing=false;noteInput(event);}},true);
  panel.addEventListener('focusout',event=>{
    if(!editingTarget(event.target))return;
    setTimeout(()=>{if(!editingTarget(document.activeElement)){inputComposing=false;if(inputTimer!==null)clearTimeout(inputTimer);finishInput();}},0);
  });
  const consent=$('llm-consent'),consentKey='log-analysis-consent:v1:'+location.pathname;
  consent.checked=false;
  try{consent.checked=sessionStorage.getItem(consentKey)==='1';}catch(_error){}
  let info=null, revision=0, busy=false,analysisText='',baseline=null,infoBusy=false;
  let analysisJobId=null, analysisStopping=false;
  let infoTimer=null,pendingInfo=null;
  const icons={analyze:['M12 3v4m0 10v4M3 12h4m10 0h4','m5.6 5.6 2.8 2.8m7.2 7.2 2.8 2.8m0-12.8-2.8 2.8m-7.2 7.2-2.8 2.8','m12 8 4 4-4 4-4-4Z'],
    history:['M3 11a9 9 0 1 1 2.6 7.4','M3 4v7h7','M12 7v5l3 2'],
    copy:['M9 9h11v11H9Z','M15 5V3H3v12h2'],external:['M14 3h7v7','m21 3-11 11','M10 3H3v18h18v-7']};
  function buttonLabel(button,text,icon){
    const svg=document.createElementNS('http://www.w3.org/2000/svg','svg');
    svg.setAttribute('viewBox','0 0 24 24');svg.setAttribute('aria-hidden','true');svg.setAttribute('focusable','false');svg.classList.add('llm-button-icon');
    for(const d of icons[icon]){const path=document.createElementNS(svg.namespaceURI,'path');path.setAttribute('d',d);svg.append(path);}
    const label=document.createElement('span');label.textContent=text;button.replaceChildren(svg,label);
  }
  for(const [id,icon] of [['llm-start','analyze'],['llm-reload','history'],['llm-copy','copy'],['llm-quota-link','external']]){
    const button=$(id);if(button)buttonLabel(button,button.textContent,icon);
  }
  const statusLabels={complete:'분석 완료',api_error:'API 오류',incomplete:'일부 응답',budget_exhausted:'호출 상한 도달',no_data:'기록 없음',blocked:'응답 차단'};
  function clearReport(message){
    baseline=null;showSnapshot();
    analysisText='';$('llm-result').textContent=message;$('llm-copy').disabled=true;$('llm-fullscreen').disabled=true;
    $('llm-result-tag').textContent='대기';delete $('llm-result-tag').dataset.status;
  }
  const topic=$('llm-topic'),round=$('llm-round'),layer=$('llm-layer'),neuron=$('llm-neuron'),call=$('llm-call');
  const topicPanels=[['dpll-theory-dashboard','dpll'],['simplex-dashboard','simplex'],['relu-dashboard','relu'],['relu-tree-dashboard','tree'],['solver-mechanisms-dashboard','mechanisms']];
  let citationTargets={}, citationRequest=0, pendingCitation=null, citationTimer;
  function citationDetail(target){
    const interval=values=>Array.isArray(values)?`[${values[0]??'−∞'}, ${values[1]??'+∞'}]`:'';
    return [window.formatLogAnalysisCitationLabel(target),
      target.result,Number.isFinite(target.seconds)?`${Number(target.seconds.toPrecision(4))}초`:'',
      target.before&&target.after?`${interval(target.before)} → ${interval(target.after)}`:''].filter(Boolean).join(' · ');
  }
  panel.addEventListener('click',event=>{
    const button=event.target.closest('button[data-citation-ref]');
    if(!button)return;
    const target=Object.hasOwn(citationTargets,button.dataset.citationRef)
      && citationTargets[button.dataset.citationRef][Number(button.dataset.citationIndex)];
    if(!target)return;
    const frame=$(topicPanels.find(([,topic])=>topic===target.topic)?.[0]);
    clearTimeout(citationTimer);
    pendingCitation=null;
    if(!frame){
      const footer=button.closest('.llm-references');
      const detail=footer.querySelector('output')||document.createElement('output');
      detail.textContent=citationDetail(target);footer.append(detail);
      return;
    }
    button.closest('dialog')?.close();
    frame.scrollIntoView({block:'start'});
    $('llm-reference-status').textContent=citationDetail(target);
    if(frame.tagName!=='IFRAME'||!(target.branchId||target.item))return;
    const request=++citationRequest;
    pendingCitation={frame,request,target};
    frame.contentWindow?.postMessage({type:'log-analysis-reference',request,target},'*');
    citationTimer=setTimeout(()=>{
      if(pendingCitation?.request===request)$('llm-reference-status').textContent=citationDetail(target)+' · 해당 위치를 아직 열지 못했습니다.';
    },10000);
  });
  const availableTopics=new Set(['all',...topicPanels.filter(([id])=>$(id)).map(([,value])=>value)]);
  for(const option of [...topic.options])if(!availableTopics.has(option.value))option.remove();
  const mechanismTabs=[...panel.querySelectorAll('[data-mechanism]')];
  const multiSelect=$('llm-multi-select');
  multiSelect.checked=true;
  let selectedItems=[];
  let mechanism='all';
  const neuronScope=()=>['relu','tree'].includes(topic.value)||(topic.value==='mechanisms'&&mechanism==='bounds');
  function chooseMechanism(value,notify=true){
    mechanism=value;
    for(const tab of mechanismTabs){const active=tab.dataset.mechanism===value;tab.setAttribute('aria-checked',String(active));tab.tabIndex=active?0:-1;}
    if(notify)changed();
  }
  const controlScope=()=>{
    const value={topic:topic.value};
    if(topic.value==='mechanisms')value.mechanism=mechanism;
    if(round.value!=='')value.round=Number(round.value);
    if(topic.value==='simplex' && call.value.trim())value.call_id=call.value.trim();
    if(neuronScope() && layer.value!==''){
      value.layer=Number(layer.value);if(neuron.value!=='')value.neuron=Number(neuron.value);
    }
    return value;
  };
  const scope=()=>selectedItems.length?
    (multiSelect.checked?{selections:selectedItems.map(item=>item.scope)}:selectedItems.at(-1).scope):controlScope();
  const caption=()=>multiSelect.checked&&selectedItems.length?
    `${selectedItems.length}개 항목 선택 · `+[...new Set(selectedItems.map(item=>[...topic.options].find(option=>option.value===item.scope.topic)?.textContent||item.scope.topic))].join(' / '):
    [topic.selectedOptions[0].textContent,
    topic.value==='mechanisms'?mechanismTabs.find(tab=>tab.dataset.mechanism===mechanism).textContent:'',round.value?`R${round.value}`:'전체 Round',
    neuronScope()&&layer.value!==''?`L${layer.value}${neuron.value!==''?':N'+neuron.value:''}`:'',
    topic.value==='simplex'?window.formatLogAnalysisText(call.value):''].filter(Boolean).join(' · ');
  const enabled=()=>{
    $('llm-start').disabled=!info || busy || !$('llm-consent').checked
      || (info.model_selection && (!modelCatalogLoaded || !$('llm-model').value));
    $('llm-stop').hidden=!busy;
    $('llm-stop').disabled=!analysisJobId||analysisStopping;
  };
  const options=(element,values,empty)=>{
    const selected=element.value;
    const fromChart=element.selectedOptions[0]?.dataset.visualization==='1';
    element.replaceChildren(new Option(empty,''));
    for(const value of values)element.add(new Option(String(value),String(value)));
    if(fromChart && selected && ![...element.options].some(option=>option.value===selected)){
      const option=new Option(selected,selected);option.dataset.visualization='1';element.add(option);
    }
    if([...element.options].some(option=>option.value===selected))element.value=selected;
  };
  function renderSelections(){
    $('llm-clear-selection').disabled=!selectedItems.length;
    $('llm-selected-items').hidden=!selectedItems.length;
    $('llm-selected-items').replaceChildren(...selectedItems.map((item,index)=>{
      const chip=document.createElement('div'),label=document.createElement('span'),remove=document.createElement('button');
      chip.className='llm-selected-item';chip.setAttribute('role','listitem');
      label.textContent=item.label||item.text.split('\n').slice(0,2).join(' · ');label.title=item.text;
      remove.type='button';remove.textContent='×';remove.setAttribute('aria-label',`${index+1}번째 선택 해제`);
      remove.addEventListener('click',()=>writeSelections(selectedItems.filter((_,position)=>position!==index)));
      chip.append(label,remove);return chip;
    }));
  }
  function setSelectionControls(value){
    topic.value=value.topic;
    if(value.topic==='mechanisms')chooseMechanism(value.mechanism||'all',false);
    const setChoice=(element,value)=>{
      const text=value==null?'':String(value);
      if(text&&![...element.options].some(option=>option.value===text)){
        const option=new Option(text,text);option.dataset.visualization='1';element.add(option);
      }
      element.value=text;
    };
    setChoice(round,value.round);call.value=value.call_id||'';
    setChoice(layer,value.layer);options(neuron,info?.options.neurons[layer.value]||[],'전체 Neuron');setChoice(neuron,value.neuron);
  }
  function writeSelections(next){
    const nextControls={...(next.at(-1)?.scope||{topic:'all'})};
    delete nextControls.branch_id;
    if(next.length===selectedItems.length&&next.every((item,index)=>item.key===selectedItems[index].key&&item.text===selectedItems[index].text&&item.label===selectedItems[index].label)&&
      JSON.stringify(Object.entries(controlScope()).sort())===JSON.stringify(Object.entries(nextControls).sort()))return true;
    if(next.length>20){$('llm-progress').textContent='한 번에 최대 20개 항목을 선택할 수 있습니다.';return false;}
    const payload=next.map(({scope,text})=>({scope,text}));
    if(new TextEncoder().encode(JSON.stringify(payload)).length>32000){$('llm-progress').textContent='선택 정보가 너무 큽니다. 일부 항목을 해제해 주세요.';return false;}
    selectedItems=next;
    setSelectionControls(next.at(-1)?.scope||{topic:'all'});
    renderSelections();changed({keepSelections:true});
    $('llm-progress').textContent=next.length?'분석 항목을 선택했습니다.':'선택 정보를 비웠습니다.';
    return true;
  }
  function selectChartItem(selection){
    const value=selection?.scope;
    if(!value || !availableTopics.has(value.topic) || value.topic==='all' || typeof selection.text!=='string' || selection.text.length>5000)return;
    if(Object.keys(value).some(key=>!['topic','mechanism','round','call_id','layer','neuron','branch_id'].includes(key)))return;
    for(const key of ['round','layer','neuron'])if(value[key]!=null && (!Number.isInteger(value[key]) || value[key]<(key==='round'?1:0) || value[key]>1000000))return;
    if(value.call_id!=null && (value.topic!=='simplex' || typeof value.call_id!=='string' || !/^[A-Za-z0-9_-]{1,100}$/.test(value.call_id)))return;
    if(value.branch_id!=null && (value.topic!=='tree' || typeof value.branch_id!=='string' || !/^[A-Za-z0-9_-]{1,100}$/.test(value.branch_id)))return;
    if(value.topic==='mechanisms' && !mechanismTabs.some(tab=>tab.dataset.mechanism===(value.mechanism||'all')))return;
    if(value.mechanism!=null && value.topic!=='mechanisms')return;
    if((value.layer!=null || value.neuron!=null) && !(['relu','tree'].includes(value.topic)||value.topic==='mechanisms'&&value.mechanism==='bounds'))return;
    if(value.neuron!=null && value.layer==null)return;
    const text=window.formatLogAnalysisText(selection.text.trim());
    const nodeId=value.topic==='tree'&&typeof selection.id==='string'&&/^tree-node-[1-9]\d{0,15}$/.test(selection.id)?selection.id:null;
    const key=nodeId||JSON.stringify(Object.fromEntries(Object.entries(value).sort()))+'\n'+text;
    const label=nodeId&&typeof selection.label==='string'?window.formatLogAnalysisText(selection.label.slice(0,300)):'';
    const item={scope:{...value},text,key,label};
    writeSelections(multiSelect.checked?[...selectedItems.filter(previous=>previous.key!==key),item]:[item]);
  }
  window.addEventListener('log-analysis-select',event=>selectChartItem(event.detail));
  window.addEventListener('message',event=>{
    if(event.data?.type==='log-analysis-reference-result' && pendingCitation?.frame.contentWindow===event.source
        && pendingCitation.request===event.data.request){
      clearTimeout(citationTimer);
      $('llm-reference-status').textContent=citationDetail(pendingCitation.target)+(event.data.found?'':' · 화면에 해당 기록이 없습니다.');
      pendingCitation=null;
      return;
    }
    if(event.data?.type!=='log-analysis-select')return;
    const source=topicPanels.find(([id])=>$(id)?.tagName==='IFRAME' && $(id).contentWindow===event.source);
    if(!source || source[1]!==event.data.scope?.topic)return;
    selectChartItem(event.data);
  });
  const endpoint=new URL('api/',location.href);
  const modelSelect=$('llm-model');
  const effortSelect=$('llm-effort');
  let modelLevels={};
  const updateEffortOptions=()=>setAnalysisEffortOptions(effortSelect,
    modelLevels[modelSelect.value] || ((!modelSelect.value || modelSelect.value===info?.model) ? info?.thinking_levels : undefined));
  let modelCatalogLoaded=false, modelCatalogPending=false, modelCatalogError='';
  async function api(path,body){
    const response=await fetch(new URL(path,endpoint),{cache:'no-store',
      signal:AbortSignal.timeout(20000),
      ...(body?{method:'POST',headers:{'Content-Type':'application/json','X-Analysis-Token':info.csrf},body:JSON.stringify(body)}:{})});
    const value=await response.json();
    if(!response.ok)throw Object.assign(Error(value.error||`HTTP ${response.status}`),{status:response.status});
    return value;
  }
  window.bindAnalysisQuotaLink($('llm-quota-link'), () => {
    if(!info?.csrf)throw Error('분석 화면 연결이 준비되면 다시 눌러 주세요.');
    return api('open-ai-studio-quota', {});
  });
  async function loadModels(){
    if(!info?.model_selection || modelCatalogLoaded || modelCatalogPending)return;
    modelCatalogPending=true;
    try{
      const {models,thinking_levels:levels}=await api('models');
      modelLevels=levels||{};
      const current=modelSelect.value;
      const choices=[...new Set(models)];
      const modelOptions=choices.map(name=>new Option(name,name));
      if(choices.includes(current)){
        modelSelect.replaceChildren(...modelOptions);
        modelSelect.value=current;
      }else{
        modelSelect.replaceChildren(new Option('모델 선택',''),...modelOptions);
      }
      updateEffortOptions();
      modelCatalogLoaded=true;
      modelCatalogError=choices.length?'':'현재 키의 모델 목록에서 사용할 수 있는 모델이 없습니다.';
    }catch(_error){
      modelCatalogError='모델 목록 조회 실패 · 모델 목록을 다시 열면 재시도합니다.';
    }finally{
      modelCatalogPending=false;
      $('llm-connection').textContent=modelCatalogError;
      enabled();
    }
  }
  modelSelect.addEventListener('pointerdown',loadModels);
  modelSelect.addEventListener('focus',loadModels);
  modelSelect.addEventListener('change',()=>{updateEffortOptions();enabled();});
  let queuedReport=null;
  function display(result){
    if(inputBusy){queuedReport={result,version:revision};return;}
    if(result){
      baseline=result.snapshot||null;showSnapshot();
      analysisText=String(result.analysis||'');
      citationTargets=result.citation_targets||{};
      window.renderLogAnalysisMarkdown($('llm-result'),analysisText,citationTargets);
      $('llm-copy').disabled=!analysisText;
      $('llm-fullscreen').disabled=!analysisText;
      $('llm-result-tag').textContent=statusLabels[result.status]||result.status;
      $('llm-result-tag').dataset.status=result.status;
    }else clearReport('선택 항목의 저장된 분석이 없습니다.');
    if(result)$('llm-progress').textContent=[statusLabels[result.status]||result.status,result.model,
      result.finish_reason==='MAX_TOKENS'?'모델 출력 한도':''].filter(Boolean).join(' · ');
  }
  function showSnapshot(){
    const state=info?.state;
    $('llm-live').textContent=state?`실행 상태: ${state.status} · 기록 ${state.indexed_records.toLocaleString()}건${state.coverage_limited?' · 일부 미수집 기록 있음':''}`:'';
    if(!baseline){$('llm-snapshot').textContent='';$('llm-new-records').textContent='';return;}
    const when=new Date(baseline.captured_at).toLocaleString();
    $('llm-snapshot').textContent=`분석 기준: ${when} · ${baseline.finished?'종료 후':'진행 중'}${baseline.last_round?' · R'+baseline.last_round:''}${baseline.last_call?' · '+window.formatLogAnalysisText(baseline.last_call):''}${baseline.coverage_limited?' · 수집 범위 제한':''}`;
    const same=state && state.identity===baseline.identity && state.indexed_records>=baseline.indexed_records && state.indexed_bytes>=baseline.indexed_bytes;
    // Identity contains an in-process generation; persisted snapshots can be older sessions.
    const added=same?Math.max(0,state.indexed_records-baseline.indexed_records):null;
    $('llm-new-records').textContent=added===null?'이전 세션 또는 다른 수집 범위의 분석입니다. 현재까지 분석을 다시 요청할 수 있습니다.':`분석 이후 새 기록 +${added.toLocaleString()}건 (전체 로그 기준)${state.source_bytes>baseline.source_bytes && added===0?' · 추가 데이터 미수집':''}`;
  }
  let resultRequest=null;
  const savedPanel=$('llm-saved-results'), savedSelect=$('llm-saved-result'), savedStatus=$('llm-saved-status');
  let savedRequest=0;
  async function loadResult(){
    if(!info||!savedPanel.hidden)return;
    const version=revision;
    if(resultRequest)resultRequest.abort();
    const controller=new AbortController();resultRequest=controller;
    try{
      const response=await fetch(new URL('result?scope='+encodeURIComponent(JSON.stringify(scope())),endpoint),{cache:'no-store',signal:controller.signal});
      const value=await response.json();if(!response.ok)throw Error(value.error||`HTTP ${response.status}`);
      if(version===revision)display(value.result);
    }catch(error){if(error.name!=='AbortError'&&version===revision)$('llm-progress').textContent=error.message;}
    finally{if(resultRequest===controller)resultRequest=null;}
  }
  function closeSavedResults(){
    savedRequest++;
    savedPanel.hidden=true;
    $('llm-reload').setAttribute('aria-expanded','false');
    queuedReport=null;
  }
  function savedLabel(entry){
    const date=entry.created_at?new Date(entry.created_at).toLocaleString():'';
    const scopes=entry.scope.selections||[entry.scope];
    const labels=scopes.map(({topic:value,round:roundId,layer:layerId,neuron:neuronId,call_id:callId,mechanism:kind})=>{
      const name=[...topic.options].find(option=>option.value===value)?.textContent||value;
      const technique=mechanismTabs.find(tab=>tab.dataset.mechanism===kind)?.textContent;
      return [name,kind&&kind!=='all'?technique:'',roundId?`R${roundId}`:'',
        layerId!=null?`L${layerId}${neuronId!=null?':N'+neuronId:''}`:'',callId?window.formatLogAnalysisText(callId):''].filter(Boolean).join(' ');
    });
    return [date,[...new Set(labels)].join(' / '),entry.question||'기본 분석',entry.status==='incomplete'?'일부 응답':''].filter(Boolean).join(' · ');
  }
  async function showSavedResult(){
    if(!savedSelect.value)return;
    const request=++savedRequest;
    savedStatus.textContent='저장 답변을 여는 중입니다.';
    try{
      const {result}=await api('saved-result?id='+encodeURIComponent(savedSelect.value));
      if(request!==savedRequest)return;
      display(result);
      savedStatus.textContent='';
    }catch(error){if(request===savedRequest)savedStatus.textContent=error.message;}
  }
  async function openSavedResults(){
    if(!info){$('llm-progress').textContent='로컬 분석 서버에 연결된 화면에서 저장 답변을 볼 수 있습니다.';return;}
    resultRequest?.abort();queuedReport=null;
    savedPanel.hidden=false;
    $('llm-reload').setAttribute('aria-expanded','true');
    savedSelect.disabled=true;savedSelect.replaceChildren();
    savedStatus.textContent='저장 답변을 찾는 중입니다.';
    const request=++savedRequest;
    try{
      const {entries}=await api('saved-results?scope='+encodeURIComponent(JSON.stringify(scope())));
      if(request!==savedRequest)return;
      if(!entries.length){savedStatus.textContent='이 실행에 저장된 답변이 없습니다.';return;}
      savedSelect.replaceChildren(...entries.map(entry=>new Option(savedLabel(entry),entry.id)));
      savedSelect.value=(entries.find(entry=>entry.matching_scope)||entries[0]).id;
      savedSelect.disabled=false;
      await showSavedResult();
    }catch(error){if(request===savedRequest)savedStatus.textContent=error.message;}
  }
  function changed({keepSelections=false}={}){
    if(!savedPanel.hidden)closeSavedResults();
    if(!keepSelections&&!multiSelect.checked&&selectedItems.length){selectedItems=[];renderSelections();}
    revision++; $('llm-selection').textContent=caption();
    $('llm-call-label').hidden=topic.value!=='simplex';
    $('llm-mechanism-label').hidden=topic.value!=='mechanisms';
    for(const id of ['llm-layer-label','llm-neuron-label'])$(id).hidden=!neuronScope();
    neuron.disabled=layer.value==='';
    enabled();
    clearReport('선택 항목의 저장 결과를 확인 중입니다.');
    $('llm-progress').textContent='';
    if(info)loadResult();else $('llm-result').textContent='로컬 분석 서버에서 이 대시보드를 열어주세요.';
  }
  for(const element of [topic,round,neuron])element.addEventListener('change',changed);
  multiSelect.addEventListener('change',()=>{
    if(!multiSelect.checked&&selectedItems.length>1)writeSelections(selectedItems.slice(-1));
    else changed({keepSelections:true});
  });
  $('llm-clear-selection').addEventListener('click',()=>writeSelections([]));
  for(const tab of mechanismTabs){
    tab.addEventListener('click',()=>chooseMechanism(tab.dataset.mechanism));
    tab.addEventListener('keydown',event=>{
      const current=mechanismTabs.indexOf(tab),count=mechanismTabs.length;
      const next=event.key==='ArrowRight'?(current+1)%count:event.key==='ArrowLeft'?(current+count-1)%count:event.key==='Home'?0:event.key==='End'?count-1:null;
      if(next===null)return;event.preventDefault();mechanismTabs[next].focus();chooseMechanism(mechanismTabs[next].dataset.mechanism);
    });
  }
  let callInputTimer=null;
  call.addEventListener('input',()=>{
    if(callInputTimer!==null)clearTimeout(callInputTimer);
    callInputTimer=setTimeout(()=>{callInputTimer=null;if(!inputComposing)changed();},180);
  });
  call.addEventListener('compositionend',()=>{if(callInputTimer!==null)clearTimeout(callInputTimer);callInputTimer=setTimeout(()=>{callInputTimer=null;changed();},180);});
  window.addEventListener('log-analysis-input-activity',event=>{
    if(event.detail.active)return;
    if(queuedReport){const queued=queuedReport;queuedReport=null;if(queued.version===revision)display(queued.result);}
    if(location.protocol!=='file:')refreshInfo();
  });
  layer.addEventListener('change',()=>{options(neuron,info?.options.neurons[layer.value]||[],'전체 Neuron');changed();});
  consent.addEventListener('change',()=>{
    try{if(consent.checked)sessionStorage.setItem(consentKey,'1');else sessionStorage.removeItem(consentKey);}catch(_error){}
    enabled();
  });
  $('llm-reload').addEventListener('click',openSavedResults);
  savedSelect.addEventListener('change',showSavedResult);
  $('llm-close-saved').addEventListener('click',()=>{
    closeSavedResults();loadResult();$('llm-reload').focus();
  });
  window.bindAnalysisFullscreen($('llm-fullscreen'),$('llm-result'));
  $('llm-copy').addEventListener('click',async()=>{
    try{await navigator.clipboard.writeText(analysisText);buttonLabel($('llm-copy'),'복사됨','copy');}
    catch(_error){buttonLabel($('llm-copy'),'복사 실패','copy');}
    setTimeout(()=>{buttonLabel($('llm-copy'),'원문 복사','copy');},1500);
  });
  function showCancelledAnalysis(){
    analysisJobId=null;analysisStopping=false;busy=false;enabled();
    if(savedPanel.hidden){
      clearReport('분석을 중단했습니다.');
      $('llm-result-tag').textContent='중단됨';
      $('llm-progress').textContent='분석을 중단했습니다.';
    }
  }
  $('llm-stop').addEventListener('click',async()=>{
    if(!analysisJobId||analysisStopping)return;
    const id=analysisJobId;
    analysisStopping=true;enabled();
    try{
      const state=await api('analyze-stop',{id});
      if(analysisJobId===id&&state.status==='cancelled')showCancelledAnalysis();
    }catch(error){if(analysisJobId===id)$('llm-progress').textContent=error.message;}
    finally{if(analysisJobId===id){analysisStopping=false;enabled();}}
  });
  $('llm-start').addEventListener('click',async()=>{
    if($('llm-start').disabled)return;
    if(!savedPanel.hidden)closeSavedResults();
    if(callInputTimer!==null){clearTimeout(callInputTimer);callInputTimer=null;changed();}
    if(selectedItems.length&&info.selection_context!==true){
      $('llm-progress').textContent='선택 정보 전달을 지원하도록 시각화 연결을 종료한 뒤 다시 실행해 주세요.';return;
    }
    const selection=scope(), version=++revision;
    const selected_items=selectedItems.map(({scope,text})=>({scope:{...scope},text}));
    busy=true;enabled();$('llm-progress').textContent='분석을 시작합니다.';
    clearReport('선택 항목을 분석 중입니다.');$('llm-result-tag').textContent='분석 중';
    try{
      const job=await api('analyze',{scope:selection,question:$('llm-question').value,selected_items,
        model:modelSelect.value||info.model,thinking_level:effortSelect.value,consent:$('llm-consent').checked});
      analysisJobId=job.id;enabled();
      if(version===revision){baseline=job.snapshot;showSnapshot();}
      const poll=async()=>{
        if(analysisJobId!==job.id)return;
        try{
          const state=await api('job?id='+encodeURIComponent(job.id));
          if(analysisJobId!==job.id)return;
          if(version===revision&&savedPanel.hidden){baseline=state.snapshot||baseline;showSnapshot();}
          if(state.status==='running'){setTimeout(poll,1000);return;}
          if(state.status==='cancelled'){showCancelledAnalysis();return;}
          analysisJobId=null;analysisStopping=false;
          busy=false;enabled();
          // A result for the previous selection must never overwrite the current item.
          if(version!==revision)loadResult();
          else if(savedPanel.hidden){if(state.result)display(state.result);else $('llm-progress').textContent=state.error||state.status;}
        }catch(error){
          if(analysisJobId!==job.id)return;
          if(version===revision)$('llm-progress').textContent=error.message;
          if(error.status===404||error.status===403){analysisJobId=null;analysisStopping=false;busy=false;enabled();}
          else setTimeout(poll,1000);
        }
      };poll();
    }catch(error){busy=false;enabled();if(version===revision)$('llm-progress').textContent=error.message;}
  });
  for(const [id,value] of topicPanels){
    const heading=document.getElementById(id)?.closest('section')?.querySelector('h2');
    if(!heading)continue;
    const button=document.createElement('button');button.type='button';button.className='llm-topic-button';buttonLabel(button,'LLM 분석','analyze');
    button.addEventListener('click',()=>{
      topic.value=value;
      if(value==='mechanisms')chooseMechanism('all',false);
      changed();
      if(getComputedStyle(panel).position==='sticky') panel.scrollTop=0;
      else panel.scrollIntoView({behavior:'smooth',block:'start'});
    });
    heading.appendChild(button);
  }
  renderSelections();changed();
  if(location.protocol==='file:'){
    $('llm-connection').textContent='LLM 분석은 로컬 서버에서 연결됩니다.';
    return;
  }
  async function refreshInfo(){
    if(infoBusy)return;
    if(infoTimer!==null){clearTimeout(infoTimer);infoTimer=null;}
    if(inputBusy){infoTimer=setTimeout(refreshInfo,3000);return;}
    infoBusy=true;
    try{
      const value=pendingInfo||await api('info');pendingInfo=null;
      if(inputBusy){pendingInfo=value;return;}
      const first=!info;
      const reset=info?.state?.identity!==value.state?.identity;
      const previous=JSON.stringify(scope());
      const choicesChanged=JSON.stringify(info?.options)!==JSON.stringify(value.options);
      info=value;
      if(reset && !first){
        round.value='';layer.value='';neuron.value='';call.value='';
        selectedItems=[];renderSelections();
      }
      if(choicesChanged || reset){
        options(round,info.options.rounds,'전체 Round');
        options(layer,Object.keys(info.options.neurons).map(Number).sort((a,b)=>a-b),'전체 Layer');
        options(neuron,info.options.neurons[layer.value]||[],'전체 Neuron');
        $('llm-call-list').replaceChildren(...info.options.calls.map(value=>new Option(value,value)));
      }
      $('llm-model-label').hidden=!info.model_selection;
      $('llm-effort-label').hidden=false;
      if(first || reset)updateEffortOptions();
      if(info.model_selection && !modelSelect.options.length){
        modelSelect.add(new Option(info.model,info.model));
        loadModels();
      }
      $('llm-connection').textContent=info.model_selection ? modelCatalogError : info.model;
      if(first || reset || previous!==JSON.stringify(scope()))changed({keepSelections:true});
      showSnapshot();enabled();
    }catch(error){$('llm-connection').textContent='로컬 분석 서버에 연결하지 못했습니다: '+error.message;}
    finally{infoBusy=false;infoTimer=setTimeout(refreshInfo,3000);}
  }
  refreshInfo();
})();
