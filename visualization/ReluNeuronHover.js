(() => {
  const maps = new Map();
  const attached = new WeakSet();
  const pending = new Map(), presses = new WeakMap();
  let hovered = null;
  let hoverFrame=null,queuedHover=null,lastHoverAt=-Infinity;
  function cancelHover(){
    if(hoverFrame!==null)cancelAnimationFrame(hoverFrame);
    hoverFrame=null;queuedHover=null;hovered=null;lastHoverAt=-Infinity;tooltip.hidden=true;
  }
  function queueHover(image,x,y){
    if(performance.now()-lastHoverAt>=16){
      if(hoverFrame!==null)cancelAnimationFrame(hoverFrame);
      hoverFrame=null;queuedHover=null;lastHoverAt=performance.now();showHover(image,x,y);return;
    }
    queuedHover={image,x,y};
    if(hoverFrame!==null)return;
    hoverFrame=requestAnimationFrame(()=>{
      hoverFrame=null;lastHoverAt=performance.now();const latest=queuedHover;queuedHover=null;
      if(latest?.image.isConnected)showHover(latest.image,latest.x,latest.y);
    });
  }
  const tooltip = document.createElement('div');
  tooltip.id='relu-neuron-tooltip';tooltip.setAttribute('role','tooltip');tooltip.hidden=true;
  Object.assign(tooltip.style,{position:'fixed',zIndex:'9999',pointerEvents:'none',whiteSpace:'pre-line',
    maxWidth:'min(340px, calc(100vw - 16px))',padding:'8px 10px',border:'1px solid #94a3b8',
    borderRadius:'6px',background:'#fffffff5',color:'#1f2937',font:'14px/1.5 "Noto Sans KR",sans-serif',boxShadow:'0 3px 12px #0003'});
  document.body.append(tooltip);
  window.addEventListener('message', event => {
    if (event.source !== window || event.data?.type !== 'relu-neuron-hover' || !Array.isArray(event.data.points)) return;
    const readers=pending.get(event.data.image);
    if(readers){for(const finish of [...readers])finish(event.data.points);return;}
    maps.set(event.data.image, event.data.points);
    if(hovered)showHover(hovered.image,hovered.x,hovered.y);
  });
  function pickNeuron(image,clientX,clientY,padding=0) {
      const box = window.visualizationImageBox?.(image) ?? image.getBoundingClientRect();
      if (!box.width || !box.height) return;
      const x = (clientX - box.left) / box.width, y = (clientY - box.top) / box.height;
      let selected = null, best = Infinity;
      const points = maps.get(image.dataset.reluHoverImage) || [];
      for (let i=points.length-1;i>=0;i--) {
        const point=points[i];
        const dx = Math.abs(x-point.x), dy = Math.abs(y-point.y);
        if (dx > Math.max(point.rx, padding/box.width) || dy > Math.max(point.ry, padding/box.height)) continue;
        if(!padding)return point;
        const distance = (dx*box.width)**2+(dy*box.height)**2;
        if (distance < best) {best=distance;selected=point;}
      }
      return selected;
  }
  function showHover(image,clientX,clientY) {
      hovered={image,x:clientX,y:clientY};
      const selected=pickNeuron(image,clientX,clientY,4);
      image.removeAttribute('title');
      const label = selected ? `Layer ${selected.layer} · Neuron ${selected.neuron}\n`+
        (selected.roundSplits || []).filter(item=>item.count>0).map(item=>`${item.round==null?'Round 미기록':'R'+item.round} split: ${item.count}회`).join('\n') : '';
      tooltip.hidden=!selected;
      if(selected){
        if(tooltip.textContent!==label)tooltip.textContent=label;
        const width=tooltip.offsetWidth,height=tooltip.offsetHeight;
        tooltip.style.left=Math.max(8,Math.min(clientX+12,innerWidth-width-8))+'px';
        tooltip.style.top=Math.max(8,Math.min(clientY+14,innerHeight-height-8))+'px';
      }
      return selected;
  }
  window.commitReluNeuronHover = (image, uri, points) => {
    const url = new URL(uri, location.href);
    image.dataset.reluHoverImage=decodeURIComponent(url.pathname.split('/').pop());
    maps.set(image.dataset.reluHoverImage,points||[]);
    if (!attached.has(image)) {
      attached.add(image);
      image.addEventListener('mousemove',event=>queueHover(image,event.clientX,event.clientY),{passive:true});
      image.addEventListener('mouseleave',()=>{if(hovered?.image===image||queuedHover?.image===image)cancelHover();});
      image.addEventListener('pointerdown',event=>{
        if(event.button===0)presses.set(image,{point:pickNeuron(image,event.clientX,event.clientY),x:event.clientX,y:event.clientY,time:performance.now(),
          points:maps.get(image.dataset.reluHoverImage)});
      });
      image.addEventListener('pointercancel',()=>presses.delete(image));
      image.addEventListener('click',event=>{
        const press=presses.get(image);presses.delete(image);
        if(press && performance.now()-press.time<2000 && Math.hypot(event.clientX-press.x,event.clientY-press.y)>8)return;
        const clicked=pickNeuron(image,event.clientX,event.clientY);
        const point=press && performance.now()-press.time<2000
          ? (press.points!==maps.get(image.dataset.reluHoverImage)||clicked ? press.point : null) : clicked;
        if(!point)return;
        const text=`ReLU 분기 이력 · Layer ${point.layer} · Neuron ${point.neuron}\n`+
          (point.roundSplits||[]).filter(item=>item.count>0).map(item=>`${item.round==null?'Round 미기록':'R'+item.round} split: ${item.count}회`).join('\n');
        window.dispatchEvent(new CustomEvent('log-analysis-select',{detail:{scope:{topic:'relu',layer:point.layer,neuron:point.neuron},text}}));
      });
    }
    if(hovered && (hovered.image===image || !hovered.image.isConnected))showHover(image,hovered.x,hovered.y);
  };
  window.loadReluNeuronHover = uri => new Promise(resolve=>{
    const url=new URL(uri,location.href),key=decodeURIComponent(url.pathname.split('/').pop());
    if(!['file:','http:','https:'].includes(url.protocol)){resolve(null);return;}
    let settled=false;
    const script=document.createElement('script');
    const finish=points=>{
      if(settled)return;settled=true;clearTimeout(timeout);script.remove();
      const readers=pending.get(key);readers?.delete(finish);if(!readers?.size)pending.delete(key);
      resolve(points);
    };
    if(!pending.has(key))pending.set(key,new Set());pending.get(key).add(finish);
    const timeout=setTimeout(()=>finish(null),1500);
    url.pathname = url.pathname.replace(/\.[^/.]+$/, '.hover.js');
    url.searchParams.set('t', Date.now());
    script.onload=()=>script.remove();script.onerror=()=>finish(null);
    script.src = url.href; document.head.append(script);
  });
  window.attachReluNeuronHover = async(image,uri)=>{
    const points=await window.loadReluNeuronHover(uri);
    if(image.isConnected)window.commitReluNeuronHover(image,uri,points);
  };
  window.addEventListener('blur',cancelHover);
  document.addEventListener('scroll',cancelHover,{capture:true,passive:true});
})();
