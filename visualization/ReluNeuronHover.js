(() => {
  const maps = new Map();
  const attached = new WeakSet();
  let hovered = null;
  const tooltip = document.createElement('div');
  tooltip.id='relu-neuron-tooltip';tooltip.setAttribute('role','tooltip');tooltip.hidden=true;
  Object.assign(tooltip.style,{position:'fixed',zIndex:'9999',pointerEvents:'none',whiteSpace:'pre-line',
    maxWidth:'min(340px, calc(100vw - 16px))',padding:'8px 10px',border:'1px solid #94a3b8',
    borderRadius:'6px',background:'#fffffff5',color:'#1f2937',font:'14px/1.5 system-ui,sans-serif',boxShadow:'0 3px 12px #0003'});
  document.body.append(tooltip);
  window.addEventListener('message', event => {
    if (event.source !== window || event.data?.type !== 'relu-neuron-hover' || !Array.isArray(event.data.points)) return;
    maps.set(event.data.image, event.data.points);
    if(hovered)showHover(hovered.image,hovered.x,hovered.y);
  });
  function showHover(image,clientX,clientY) {
      hovered={image,x:clientX,y:clientY};
      const box = image.getBoundingClientRect();
      if (!box.width || !box.height) return;
      const x = (clientX - box.left) / box.width, y = (clientY - box.top) / box.height;
      let selected = null, best = Infinity;
      for (const point of maps.get(image.dataset.reluHoverImage) || []) {
        const dx = Math.abs(x-point.x), dy = Math.abs(y-point.y);
        if (dx > Math.max(point.rx, 4/box.width) || dy > Math.max(point.ry, 4/box.height)) continue;
        const distance = (dx*box.width)**2+(dy*box.height)**2;
        if (distance < best) {best=distance;selected=point;}
      }
      image.removeAttribute('title');
      const label = selected ? `Layer ${selected.layer} · Neuron ${selected.neuron}\n`+
        (selected.roundSplits || []).filter(item=>item.count>0).map(item=>`${item.round==null?'Round 미기록':'R'+item.round} split: ${item.count}회`).join('\n') : '';
      tooltip.hidden=!selected;
      if(selected){
        tooltip.textContent=label;
        tooltip.style.left=Math.max(8,Math.min(clientX+12,innerWidth-tooltip.offsetWidth-8))+'px';
        tooltip.style.top=Math.max(8,Math.min(clientY+14,innerHeight-tooltip.offsetHeight-8))+'px';
      }
  }
  window.attachReluNeuronHover = (image, uri) => {
    const url = new URL(uri, location.href);
    image.dataset.reluHoverImage=decodeURIComponent(url.pathname.split('/').pop());
    if (!attached.has(image)) {
      attached.add(image);
      image.addEventListener('mousemove',event=>showHover(image,event.clientX,event.clientY),{passive:true});
      image.addEventListener('mouseleave',()=>{if(hovered?.image===image){hovered=null;tooltip.hidden=true;}});
    }
    if(hovered && !hovered.image.isConnected)showHover(image,hovered.x,hovered.y);
    url.pathname = url.pathname.replace(/\.[^/.]+$/, '.hover.js');
    url.searchParams.set('t', Date.now());
    const script = document.createElement('script');
    script.onload = script.onerror = () => script.remove();
    script.src = url.href; document.head.append(script);
  };
  window.addEventListener('blur',()=>{hovered=null;tooltip.hidden=true;});
  document.addEventListener('scroll',()=>{hovered=null;tooltip.hidden=true;},{capture:true,passive:true});
})();
