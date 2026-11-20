(() => {
  const attached=new WeakSet(), maps=new WeakMap(), presses=new WeakMap();
  window.receiveAnalysisMap=(script,payload)=>{
    if(script?._selectionResolve){script._selectionResolve(payload);return;}
    const image=script?._selectionImage;
    if(!image?.isConnected || !Array.isArray(payload?.regions) || !Array.isArray(payload?.items))return;
    maps.set(image,payload);image.style.cursor=payload.regions.length?'pointer':'';
  };
  function hit(image,clientX,clientY){
    const map=maps.get(image),box=image.getBoundingClientRect();
    if(!map || !box.width || !box.height)return null;
    const x=(clientX-box.left)/box.width,y=(clientY-box.top)/box.height;
    for(const region of map.regions){
      const [left,top,right,bottom]=region.box;
      if(x<left||x>right||y<top||y>bottom||!region.items.length)continue;
      let index=0;
      if(region.positions.length){
        const value=region.direction==='y'?y:x;
        let best=Infinity;
        region.positions.forEach((position,i)=>{const delta=Math.abs(position-value);if(delta<best){best=delta;index=i;}});
      }
      const item=map.items[region.items[index]];
      return item?{scope:item.scope,text:[region.label,item.text].filter(Boolean).join('\n')}:null;
    }
    return null;
  }
  window.commitAnalysisSelection=(image,payload)=>{
    if(!attached.has(image)){
      attached.add(image);
      image.draggable=false;
      image.addEventListener('pointerdown',event=>{
        if(event.button!==0)return;
        presses.set(image,{x:event.clientX,y:event.clientY,time:performance.now(),selection:hit(image,event.clientX,event.clientY)});
      });
      image.addEventListener('pointercancel',()=>presses.delete(image));
      image.addEventListener('click',event=>{
        const press=presses.get(image);presses.delete(image);
        if(press && performance.now()-press.time<2000 && Math.hypot(event.clientX-press.x,event.clientY-press.y)>8)return;
        const selection=press && performance.now()-press.time<2000?press.selection:hit(image,event.clientX,event.clientY);
        if(selection){
          window.dispatchEvent(new CustomEvent('log-analysis-select',{detail:selection}));
        }
      });
    }
    if(payload && Array.isArray(payload.regions) && Array.isArray(payload.items))maps.set(image,payload);
    else maps.delete(image);
    image.style.cursor=payload?.regions?.length?'pointer':'';
  };
  window.loadAnalysisSelection=uri=>new Promise(resolve=>{
    const url=new URL(uri,location.href);
    if(!['file:','http:','https:'].includes(url.protocol)){resolve(null);return;}
    url.pathname=url.pathname.replace(/\.[^/.]+$/,'.selection.js');
    if(location.protocol==='file:')url.searchParams.set('t',Date.now());
    const script=document.createElement('script');let settled=false;
    const finish=payload=>{
      if(settled)return;settled=true;clearTimeout(timeout);script.remove();
      resolve(payload && Array.isArray(payload.regions) && Array.isArray(payload.items)?payload:null);
    };
    const timeout=setTimeout(()=>finish(null),1500);
    script._selectionResolve=finish;script.onerror=()=>finish(null);script.onload=()=>script.remove();
    script.src=url.href;document.head.append(script);
  });
  window.attachAnalysisSelection=async(image,uri)=>{
    const payload=await window.loadAnalysisSelection(uri);
    if(image.isConnected)window.commitAnalysisSelection(image,payload);
  };
})();
