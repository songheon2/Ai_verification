(() => {
  const attached=new WeakSet(), maps=new WeakMap(), presses=new WeakMap();
  window.receiveAnalysisMap=(script,payload)=>{
    if(script?._selectionResolve){script._selectionResolve(payload);return;}
    const image=script?._selectionImage;
    if(!image?.isConnected || !Array.isArray(payload?.regions) || !Array.isArray(payload?.items))return;
    maps.set(image,payload);setMapCursor(image,payload);
  };
  function setMapCursor(image,map){
    image.style.cursor=map?.regions?.some(region=>!region.targets&&!region.positions?.length)?'pointer':'';
  }
  function contains(target,x,y){
    const [,kind,a,b,c,d,e,f]=target;
    if(kind==='box')return x>=a&&x<=c&&y>=b&&y<=d;
    if(kind==='stroke'){
      if(x<Math.min(a,c)-e||x>Math.max(a,c)+e||y<Math.min(b,d)-f||y>Math.max(b,d)+f)return false;
      const dx=(c-a)/e,dy=(d-b)/f,px=(x-a)/e,py=(y-b)/f;
      const length=dx*dx+dy*dy,t=(px*dx+py*dy)/length;
      return t>=0&&t<=1&&(px-t*dx)**2+(py-t*dy)**2<=1;
    }
    if(kind!=='triangle')return false;
    const cross=(x0,y0,x1,y1)=>(x1-x0)*(y-y0)-(y1-y0)*(x-x0);
    const sides=[cross(a,b,c,d),cross(c,d,e,f),cross(e,f,a,b)];
    return sides.every(side=>side>=0)||sides.every(side=>side<=0);
  }
  function hit(image,clientX,clientY){
    const map=maps.get(image),box=window.visualizationImageBox?.(image) ?? image.getBoundingClientRect();
    if(!map || !box.width || !box.height)return null;
    const x=(clientX-box.left)/box.width,y=(clientY-box.top)/box.height;
    for(const region of map.regions){
      const [left,top,right,bottom]=region.box;
      if(x<left||x>right||y<top||y>bottom||!region.items.length)continue;
      let index=0;
      if(region.targets){
        const target=region.targets.findLast(candidate=>contains(candidate,x,y));
        if(!target)continue;
        index=target[0];
      }else if(region.positions?.length)continue;
      const item=map.items[region.items[index]];
      return item?{scope:item.scope,text:[region.label,item.text].filter(Boolean).join('\n')}:null;
    }
    return null;
  }
  window.commitAnalysisSelection=(image,payload)=>{
    if(!attached.has(image)){
      attached.add(image);
      image.draggable=false;
      let cursorFrame=null, pointerX, pointerY;
      image.addEventListener('pointermove',event=>{
        pointerX=event.clientX;pointerY=event.clientY;
        if(cursorFrame!==null)return;
        cursorFrame=requestAnimationFrame(()=>{
          cursorFrame=null;
          image.style.cursor=hit(image,pointerX,pointerY)?'pointer':'';
        });
      },{passive:true});
      image.addEventListener('pointerleave',()=>{
        if(cursorFrame!==null)cancelAnimationFrame(cursorFrame);
        cursorFrame=null;
        setMapCursor(image,maps.get(image));
      });
      image.addEventListener('pointerdown',event=>{
        if(event.button!==0)return;
        presses.set(image,{x:event.clientX,y:event.clientY,time:performance.now(),map:maps.get(image),
          selection:hit(image,event.clientX,event.clientY)});
      });
      image.addEventListener('pointercancel',()=>presses.delete(image));
      image.addEventListener('click',event=>{
        const press=presses.get(image);presses.delete(image);
        if(press && performance.now()-press.time<2000 && Math.hypot(event.clientX-press.x,event.clientY-press.y)>8)return;
        const clicked=hit(image,event.clientX,event.clientY);
        const selection=press && performance.now()-press.time<2000
          ? (press.map!==maps.get(image)||clicked ? press.selection : null) : clicked;
        if(selection){
          window.dispatchEvent(new CustomEvent('log-analysis-select',{detail:selection}));
        }
      });
    }
    if(payload && Array.isArray(payload.regions) && Array.isArray(payload.items))maps.set(image,payload);
    else maps.delete(image);
    setMapCursor(image,payload);
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
