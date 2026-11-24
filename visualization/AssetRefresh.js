(() => {
  window.visualizationImageBox ??= image=>{
    const box=image.getBoundingClientRect();
    if(!image.naturalWidth||!image.naturalHeight)return box;
    const scale=Math.min(box.width/image.naturalWidth,box.height/image.naturalHeight);
    const width=image.naturalWidth*scale,height=image.naturalHeight*scale;
    return {left:box.left+(box.width-width)/2,top:box.top+(box.height-height)/2,width,height};
  };
  if(window.visualizationAssetUpdate)return;
  window.visualizationAssetUpdate=async(uri,state,force=false)=>{
    const url=new URL(uri,location.href);
    if(!['file:','http:','https:'].includes(url.protocol))return {url:url.href,version:null};
    const fallback=()=>{url.searchParams.set('t',Date.now());return {url:url.href,version:null};};
    if(!['http:','https:'].includes(location.protocol)||url.origin!==location.origin||state.noHead)return fallback();
    const controller=new AbortController(),timeout=setTimeout(()=>controller.abort(),5000);
    try{
      const response=await fetch(url.href,{method:'HEAD',cache:'no-store',signal:controller.signal,
        headers:state.version?{'If-None-Match':state.version}:{}});
      if(response.status===405||response.status===501){state.noHead=true;return fallback();}
      if(response.status===304&&!force)return null;
      if(!response.ok&&response.status!==304)throw Error('Asset unavailable');
      const version=response.headers.get('ETag');
      if(!version){state.noHead=true;return fallback();}
      if(!force&&version===state.version)return null;
      url.searchParams.set('v',version);
      return {url:url.href,version};
    }finally{clearTimeout(timeout);}
  };
})();
