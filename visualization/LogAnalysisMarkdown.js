// Small dependency-free Markdown renderer. Untrusted output never becomes innerHTML.
(() => {
  const node=(tag,text)=>{const value=document.createElement(tag);if(text!==undefined)value.textContent=text;return value;};
  const vocabulary=window.logAnalysisVocabulary||{terms:{},sources:{}};
  const escaped=value=>value.replace(/[.*+?^${}()|[\]\\]/g,'\\$&');
  const keys=Object.keys(vocabulary.terms).sort((a,b)=>b.length-a.length);
  const termPattern=keys.length?new RegExp('\\b(?:'+keys.map(escaped).join('|')+')\\b','g'):null;
  const digits=value=>value.replace(/^0+(?=\d)/,'');
  const translate=text=>{
    text=text.replace(/\b([zh])(\d+)_(\d+)(?:_[A-Za-z0-9_]+)?\b/g,(_all,role,layer,neuron)=>
      `L${digits(layer)}:N${digits(neuron)} (${role} · ${role==='z'?'ReLU 적용 전':'ReLU 출력'})`);
    text=text.replace(/\bsimplex-(\d+)\b/g,(_all,id)=>`Simplex 호출 ID ${digits(id)}`);
    return termPattern?text.replace(termPattern,term=>vocabulary.terms[term]):text;
  };
  window.formatLogAnalysisText=source=>{
    // Preserve source references for audit matching and URLs for link safety.
    const text=String(source||'').slice(0,200000),parts=[];
    const protectedPattern=/\[(?:run|compare):[^\]\n]+\]|\[[^\]\n]+\]\([^\s)]+\)|https?:\/\/[^\s)]+/g;
    let offset=0;
    for(const match of text.matchAll(protectedPattern)){
      parts.push(translate(text.slice(offset,match.index)),match[0]);offset=match.index+match[0].length;
    }
    parts.push(translate(text.slice(offset)));return parts.join('');
  };
  const citationLabel=reference=>{
    const match=reference.match(/^(run|compare):([^:]+)(?::(\d+))?$/);
    if(!match)return '분석 기록';
    return (match[1]==='compare'?'비교 실행 · ':'')+(vocabulary.sources[match[2]]||'분석 기록')+(match[3]?' · 기록 '+match[3]:'');
  };
  function inline(root,source,depth=0){
    if(depth>6){root.append(document.createTextNode(source));return;}
    const pattern=/(`{1,2})([^`\n]{0,4096}?)\1|\*\*([^\n]+?)\*\*|__([^\n]+?)__|\*([^*\n]+?)\*|\[([^\]\n]+)\]\(([^\s)]+)\)|\[((?:run|compare):[^\]\n]+)\]/g;
    let offset=0;
    for(const match of source.matchAll(pattern)){
      root.append(document.createTextNode(source.slice(offset,match.index)));
      if(match[1])root.append(node('code',match[2]));
      else if(match[3]||match[4]){const strong=node('strong');inline(strong,match[3]||match[4],depth+1);root.append(strong);}
      else if(match[5]){const em=node('em');inline(em,match[5],depth+1);root.append(em);}
      else if(match[6]){
        let url=null;try{const candidate=new URL(match[7]);if(['http:','https:'].includes(candidate.protocol))url=candidate;}catch(_error){}
        if(url){const link=node('a',match[6]);link.href=url.href;link.target='_blank';link.rel='noopener noreferrer';root.append(link);}
        else root.append(document.createTextNode(match[0]));
      }else{const chip=node('span',citationLabel(match[8]));chip.className='llm-citation';root.append(chip);}
      offset=match.index+match[0].length;
    }
    root.append(document.createTextNode(source.slice(offset)));
  }
  function cells(line){
    line=line.trim().replace(/^\|/,'').replace(/\|$/,'');
    const values=[];let current='',quoted=false;
    for(let i=0;i<line.length;i++){
      const char=line[i];
      if(char==='\\' && line[i+1]==='|'){current+='|';i++;}
      else if(char==='`'){quoted=!quoted;current+=char;}
      else if(char==='|'&&!quoted){values.push(current.trim());current='';}
      else current+=char;
    }
    values.push(current.trim());return values;
  }
  const rule=line=>/^\s{0,3}(?:-{3,}|\*{3,}|_{3,})\s*$/.test(line);
  const list=line=>line.match(/^(\s*)([-+*]|\d+[.)])\s+(.+)$/);
  const table=(lines,i)=>i+1<lines.length && lines[i].includes('|') &&
    cells(lines[i+1]).length===cells(lines[i]).length && cells(lines[i+1]).every(v=>/^:?-{3,}:?$/.test(v));
  function blocks(root,source,depth=0){
    if(depth>12){root.append(node('p',source));return;}
    const lines=source.replace(/\r\n?/g,'\n').split('\n');let i=0;
    const special=index=>/^(?:\s{0,3}#{1,6}\s|\s*```|\s*~~~|\s*>)/.test(lines[index]) || rule(lines[index]) || list(lines[index]) || table(lines,index);
    while(i<lines.length){
      if(!lines[i].trim()){i++;continue;}
      const fence=lines[i].match(/^\s{0,3}(`{3,}|~{3,})(.*)$/);
      if(fence){
        const text=[],mark=fence[1][0],length=fence[1].length,language=fence[2].trim().split(/\s/)[0];i++;
        while(i<lines.length && !(new RegExp('^\\s{0,3}'+mark+'{'+length+',}\\s*$')).test(lines[i]))text.push(lines[i++]);
        if(i<lines.length)i++;
        const group=node('div');group.className='llm-code-block';
        if(language){const label=node('div',language.slice(0,40));label.className='llm-code-language';group.append(label);}
        const pre=node('pre'),code=node('code',text.join('\n'));pre.append(code);group.append(pre);root.append(group);continue;
      }
      const heading=lines[i].match(/^\s{0,3}(#{1,6})\s+(.+?)\s*#*\s*$/);
      if(heading){const h=node('h'+heading[1].length);inline(h,heading[2]);root.append(h);i++;continue;}
      if(rule(lines[i])){root.append(node('hr'));i++;continue;}
      if(/^\s*>/.test(lines[i])){
        const text=[];while(i<lines.length && /^\s*>/.test(lines[i]))text.push(lines[i++].replace(/^\s*>\s?/,''));
        const quote=node('blockquote');blocks(quote,text.join('\n'),depth+1);root.append(quote);continue;
      }
      if(table(lines,i)){
        const wrapper=node('div');wrapper.className='llm-table-wrap';wrapper.tabIndex=0;
        const grid=node('table'),head=node('thead'),body=node('tbody'),headers=cells(lines[i]),align=cells(lines[i+1]);
        const row=(values,tag)=>{const tr=node('tr');headers.forEach((_,j)=>{const cell=node(tag);inline(cell,values[j]||'');
          cell.style.textAlign=align[j].startsWith(':')&&align[j].endsWith(':')?'center':align[j].endsWith(':')?'right':'left';tr.append(cell);});return tr;};
        head.append(row(headers,'th'));i+=2;
        while(i<lines.length && lines[i].trim() && lines[i].includes('|'))body.append(row(cells(lines[i++]),'td'));
        grid.append(head,body);wrapper.append(grid);root.append(wrapper);continue;
      }
      const first=list(lines[i]);
      if(first){
        const indent=first[1].length,ordered=/\d/.test(first[2]),group=node(ordered?'ol':'ul');
        if(ordered)group.start=parseInt(first[2],10);
        while(i<lines.length){
          const item=list(lines[i]);if(!item || item[1].length!==indent || /\d/.test(item[2])!==ordered)break;
          const li=node('li');inline(li,item[3]);i++;
          const nested=[];
          while(i<lines.length){
            if(!lines[i].trim()){if(i+1<lines.length && /^\s/.test(lines[i+1])){nested.push('');i++;continue;}break;}
            const next=list(lines[i]);if(next && next[1].length<=indent)break;
            const spaces=lines[i].match(/^\s*/)[0].length;if(spaces<=indent)break;
            nested.push(lines[i].slice(Math.min(spaces,indent+2)));i++;
          }
          if(nested.length)blocks(li,nested.join('\n'),depth+1);
          group.append(li);
        }
        root.append(group);continue;
      }
      const text=[lines[i++]];
      while(i<lines.length && lines[i].trim() && !special(i))text.push(lines[i++]);
      const p=node('p');text.forEach((line,j)=>{if(j)p.append(document.createElement('br'));inline(p,line);});root.append(p);
    }
  }
  window.renderLogAnalysisMarkdown=(root,source)=>{
    const fragment=document.createDocumentFragment();blocks(fragment,window.formatLogAnalysisText(source).slice(0,200000));root.replaceChildren(fragment);
  };
})();
