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
    const text=String(source||''),parts=[];
    const protectedPattern=/\[(?:run|compare):[^\]\n]+\]|\[[^\]\n]+\]\([^\s)]+\)|https?:\/\/[^\s)]+|\$\$[^$]+\$\$|\$[^$\n]+\$|\\\([^\n]*?\\\)|\\\[[\s\S]*?\\\]/g;
    let offset=0;
    for(const match of text.matchAll(protectedPattern)){
      parts.push(translate(text.slice(offset,match.index)),match[0]);offset=match.index+match[0].length;
    }
    parts.push(translate(text.slice(offset)));return parts.join('');
  };
  const mathCommands={max:'max',min:'min',le:'≤',leq:'≤',ge:'≥',geq:'≥',ne:'≠',neq:'≠',
    approx:'≈',sim:'∼',cdot:'·',times:'×',pm:'±',infty:'∞',epsilon:'ε',varepsilon:'ε',alpha:'α',beta:'β',
    delta:'δ',sum:'∑',left:'',right:'',quad:' ',qquad:' '};
  function math(source,display=false){
    const result=node(display?'div':'span');result.className=display?'llm-math llm-math-display':'llm-math';
    const value=source.trim().replace(/\\([A-Za-z]+|[,;! ])/g,(all,command)=>
      Object.hasOwn(mathCommands,command)?mathCommands[command]:command==='!'?'':/^[,; ]$/.test(command)?' ':all);
    let offset=0;
    for(const match of value.matchAll(/([_^])(?:\{([^{}]+)\}|([\p{L}\p{N}∞]))/gu)){
      result.append(value.slice(offset,match.index),node(match[1]==='_'?'sub':'sup',match[2]||match[3]));
      offset=match.index+match[0].length;
    }
    result.append(value.slice(offset));return result;
  }
  const readableText=text=>text.replace(/&(?:nbsp|#160|#32|#x(?:a0|20));/gi,' ').replace(/\\sim(?![A-Za-z])/g,mathCommands.sim);
  function inline(root,source,depth=0,references){
    if(depth>6){root.append(document.createTextNode(source));return;}
    const pattern=/(`{1,2})([^`\n]{0,4096}?)\1|\*\*([^\n]+?)\*\*|__([^\n]+?)__|\*([^*\n]+?)\*|\[([^\]\n]+)\]\(([^\s)]+)\)|\[((?:run|compare):[^\]\n]+)\]|\$([^$\n]+?)\$|\\\(([^\n]*?)\\\)/g;
    let offset=0;
    for(const match of source.matchAll(pattern)){
      root.append(readableText(source.slice(offset,match.index)));
      if(match[1])root.append(node('code',match[2]));
      else if(match[3]||match[4]){const strong=node('strong');inline(strong,match[3]||match[4],depth+1,references);root.append(strong);}
      else if(match[5]){const em=node('em');inline(em,match[5],depth+1,references);root.append(em);}
      else if(match[6]){
        let url=null;try{const candidate=new URL(match[7]);if(['http:','https:'].includes(candidate.protocol))url=candidate;}catch(_error){}
        if(url){const link=node('a',match[6]);link.href=url.href;link.target='_blank';link.rel='noopener noreferrer';root.append(link);}
        else root.append(document.createTextNode(match[0]));
      }else if(match[8])references.add(match[8]);
      else root.append(math(match[9]??match[10]));
      offset=match.index+match[0].length;
    }
    root.append(readableText(source.slice(offset)));
  }
  window.formatLogAnalysisCitationLabel=target=>{
    if(target.topic==='mechanisms'){
      const role=target.role||String(target.variable||'').match(/^([zh])\d+_/)?.[1];
      const label=role==='z'?'z (입력)':role==='h'?'h (출력)':'';
      return [target.label,label].filter(Boolean).join(' · ');
    }
    const location=target.topic==='tree'?target.phase:target.round==null?'':`R${target.round}`;
    return [target.label,location,target.call==null?'':`호출 ${target.call}`].filter(Boolean).join(' · ');
  };
  function addReferences(root,references,{targets,shown}){
    if(!references.size)return;
    const footer=node('span','참고: ');footer.className='llm-references';
    for(const reference of references){
      const choices=Object.hasOwn(targets,reference)?targets[reference]:[];
      choices.forEach((target,index)=>{
        if(target.topic==='mechanisms'&&!target.item)return;
        if(target.topic==='tree'&&!target.phase)return;
        const identity=target.branchId||target.callId||target.splitId||[target.line,target.item??null];
        const key=JSON.stringify([target.topic,identity]);
        if(shown.has(key))return;
        shown.add(key);
        const chip=node('button',window.formatLogAnalysisCitationLabel(target));chip.type='button';chip.className='llm-citation';
        chip.dataset.citationRef=reference;chip.dataset.citationIndex=index;
        footer.append(chip);
      });
    }
    if(footer.childElementCount)root.append(footer);
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
  function blocks(root,source,targets,depth=0){
    if(depth>12){root.append(node('p',source));return;}
    const lines=source.replace(/\r\n?/g,'\n').split('\n');let i=0;
    const displayMath=line=>line.trim().match(/^(\$\$|\\\[)(.*)$/);
    const special=index=>/^(?:\s{0,3}#{1,6}\s|\s*```|\s*~~~|\s*>)/.test(lines[index]) || displayMath(lines[index]) || rule(lines[index]) || list(lines[index]) || table(lines,index);
    while(i<lines.length){
      if(!lines[i].trim()){i++;continue;}
      const formula=displayMath(lines[i]);
      if(formula){
        const end=formula[1]==='$$'?'$$':'\\]', parts=[formula[2].trim()];let j=i+1;
        while(!parts.at(-1).endsWith(end) && j<lines.length && lines[j].trim())parts.push(lines[j++].trim());
        if(parts.at(-1).endsWith(end)){
          parts[parts.length-1]=parts.at(-1).slice(0,-end.length);
          root.append(math(parts.join(' '),true));i=j;continue;
        }
      }
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
      if(heading){const h=node('h'+heading[1].length),refs=new Set();inline(h,heading[2],0,refs);root.append(h);addReferences(root,refs,targets);i++;continue;}
      if(rule(lines[i])){root.append(node('hr'));i++;continue;}
      if(/^\s*>/.test(lines[i])){
        const text=[];while(i<lines.length && /^\s*>/.test(lines[i]))text.push(lines[i++].replace(/^\s*>\s?/,''));
        const quote=node('blockquote');blocks(quote,text.join('\n'),targets,depth+1);root.append(quote);continue;
      }
      if(table(lines,i)){
        const wrapper=node('div');wrapper.className='llm-table-wrap';wrapper.tabIndex=0;
        const grid=node('table'),head=node('thead'),body=node('tbody'),headers=cells(lines[i]),align=cells(lines[i+1]);
        const row=(values,tag)=>{const tr=node('tr');headers.forEach((_,j)=>{const cell=node(tag),refs=new Set();inline(cell,values[j]||'',0,refs);addReferences(cell,refs,targets);
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
          const li=node('li'),refs=new Set();inline(li,item[3],0,refs);addReferences(li,refs,targets);i++;
          const nested=[];
          while(i<lines.length){
            if(!lines[i].trim()){if(i+1<lines.length && /^\s/.test(lines[i+1])){nested.push('');i++;continue;}break;}
            const next=list(lines[i]);if(next && next[1].length<=indent)break;
            const spaces=lines[i].match(/^\s*/)[0].length;if(spaces<=indent)break;
            nested.push(lines[i].slice(Math.min(spaces,indent+2)));i++;
          }
          if(nested.length)blocks(li,nested.join('\n'),targets,depth+1);
          group.append(li);
        }
        root.append(group);continue;
      }
      const text=[lines[i++]];
      while(i<lines.length && lines[i].trim() && !special(i))text.push(lines[i++]);
      const p=node('p'),refs=new Set();text.forEach((line,j)=>{if(j)p.append(document.createElement('br'));inline(p,line,0,refs);});addReferences(p,refs,targets);root.append(p);
    }
  }
  window.renderLogAnalysisMarkdown=(root,source,targets={})=>{
    const fragment=document.createDocumentFragment();blocks(fragment,window.formatLogAnalysisText(source),{targets,shown:new Set()});root.replaceChildren(fragment);
  };
})();
