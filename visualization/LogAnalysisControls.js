window.setAnalysisEffortOptions = (select, levels) => {
  const previous = select.value;
  const labels = {default:'모델 기본값', minimal:'최소', low:'낮음', medium:'보통', high:'높음', xhigh:'매우 높음'};
  levels ||= ['default'];
  select.replaceChildren(...levels.map(level => new Option(labels[level], level)));
  select.value = levels.includes(previous) ? previous : levels.includes('low') ? 'low' : levels[0];
};

window.bindAnalysisQuotaLink = (link, openQuota) => {
  if (!link || location.protocol === 'file:') return;
  const status = link.parentElement.querySelector('[data-quota-status]');
  let opening = false;
  const open = async event => {
    event.preventDefault();
    if (opening) return;
    opening = true;
    status.textContent = '';
    try {
      await openQuota();
      status.textContent = '기본 브라우저에서 열었습니다.';
    } catch (error) {
      status.textContent = error.message;
    } finally {
      opening = false;
    }
  };
  link.addEventListener('click', open);
  link.addEventListener('auxclick', event => { if (event.button === 1) open(event); });
};

window.bindAnalysisFullscreen = (button, report) => {
  const scrollContainer = content => {
    for (let element = content; element && element !== document.body; element = element.parentElement) {
      if (/^(auto|scroll|overlay)$/.test(getComputedStyle(element).overflowY)
          && element.scrollHeight > element.clientHeight) return element;
    }
    return document.scrollingElement;
  };
  const viewportTop = scroller => scroller === document.scrollingElement ? 0 :
    scroller.getBoundingClientRect().top + scroller.clientTop;
  const readingPosition = (content, scroller) => {
    const edge = viewportTop(scroller);
    const blocks = Array.from(content.children);
    const index = blocks.findIndex(block => block.getBoundingClientRect().bottom > edge);
    const block = index < 0 ? blocks.at(-1) || content : blocks[index];
    const rect = block.getBoundingClientRect();
    return {index: blocks.indexOf(block), fraction: Math.max(0, Math.min(1, (edge - rect.top) / Math.max(1, rect.height))),
      atStart: (blocks[0] || content).getBoundingClientRect().top >= edge};
  };
  const moveToPosition = (content, scroller, position) => {
    const block = content.children[position.index] || content;
    const rect = block.getBoundingClientRect();
    const top = position.atStart && scroller === content ? 0 :
      scroller.scrollTop + rect.top + rect.height * position.fraction - viewportTop(scroller);
    scroller.scrollTo({top, behavior:'instant'});
  };
  button.addEventListener('click', () => {
    const normalScroll = scrollContainer(report);
    const originalPosition = readingPosition(report, normalScroll);
    const originalContent = report.firstChild;
    const dialog = document.createElement('dialog');
    dialog.className = 'llm-fullscreen';
    dialog.setAttribute('aria-label', '분석 결과');
    const header = document.createElement('header');
    header.className = 'llm-report-header';
    const title = document.createElement('span');
    title.className = 'llm-report-title';
    title.textContent = '분석 결과';
    const close = document.createElement('button');
    close.className = 'llm-button llm-fullscreen-close';
    close.type = 'button';
    close.textContent = '닫기';
    header.append(title, close);
    const content = report.cloneNode(true);
    content.removeAttribute('id');
    content.tabIndex = 0;
    dialog.append(header, content);
    button.closest('section').append(dialog);
    const closeReport = () => {
      const position = readingPosition(content, content);
      dialog.close();
      if (report.firstChild === originalContent && (!position.atStart || !originalPosition.atStart)) {
        moveToPosition(report, scrollContainer(report), position);
      }
    };
    close.addEventListener('click', closeReport);
    dialog.addEventListener('cancel', event => { event.preventDefault(); closeReport(); });
    dialog.addEventListener('close', () => dialog.remove(), {once:true});
    dialog.showModal();
    content.focus({preventScroll:true});
    if (!originalPosition.atStart) moveToPosition(content, content, originalPosition);
  });
};
