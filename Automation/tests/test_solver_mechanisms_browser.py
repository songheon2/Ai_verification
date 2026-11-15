"""Browser checks for the independent, live mechanism report."""
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
import re
import subprocess
import tempfile
import threading
import unittest
from unittest.mock import patch
from urllib.parse import parse_qs, urlsplit

from Automation.tests.test_reactive_tree_browser import chromium_path
from visualization.SolverMechanisms import write_solver_mechanisms_html
from visualization.LiveSnapshot import snapshot_path


class MechanismBrowserTests(unittest.TestCase):
    @unittest.skipUnless(chromium_path(), "Chromium is not installed")
    def test_independent_report_refreshes_and_keeps_selected_filter(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            stages = []
            snapshots = []
            for count, status in ((0, "RUNNING"), (3, "RUNNING"), (7, "SAT")):
                event = {"sequence": 1, "kind": "numerical", "action": "check", "drift": 0.001,
                         "residual": 0.25, "threshold": 1e-6, "after_drift": 0,
                         "outcome": "restored", "lineage_pivots": 1000,
                         "depth": 1, "nodeId": 2, "round_index": 1}
                report = {"available": True, "totals": {"drift_checks": count, "restores": count},
                          "settings": [{"backjump": True, "drift_threshold": 1e-6,
                                        "underapprox_enabled": count > 0}], "eventCount": count,
                          "boundRounds": [{"round_index": 1, "ranges": [
                              {"variable": "z1_0", "state": "unstable", "before": [None,None], "after": [-2,3]},
                              {"variable": "z1_1", "state": "active", "before": [0,5], "after": [1,2]},
                              {"variable": "z2_49_vnnlib_199", "state": "unstable", "before": [-1,2], "after": [-1,2]}]}],
                          "numericalChecks": [event] if count else [],
                          "numericalPivots": [{"sequence": 3,"coefficient": 1, "zero_threshold": 1e-12}],
                          "eventsByKind": {"numerical": [event] if count else [],
                                          "backjump": [{"kind": "backjump", "sequence": 2, "from_depth": 6,
                                                        "target_depth": 2, "round_index": 1}] if count else [],
                                          "underapprox": []}}
                history = [
                    {"variable":"z1_0","neuron":"z1_0","role":"z","before":[None,None],"after":[-2,3]},
                    {"variable":"h1_0","neuron":"z1_0","role":"h","before":[0,None],"after":[0,3]},
                    {"variable":"z1_0","neuron":"z1_0","role":"z","before":[-2,3],"after":[-1,2]},
                    {"variable":"z1_1","neuron":"z1_1","role":"z","before":[0,5],"after":[1,2]},
                ][:0 if count==0 else 2 if count==3 else 4]
                report['boundHistory'] = [{**record,"id":index+1,"source":"search","round_index":1,"depth":1}
                                          for index,record in enumerate(history)]
                report['eventsByKind']['bounds'] = [
                    {"kind":"stable","variable":f"z2_{index}","state":"active","source":"search",
                     "sequence":100+index,"depth":1,"round_index":1,"before":[-1,2],"after":[0,2]}
                    for index in range(65)] if count else []
                with patch('visualization.LiveSnapshot.INLINE_DATA_LIMIT', 1):
                    path = write_solver_mechanisms_html(report, root / f"report-{count}.html", status=status,
                                                        live_snapshots=True, snapshot_version=count + 1)
                stages.append(path.read_bytes())
                snapshots.append(snapshot_path(path).read_bytes())
            limited = write_solver_mechanisms_html(report, root / 'limited.html', status='SAT', panels=['numerical','underapprox']).read_bytes()
            harness = r'''<!doctype html><iframe id="report" src="/report.html"></iframe><script>
(async () => {
  const assert = (ok, message) => { if (!ok) throw Error(message); };
  const wait = async predicate => { for (let i=0; i<400; i++) { if (predicate()) return; await new Promise(resolve=>setTimeout(resolve,30)); } throw Error('wait timed out'); };
  try {
    const frame = document.querySelector('#report');
    await wait(()=>frame.contentDocument?.querySelector('#numericalValue')?.textContent.includes('오차 검사 0'));
    const doc = frame.contentDocument, filter = doc.querySelector('#mechanismFilter');
    await wait(()=>doc.body.getAttribute('aria-busy')==='false');
    const recordsPanel=doc.querySelector('#occurrenceRecords');
    assert(recordsPanel && !recordsPanel.open,'occurrence records are collapsed by default');
    assert(!doc.querySelector('#events .event'),'collapsed records do not build hidden rows');
    frame.style.width='1400px';frame.style.height='850px';
    await wait(()=>frame.contentWindow.innerWidth>=1390);
    frame.contentWindow.dispatchEvent(new frame.contentWindow.Event('resize'));
    await new Promise(resolve=>setTimeout(resolve,100));
    assert([...doc.querySelectorAll('.card')].every(card=>card.getBoundingClientRect().bottom<=frame.contentWindow.innerHeight),'all four techniques fit in the desktop viewport');
    assert(recordsPanel.getBoundingClientRect().bottom<=frame.contentWindow.innerHeight,'collapsed records header also fits');
    const compactChartHeight=doc.querySelector('.chart').getBoundingClientRect().height;
    const modeBox=doc.querySelector('#boundFollowMode').getBoundingClientRect();
    const playBox=doc.querySelector('#boundPlay').getBoundingClientRect();
    assert(Math.abs(modeBox.top-playBox.top)<4,'playback buttons sit on the mode toolbar row');
    assert(doc.querySelector('#boundPending').closest('.bound-toolbar'),'pending count sits on the toolbar row');
    const toolbarRight=doc.querySelector('.bound-toolbar').getBoundingClientRect().right;
    assert(Math.abs(doc.querySelector('#boundPending').getBoundingClientRect().right-toolbarRight)<2,'playback and pending count align to toolbar right edge');
    const chartHeights=[...doc.querySelectorAll('.chart')].map(chart=>chart.getBoundingClientRect().height);
    assert(Math.max(...chartHeights)-Math.min(...chartHeights)<2,'all four charts share the same display height: '+chartHeights.join(',')+' controls '+[...doc.querySelectorAll('.card-content')].map(item=>item.getBoundingClientRect().height).join(',')+' variable '+doc.documentElement.style.getPropertyValue('--mechanism-controls-height')+' min '+frame.contentWindow.getComputedStyle(doc.querySelector('.card-content')).minHeight);
    assert([...doc.querySelectorAll('.card')].every(card=>frame.contentWindow.getComputedStyle(card).paddingLeft==='8px'),'four cards share moderate inner margins');
    const unchangedTextSize=frame.contentWindow.getComputedStyle(doc.querySelector('.value')).fontSize;
    frame.style.height='1240px';
    frame.contentWindow.dispatchEvent(new frame.contentWindow.Event('resize'));
    await new Promise(resolve=>setTimeout(resolve,100));
    assert(doc.querySelector('.chart').getBoundingClientRect().height>compactChartHeight+100,'larger page grows the picture area substantially');
    assert(frame.contentWindow.getComputedStyle(doc.querySelector('.value')).fontSize===unchangedTextSize,'picture enlargement does not enlarge UI text');
    let fullscreenCalls=0;
    const originalFullscreen=doc.documentElement.requestFullscreen;
    doc.documentElement.requestFullscreen=()=>{fullscreenCalls++;return Promise.resolve();};
    doc.querySelector('#expandCharts').click();
    assert(fullscreenCalls===1,'large-picture button requests the full display');
    doc.documentElement.requestFullscreen=originalFullscreen;
    assert(doc.querySelectorAll('.card')[0].getBoundingClientRect().top===doc.querySelectorAll('.card')[1].getBoundingClientRect().top,'techniques form two columns');
    recordsPanel.querySelector('summary').click();
    assert(recordsPanel.open,'records expand synchronously');
    assert(!doc.querySelector('#tree'), 'independent report has no search tree');
    assert(doc.querySelectorAll('.card').length===4, 'four mechanism cards');
    assert(doc.querySelectorAll('.chart').length===4, 'four dedicated mechanism charts');
    const scope=doc.querySelector('#boundScope');
    assert(scope.value==='stable','bound chart defaults to stable fixations only');
    assert(doc.querySelectorAll('.chart')[0].textContent.includes('STABLE 확정 기록 없음'),'no raw boundary changes shown by default');
    scope.value='all';scope.dispatchEvent(new frame.contentWindow.Event('change'));
    assert(doc.querySelectorAll('.chart-empty').length===2, 'all-boundary option preserves original chart behavior');
    assert(!doc.querySelector('#repairValue,#cyclesValue'),'wrong mechanism panels removed');
    assert(doc.querySelector('#underapproxValue').textContent.includes('미적용'),'underapprox is explicitly not applied');
    assert(frame.contentWindow.getComputedStyle(doc.body).fontSize==='18px','moderate mechanism body text');
    assert(frame.contentWindow.getComputedStyle(doc.querySelector('.value')).fontSize==='16px','moderate mechanism values');
    assert(frame.contentWindow.getComputedStyle(doc.querySelector('.chart')).paddingLeft==='5px','chart padding remains modest but nonzero');
    assert(!doc.querySelector('.description,.chart-note'),'explanatory footnotes removed');
    assert(!doc.body.textContent.includes('항목별 의미가 다르므로'),'old chart note removed');
    assert(doc.documentElement.scrollWidth<=doc.documentElement.clientWidth,'charts fit narrow iframe');
    filter.value='numerical'; filter.dispatchEvent(new frame.contentWindow.Event('change'));
    assert(!doc.querySelector('#detailFilter,#details'),'extra detail section removed');
    await fetch('/stage?value=1');
    frame.contentWindow.postMessage({type:'relu-search-tree-refresh'},'*');
    await wait(()=>doc.querySelector('#numericalValue').textContent.includes('오차 검사 3'));
    assert(doc.querySelector('#underapproxValue').textContent.includes('솔버 과소근사 활성'),'remote underapprox option is reflected after refresh');
    assert(frame.contentDocument===doc && filter.value==='numerical','document and filter survive refresh');
    assert(doc.querySelector('#events').textContent.includes('타블로 복원'),'actual restoration displayed');
    filter.value='bounds';filter.dispatchEvent(new frame.contentWindow.Event('change'));
    assert(doc.querySelectorAll('#events .event').length===65,'all stable records rendered beyond old 50 limit');
    assert([...doc.querySelectorAll('#events .event')].every(row=>row.textContent.includes('STABLE')),'bounds occurrence list contains stable fixations only');
    assert(doc.querySelector('#events').textContent.includes('L2:N0') && doc.querySelector('#events').textContent.includes('L2:N64'),'first and last stable records retained');
    assert(doc.documentElement.scrollHeight>frame.contentWindow.innerHeight,'expanded records scroll the report page');
    assert(frame.contentWindow.getComputedStyle(doc.querySelector('#events')).maxHeight==='none','records are not trapped in a short inner scrollbar');
    recordsPanel.querySelector('summary').click();
    assert(!recordsPanel.open,'records collapse synchronously');
    recordsPanel.querySelector('summary').click();
    assert(recordsPanel.open && doc.querySelectorAll('#events .event').length===65,'reopening preserves all records');
    filter.value='numerical';filter.dispatchEvent(new frame.contentWindow.Event('change'));
    scope.value='stable';scope.dispatchEvent(new frame.contentWindow.Event('change'));
    assert(doc.querySelector('#boundStep').max==='65','stable mode keeps every fixation in playback');
    assert(doc.querySelectorAll('.chart')[0].textContent.includes('STABLE'),'stable playback shows fixation rather than ordinary update');
    assert(doc.querySelector('#boundsValue').textContent.startsWith('R'),'bound playback uses R round prefix');
    assert(doc.querySelector('#boundsValue').textContent.includes('z (ReLU 적용 전)'),'pre-activation role has a short explanation');
    assert([...doc.querySelector('#boundRound').options].every(option=>option.textContent.startsWith('R')),'bound round selector uses R prefix');
    scope.value='all';scope.dispatchEvent(new frame.contentWindow.Event('change'));
    assert(doc.querySelector('#boundStep').value==='1','playback begins at first change, not latest');
    assert(doc.querySelector('#boundPending').textContent==='+1개 재생 대기','pending changes counted separately');
    assert(doc.querySelectorAll('.chart svg').length===4,'all four graphs populated');
    assert(doc.querySelectorAll('.chart')[1].querySelector('path'),'backjump depth destination plotted');
    assert(doc.querySelectorAll('.chart')[2].querySelector('path'),'drift restoration plotted');
    const metric=doc.querySelector('#numericalMetric');metric.value='residual';metric.dispatchEvent(new frame.contentWindow.Event('change'));
    assert(doc.querySelectorAll('.chart')[2].textContent.includes('잔차 합'),'residual sum has its own metric');
    const epsilon=doc.querySelector('#epsilon');epsilon.value='0.5';epsilon.dispatchEvent(new frame.contentWindow.Event('change'));
    assert(doc.querySelectorAll('.chart')[3].textContent.includes('[-1.5, 2.5]'),'epsilon interval preview correct');
    epsilon.value='3';epsilon.dispatchEvent(new frame.contentWindow.Event('change'));
    assert(doc.querySelectorAll('.chart')[3].textContent.includes('공집합'),'over-shrinking does not fabricate a feasible interval');
    epsilon.value='0.5';epsilon.dispatchEvent(new frame.contentWindow.Event('change'));
    await fetch('/stage?value=2');
    frame.contentWindow.postMessage({type:'relu-search-tree-refresh',final:true},'*');
    await wait(()=>doc.querySelector('#status').textContent.includes('SAT'));
    assert(doc.querySelector('#numericalValue').textContent.includes('오차 검사 7'),'terminal summary loaded');
    assert(filter.value==='numerical' && metric.value==='residual' && epsilon.value==='0.5','terminal refresh keeps controls');
    assert(doc.querySelector('#boundStep').value==='1','new snapshots do not skip queued changes');
    assert(doc.querySelector('#boundPending').textContent==='+3개 재생 대기','new snapshots grow pending count without skipping');
    const preservedEvent=doc.querySelector('#events .event');
    doc.querySelector('#boundSearchMode').click();
    assert(doc.querySelector('#events .event')===preservedEvent,'button redraw does not rebuild occurrence records');
    const layerQuery=doc.querySelector('#boundLayerSearch'),query=doc.querySelector('#boundNeuronSearch');
    layerQuery.value='1';layerQuery.dispatchEvent(new frame.contentWindow.Event('input'));
    query.value='0 (z)';query.dispatchEvent(new frame.contentWindow.Event('input'));
    assert(doc.querySelector('#boundSearchControls').hidden===false,'search mode controls visible');
    assert(doc.querySelector('#boundNeuron').value==='z1_0','separate layer and neuron fields select the intended variable');
    query.click();
    assert(doc.querySelectorAll('#boundNeuronSearchChoices [role="option"]').length===3,'filled neuron input shows all recorded values in the layer');
    doc.querySelector('#boundNeuronSearchChoices [data-value="0 (z)"]').click();
    assert(doc.querySelector('#boundSearchPlayback').hidden===false,'neuron history slider is shown in search mode');
    const neuronStep=doc.querySelector('#boundSearchStep');
    assert(neuronStep.max==='2' && neuronStep.value==='1','neuron history contains only selected neuron changes');
    assert(doc.querySelectorAll('.chart')[0].textContent.includes('[-2, 3]'),'neuron history begins at first change');
    neuronStep.value='2';neuronStep.dispatchEvent(new frame.contentWindow.Event('input'));
    assert(doc.querySelectorAll('.chart')[0].textContent.includes('[-1, 2]'),'neuron slider selects later boundary');
    assert(doc.querySelector('#boundSearchPosition').textContent.includes('변경 #3'),'neuron slider retains original event order');
    assert(doc.querySelector('#boundStep').value==='1','neuron slider does not move global playback cursor');
    doc.querySelector('#boundSearchPrev').click();
    assert(neuronStep.value==='1','neuron previous arrow steps within filtered history');
    doc.querySelector('#boundSearchNext').click();
    assert(neuronStep.value==='2','neuron next arrow steps within filtered history');
    query.value='1 (z)';query.dispatchEvent(new frame.contentWindow.Event('input'));
    assert(neuronStep.max==='1' && neuronStep.disabled,'single-change neuron has a disabled slider');
    query.value='0 (z)';query.dispatchEvent(new frame.contentWindow.Event('input'));
    assert(neuronStep.value==='2','each neuron keeps its own history position');
    frame.contentWindow.postMessage({type:'relu-search-tree-refresh',final:true},'*');
    await new Promise(resolve=>setTimeout(resolve,100));
    assert(neuronStep.value==='2','snapshot refresh preserves searched neuron position');
    await new Promise(resolve=>setTimeout(resolve,3200));
    assert(doc.querySelectorAll('.chart')[0].textContent.includes('L1:N0 (z)'),'search remains pinned while following is off');
    doc.querySelector('#boundFollowMode').click();
    assert(doc.querySelector('#boundSearchPlayback').hidden,'search slider hides during global following');
    const started=performance.now();
    await wait(()=>doc.querySelector('#boundStep').value==='2');
    assert(performance.now()-started>=2900,'playback advances at approximately three seconds');
    assert(doc.querySelectorAll('.chart')[0].textContent.includes('L1:N0 (h)'),'next change is shown without skipping');
    assert(doc.querySelector('#boundsValue').textContent.includes('h (ReLU 출력)'),'output role has a short explanation');
    const step=doc.querySelector('#boundStep'),skipStarted=performance.now();
    doc.querySelector('#boundNext').click();
    assert(step.value==='3' && performance.now()-skipStarted<500,'right arrow immediately skips current wait');
    assert(doc.querySelector('#boundPlay').textContent==='일시정지','right arrow preserves automatic playback');
    await wait(()=>step.value==='4');
    assert(doc.querySelector('#boundPlay').textContent==='다시 재생','queued playback finishes after solver terminal status');
    assert(doc.querySelector('#boundPending').textContent==='+0개 재생 대기','pending count reaches zero after playback');
    doc.querySelector('#boundSearchMode').click();scope.value='all';scope.dispatchEvent(new frame.contentWindow.Event('change'));
    layerQuery.value='2';layerQuery.dispatchEvent(new frame.contentWindow.Event('input'));
    query.value='49 (z)';query.dispatchEvent(new frame.contentWindow.Event('input'));
    assert(doc.querySelector('#boundNeuron').options.length===1 && doc.querySelector('#boundNeuron').value==='z2_49_vnnlib_199','formatted neuron names remain searchable with raw identifiers preserved');
    assert(doc.querySelector('#boundNeuron').selectedOptions[0].textContent==='L2:N49 (z)','dropdown hides internal suffix');
    assert(doc.querySelectorAll('.chart')[0].textContent.includes('L2:N49 (z)') && !doc.querySelectorAll('.chart')[0].textContent.includes('_vnnlib_'),'chart uses layer and neuron notation');
    layerQuery.click();
    assert(doc.querySelectorAll('#boundLayerSearchChoices [role="option"]').length===2,'filled layer field shows all available layers');
    assert(!doc.querySelector('#boundSearch'),'old single neuron-name search is removed');
    const limitedFrame=document.createElement('iframe'); limitedFrame.src='/limited.html';document.body.append(limitedFrame);
    await wait(()=>limitedFrame.contentDocument?.querySelectorAll('.chart').length===4);
    const limitedDoc=limitedFrame.contentDocument;
    assert(limitedDoc.querySelectorAll('.card:not([hidden])').length===2,'only selected mechanism panels visible');
    assert([...limitedDoc.querySelectorAll('.card[hidden]')].every(card=>limitedFrame.contentWindow.getComputedStyle(card).display==='none'),'compact flex layout respects disabled panels');
    assert(![...limitedDoc.querySelector('#mechanismFilter').options].some(option=>option.value==='bounds'),'disabled mechanism absent from filter');
    assert(limitedDoc.querySelectorAll('.card')[0].hidden && !limitedDoc.querySelectorAll('.card')[2].hidden,'numerical independent from bounds');
    const result=document.createElement('pre');result.id='browser-test-result';result.textContent='PASS: independent mechanism refresh';document.body.appendChild(result);
  } catch(error) { const result=document.createElement('pre');result.id='browser-test-result';result.textContent='FAIL: '+error.message;document.body.appendChild(result); }
})();</script>'''
            stage = [0]

            class Handler(BaseHTTPRequestHandler):
                def do_GET(self):
                    url = urlsplit(self.path)
                    if url.path == "/stage":
                        stage[0] = int(parse_qs(url.query)["value"][0])
                        content = b"ok"
                    elif url.path == "/limited.html":
                        content = limited
                    elif url.path == "/report.html":
                        content = stages[stage[0]]
                    elif url.path.endswith(".snapshot.html"):
                        content = snapshots[stage[0]]
                    else:
                        content = harness.encode("utf-8")
                    self.send_response(200)
                    self.send_header("Content-Type", "text/html; charset=utf-8")
                    self.send_header("Cache-Control", "no-store")
                    self.send_header("Content-Length", str(len(content)))
                    self.end_headers()
                    self.wfile.write(content)

                def log_message(self, *_args):
                    pass

            server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
            worker = threading.Thread(target=server.serve_forever, daemon=True)
            worker.start()
            try:
                result = subprocess.run([
                    chromium_path(), "--headless", "--no-sandbox", "--disable-gpu", "--no-first-run",
                    f"--user-data-dir={root / 'profile'}", "--virtual-time-budget=20000", "--dump-dom",
                    f"http://127.0.0.1:{server.server_port}/index.html",
                ], capture_output=True, timeout=30)
                dom = result.stdout.decode("utf-8", errors="replace")
                match = re.search(r'<pre id="browser-test-result">([^<]+)</pre>', dom)
                self.assertIsNotNone(match, dom[-1000:])
                self.assertTrue(match.group(1).startswith("PASS:"), match.group(1))
            finally:
                server.shutdown()
                server.server_close()
                worker.join(timeout=5)
