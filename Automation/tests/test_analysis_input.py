"""Offline browser regression for Korean IME input during live updates."""
from pathlib import Path
import re
import subprocess
import tempfile
import threading
import unittest
from urllib.parse import urlsplit

from Automation.LogAnalysisServer import AnalysisApp, make_server
from Automation.tests.test_reactive_tree_browser import chromium_path
from visualization.ReluSearchTree import write_relu_search_tree_html
from visualization.SolveTrace import SolveTrace, TraceEvent
from visualization.SolverMechanisms import write_solver_mechanisms_html
from visualization.UnifiedRealtimeDashboard import UnifiedRealtimeDashboard


class AnalysisInputTests(unittest.TestCase):
    @unittest.skipUnless(chromium_path(), 'Chromium is not installed')
    def test_ime_defers_background_work_and_resumes_latest_without_changing_draft(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            image, tree, mechanisms = root / 'plot.svg', root / 'tree.html', root / 'mechanisms.html'
            trace = SolveTrace(events=[TraceEvent('reluplex_branch', 1, None, 1, 'z1_0',
                {'phase': '+', 'result': 'RUNNING', 'round_index': 1})])
            records = [{'kind': 'stable', 'sequence': index + 1, 'id': index + 1,
                        'variable': f'z1_{index}', 'round_index': 1, 'depth': 1,
                        'source': 'search', 'state': 'active', 'before': [-1, 2], 'after': [0, 2]}
                       for index in range(3)]
            report = {'available': True, 'settings': [{'backjump': False}], 'totals': {'drift_checks': 0},
                      'boundRounds': [{'round_index': 1, 'ranges': records}], 'boundHistory': records,
                      'eventsByKind': {'bounds': records}}
            def publish(version):
                image.write_text(f'<svg xmlns="http://www.w3.org/2000/svg" width="{1050+version}" height="720"></svg>', encoding='utf-8')
                write_relu_search_tree_html(trace, tree, overall_status='RUNNING', mechanism_panels=[],
                    live_snapshots=True, snapshot_only=version > 1, snapshot_version=version,
                    title='LATEST TREE' if version > 1 else 'INITIAL TREE')
                write_solver_mechanisms_html(report, mechanisms, status='RUNNING', live_snapshots=True,
                    snapshot_only=version > 1, snapshot_version=version)
            publish(1)
            image.with_suffix('.selection.js').write_text('window.receiveAnalysisMap?.(document.currentScript,{"items":[],"regions":[]});', encoding='utf-8')
            page = root / 'relu_realtime.html'
            UnifiedRealtimeDashboard({'simplex': image}, page,
                embedded_pages={'relu-tree': tree, 'solver-mechanisms': mechanisms}).write()
            harness = r'''<script>(async()=>{
const $=id=>document.getElementById(id),assert=(p,m)=>{if(!p)throw Error(m);};
const wait=async(p,label)=>{for(let i=0;i<450;i++){if(p())return;await new Promise(r=>setTimeout(r,20));}throw Error('timeout '+label);};
const out=document.createElement('pre');out.id='input-test';
const payload=frame=>{const text=frame.contentDocument.querySelector('script').textContent;return JSON.parse(text.slice(text.indexOf('parent.postMessage(')+19,text.lastIndexOf(',"*");'))).data;};
try{
 await wait(()=>$('llm-connection').textContent.includes('gemini-'),'connection');
 const tf=$('relu-tree-dashboard'),mf=$('solver-mechanisms-dashboard'),image=$('simplex-dashboard');
 await wait(()=>tf.contentDocument?.querySelector('#treeSnapshotLoader')?.contentDocument?.querySelector('script')&&mf.contentDocument?.querySelector('#snapshotLoader')?.contentDocument?.querySelector('script'),'initial snapshots');
 await wait(()=>!panels[0].pending,'initial image');
 const td=tf.contentDocument,md=mf.contentDocument;
 const tl=td.querySelector('#treeSnapshotLoader'),ml=md.querySelector('#snapshotLoader');
 const treeData=payload(tl),mechanismData=payload(ml);
 const version=image.dataset.renderVersion,beforeCount=td.querySelector('#count').textContent;
 const question=$('llm-question');question.value='왜 이 노드에서 느린가요?';question.focus();question.setSelectionRange(4,4);
 question.dispatchEvent(new CompositionEvent('compositionstart',{bubbles:true,data:'한'}));
 question.dispatchEvent(new InputEvent('input',{bubbles:true,isComposing:true,data:'한'}));
 await new Promise(r=>setTimeout(r,30));
 treeData.snapshotVersion=2;treeData.title='LATEST TREE';treeData.nodeCount=2;
 mechanismData.snapshotVersion=2;mechanismData.report.totals.drift_checks=77;
 tf.contentWindow.dispatchEvent(new tf.contentWindow.MessageEvent('message',{source:tl.contentWindow,data:{type:'relu-search-tree-snapshot',data:treeData}}));
 mf.contentWindow.dispatchEvent(new mf.contentWindow.MessageEvent('message',{source:ml.contentWindow,data:{type:'solver-mechanisms-snapshot',data:mechanismData}}));
 await fetch(new URL('__advance',location.href));
 for(let i=0;i<5;i++)refresh();
 await new Promise(r=>setTimeout(r,1100));
 assert(window.logAnalysisInputBusy(),'Korean composition stays protected beyond the idle delay');
 assert(image.dataset.renderVersion===version,'image replacement is deferred while composing');
 assert(td.querySelector('#count').textContent===beforeCount,'in-flight tree update is buffered');
 assert(!md.querySelector('#numericalValue').textContent.includes('77'),'in-flight mechanism update is buffered');
 assert(md.querySelector('#boundStep').value==='1','automatic boundary playback does not rebuild graphs during composition');
 assert(question===document.activeElement&&question.value==='왜 이 노드에서 느린가요?'&&question.selectionStart===4&&question.selectionEnd===4,'draft, focus and cursor remain intact');
 question.dispatchEvent(new CompositionEvent('compositionend',{bubbles:true,data:'한'}));
 await wait(()=>!window.logAnalysisInputBusy(),'input idle');
 await wait(()=>image.naturalWidth===1052&&td.querySelector('#title').textContent==='LATEST TREE'&&md.querySelector('#numericalValue').textContent.includes('77'),'latest data catches up');
 assert(question===document.activeElement&&question.value==='왜 이 노드에서 느린가요?'&&question.selectionStart===4,'resuming updates never changes the draft or cursor');
 question.dispatchEvent(new InputEvent('input',{bubbles:true,data:'?'}));
 assert(window.logAnalysisInputBusy(),'ordinary typing is also protected');
 question.blur();await wait(()=>!window.logAnalysisInputBusy(),'blur resumes immediately');
 out.textContent='PASS';
}catch(error){out.textContent='FAIL '+error.message;}document.body.append(out);
})();</script>'''
            page.write_text(page.read_text(encoding='utf-8').replace('</body>', harness + '</body>'), encoding='utf-8')
            app = AnalysisApp(root, output_dir=root / 'analysis', client_factory=lambda _progress: None)
            server = make_server(app)
            original = server.RequestHandlerClass.do_GET
            def get(handler):
                if urlsplit(handler.path).path.endswith('/__advance'):
                    trace.events.append(TraceEvent('reluplex_branch', 2, None, 2, 'z2_0',
                        {'phase': '+', 'result': 'RUNNING', 'round_index': 1}))
                    report['totals']['drift_checks'] = 77
                    publish(2)
                    handler.send({'ok': True})
                else:
                    original(handler)
            server.RequestHandlerClass.do_GET = get
            worker = threading.Thread(target=server.serve_forever, daemon=True)
            worker.start()
            try:
                process = subprocess.run([chromium_path(), '--headless', '--no-sandbox', '--disable-gpu', '--no-first-run',
                    f'--user-data-dir={root / "profile"}', '--virtual-time-budget=15000', '--dump-dom',
                    f'http://127.0.0.1:{server.server_port}/s/{app.token}/'], capture_output=True, timeout=35)
                dom = process.stdout.decode('utf-8', errors='replace')
                match = re.search(r'<pre id="input-test">([^<]+)</pre>', dom)
                self.assertIsNotNone(match, dom[-1500:])
                self.assertEqual(match[1], 'PASS')
                self.assertFalse(app.jobs, 'typing and resuming must never send logs to Gemini')
            finally:
                server.shutdown();server.server_close();worker.join(timeout=5);app.close()


if __name__ == '__main__':
    unittest.main()
