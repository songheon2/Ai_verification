"""Offline checks that unchanged assets do not trigger decode/DOM rebuilds."""
from collections import Counter
from http.client import HTTPConnection
import json
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
from visualization.UnifiedRealtimeDashboard import UnifiedRealtimeDashboard, publish_dashboard_status


class VisualizationRefreshTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.app = AnalysisApp(self.root, client_factory=lambda _progress: None,
                               output_dir=self.root / 'analysis')
        self.server = make_server(self.app)
        self.base = f'/s/{self.app.token}/'
        self.counts = Counter()
        self.advance = lambda: None
        original = self.server.RequestHandlerClass.do_GET
        def observed(handler):
            path = urlsplit(handler.path).path.removeprefix(self.base)
            self.counts[handler.command + ' ' + path] += 1
            if path == '__test_counts':
                handler.send(dict(self.counts))
            elif path == '__test_advance':
                self.advance()
                handler.send({'ok': True})
            else:
                original(handler)
        self.server.RequestHandlerClass.do_GET = observed
        self.worker = threading.Thread(target=self.server.serve_forever, daemon=True)
        self.worker.start()
        self.addCleanup(self.close)

    def close(self):
        self.server.shutdown()
        self.server.server_close()
        self.worker.join(timeout=5)
        self.app.close()

    def request(self, path, method='GET', headers=None):
        connection = HTTPConnection('127.0.0.1', self.server.server_port, timeout=5)
        try:
            connection.request(method, self.base + path, headers=headers or {})
            response = connection.getresponse()
            return response.status, dict(response.getheaders()), response.read()
        finally:
            connection.close()

    def browser(self, harness):
        page = self.root / 'relu_realtime.html'
        page.write_text(page.read_text(encoding='utf-8').replace('</body>', harness + '</body>'), encoding='utf-8')
        process = subprocess.run([chromium_path(), '--headless', '--no-sandbox', '--disable-gpu', '--no-first-run',
            f'--user-data-dir={self.root / "profile"}', '--virtual-time-budget=18000', '--dump-dom',
            f'http://127.0.0.1:{self.server.server_port}{self.base}'], capture_output=True, timeout=40)
        dom = process.stdout.decode('utf-8', errors='replace')
        match = re.search(r'<pre id="refresh-test">([^<]+)</pre>', dom)
        self.assertIsNotNone(match, dom[-1500:])
        self.assertEqual(match[1], 'PASS')
        self.assertFalse(self.app.jobs, 'polling must not call Gemini')

    def test_head_revalidates_images_scripts_and_snapshot_files_without_bodies(self):
        for name in ('plot.png', 'plot.svg', 'plot.hover.js', 'tree.snapshot.html'):
            with self.subTest(name=name):
                path = self.root / name
                path.write_bytes(b'first snapshot')
                status, headers, body = self.request(name, 'HEAD')
                self.assertEqual((status, body), (200, b''))
                self.assertEqual(int(headers['Content-Length']), path.stat().st_size)
                self.assertEqual(self.request(name, 'HEAD', {'If-None-Match': headers['ETag']})[::2], (304, b''))
                path.write_bytes(b'changed snapshot')
                status, updated, body = self.request(name, 'HEAD', {'If-None-Match': headers['ETag']})
                self.assertEqual((status, body), (200, b''))
                self.assertNotEqual(updated['ETag'], headers['ETag'])
                self.assertEqual(self.request(name)[::2], (200, b'changed snapshot'))
        self.assertEqual(self.request('result.json', 'HEAD')[0], 403)
        self.assertEqual(self.request('../outside.html', 'HEAD')[0], 403)

    @unittest.skipUnless(chromium_path(), 'Chromium is not installed')
    def test_worker_error_appears_even_when_no_image_was_published(self):
        dashboard = self.root / 'relu_realtime.html'
        UnifiedRealtimeDashboard({'simplex': self.root / 'missing.png'}, dashboard,
                                 managed_completion=True).write()
        self.advance = lambda: publish_dashboard_status(dashboard, 'ERROR', message='GLPK DLL blocked')
        self.browser(r'''<script>(async()=>{
const out=document.createElement('pre');out.id='refresh-test';
try{
 await fetch('__test_advance');
 refresh();
 for(let i=0;i<200&&!verificationComplete;i++){
   await fetch('__test_counts');
   await new Promise(resolve=>setTimeout(resolve,20));
 }
 if(!verificationComplete || document.getElementById('status').textContent!=='DONE · ERROR · GLPK DLL blocked')
   throw Error('the worker error and reason should end refresh');
 out.textContent='PASS';
}catch(error){out.textContent='FAIL '+error.message;}document.body.append(out);
})();</script>''')

    @unittest.skipUnless(chromium_path(), 'Chromium is not installed')
    def test_unchanged_image_skips_download_and_keeps_clicks_through_failed_map_update(self):
        image = self.root / 'plot.svg'
        sidecar = image.with_suffix('.selection.js')
        def write_image(width):
            image.write_text(f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="720"></svg>', encoding='utf-8')
        def write_map(call):
            payload = {'items': [{'scope': {'topic': 'simplex', 'call_id': call}, 'text': call}],
                       'regions': [{'box': [0, 0, 1, 1], 'positions': [], 'direction': 'x', 'items': [0], 'label': '호출'}]}
            sidecar.write_text('window.receiveAnalysisMap?.(document.currentScript,' + json.dumps(payload) + ');', encoding='utf-8')
        write_image(1050)
        write_map('simplex-old')
        stage = [0]
        def advance():
            stage[0] += 1
            if stage[0] == 1:
                write_image(1200)
                sidecar.unlink()
            else:
                write_map('simplex-new')
        self.advance = advance
        UnifiedRealtimeDashboard({'simplex': image}, self.root / 'relu_realtime.html').write()
        self.browser(r'''<script>(async()=>{
const $=id=>document.getElementById(id),assert=(p,m)=>{if(!p)throw Error(m);};
const get=async path=>(await fetch(new URL(path,location.href))).json();
const wait=async(p,label)=>{for(let i=0;i<350;i++){if(p())return;await new Promise(r=>setTimeout(r,20));}throw Error('timeout '+label);};
const out=document.createElement('pre');out.id='refresh-test';
try{
 clearInterval(imageTimer);clearInterval(pageTimer);
 const panel=panels[0],image=$('simplex-dashboard');
 await wait(()=>!panel.pending&&image.style.cursor==='pointer','initial image');
 const before=await get('__test_counts'),version=image.dataset.renderVersion,src=image.src;
 for(let i=0;i<5;i++){await refreshPanel(panel);await wait(()=>!panel.pending,'poll');}
 const after=await get('__test_counts');
 assert(after['GET plot.svg']===before['GET plot.svg'],'unchanged image is not downloaded again');
 assert(after['HEAD plot.svg']>=before['HEAD plot.svg']+5,'all five polls still check for updates');
 assert(image.src===src&&image.dataset.renderVersion===version,'unchanged frame is not repainted');
 const click=()=>{const box=image.getBoundingClientRect();image.dispatchEvent(new MouseEvent('click',{clientX:box.left+box.width/2,clientY:box.top+box.height/2}));};
 click();assert($('llm-call').value==='simplex-old','cached frame still responds immediately');
 await get('__test_advance');await refreshPanel(panel);await wait(()=>!panel.pending,'missing map');
 assert(image.dataset.renderVersion===version,'failed metadata retains old frame');
 click();assert($('llm-call').value==='simplex-old','failed refresh retains working clicks');
 await get('__test_advance');await refreshPanel(panel);await wait(()=>!panel.pending,'retry');
 assert(image.naturalWidth===1200&&image.dataset.renderVersion!==version,'retry does not skip the new frame');
 click();assert($('llm-call').value==='simplex-new','new frame uses new click metadata');
 const fresh=image.dataset.renderVersion;await refreshPanel(panel,true);await wait(()=>!panel.pending,'final refresh');
 assert(image.dataset.renderVersion!==fresh,'final snapshot is never suppressed');
 out.textContent='PASS';
}catch(error){out.textContent='FAIL '+error.message;}document.body.append(out);
})();</script>''')

    @unittest.skipUnless(chromium_path(), 'Chromium is not installed')
    def test_tree_can_finish_before_final_dpll_image_without_stopping_refresh(self):
        dashboard = self.root / 'relu_realtime.html'
        image = self.root / 'dpll.svg'
        sidecar = image.with_suffix('.selection.js')
        tree = self.root / 'tree.html'
        tree.write_text('<html><body>Tree</body></html>', encoding='utf-8')

        def write_frame(width):
            image.write_text(f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="720"></svg>',
                             encoding='utf-8')

        def write_map():
            sidecar.write_text('window.receiveAnalysisMap?.(document.currentScript,{items:[],regions:[]});',
                               encoding='utf-8')

        write_frame(1050)
        write_map()
        UnifiedRealtimeDashboard({'dpll-theory': image}, dashboard, embedded_pages={'relu-tree': tree},
                                 managed_completion=True).write()
        stage = [0]

        def advance():
            stage[0] += 1
            if stage[0] == 1:
                write_frame(1200)
                sidecar.unlink()
                publish_dashboard_status(dashboard, 'UNSAT', decided_at='2026-11-23T09:04:12+09:00',
                                         elapsed_seconds=38.42)
            else:
                write_map()

        self.advance = advance
        self.browser(r'''<script>(async()=>{
const $=id=>document.getElementById(id),assert=(p,m)=>{if(!p)throw Error(m);};
const advance=()=>fetch(new URL('__test_advance',location.href));
const wait=async(p,label)=>{for(let i=0;i<350;i++){if(p())return;await new Promise(r=>setTimeout(r,20));}throw Error('timeout '+label);};
const out=document.createElement('pre');out.id='refresh-test';
try{
 const image=$('dpll-theory-dashboard'),frame=$('relu-tree-dashboard');
 await wait(()=>image.naturalWidth===1050&&panels[0].hasSelectionMap&&frame.dataset.loaded,'initial frames');
 frame.contentWindow.eval("parent.postMessage({type:'relu-search-tree-status',status:'UNSAT'},'*')");
 await new Promise(r=>setTimeout(r,100));
 assert(!verificationComplete,'early terminal tree must leave refresh running');
 await advance();
 await wait(()=>completionStatus==='UNSAT'&&!panels[0].pending,'first final refresh');
 assert(!verificationComplete&&image.naturalWidth===1050,'temporarily missing selection map keeps final refresh pending');
 await advance();
 await wait(()=>verificationComplete,'final frame publication');
 assert(image.naturalWidth===1200&&$('status').textContent==='DONE · UNSAT · 38.4초','terminal status follows final frame');
 const expectedDate=new Intl.DateTimeFormat('sv-SE',{dateStyle:'short',timeStyle:'medium'})
   .format(new Date('2026-11-23T09:04:12+09:00'));
 assert($('status').title===expectedDate,'terminal status shows the recorded decision timestamp');
 out.textContent='PASS';
}catch(error){out.textContent='FAIL '+error.message;}document.body.append(out);
})();</script>''')

    @unittest.skipUnless(chromium_path(), 'Chromium is not installed')
    def test_unchanged_tree_snapshot_skips_iframe_reload_and_still_accepts_updates(self):
        trace = SolveTrace(events=[TraceEvent('reluplex_branch', 1, None, 1, 'z1_3',
            {'phase': '+', 'result': 'RUNNING', 'layer': 1, 'index': 3})])
        tree = self.root / 'tree.html'
        write_relu_search_tree_html(trace, tree, overall_status='RUNNING', mechanism_panels=[],
                                   live_snapshots=True, snapshot_version=1)
        version = [1]
        def advance():
            version[0] += 1
            trace.events.append(TraceEvent('reluplex_branch', 2, None, 2, 'z2_8',
                {'phase': '+', 'result': 'RUNNING', 'layer': 2, 'index': 8}))
            write_relu_search_tree_html(trace, tree, overall_status='RUNNING', mechanism_panels=[],
                                       live_snapshots=True, snapshot_only=True, snapshot_version=version[0])
        self.advance = advance
        UnifiedRealtimeDashboard({}, self.root / 'relu_realtime.html', embedded_pages={'relu-tree': tree}).write()
        self.browser(r'''<script>(async()=>{
const assert=(p,m)=>{if(!p)throw Error(m);},frame=document.getElementById('relu-tree-dashboard');
const get=async path=>(await fetch(new URL(path,location.href))).json();
const wait=async(p,label)=>{for(let i=0;i<350;i++){if(p())return;await new Promise(r=>setTimeout(r,20));}throw Error('timeout '+label);};
const out=document.createElement('pre');out.id='refresh-test';
try{
 clearInterval(imageTimer);clearInterval(pageTimer);
 await wait(()=>frame.contentDocument?.querySelector('#tree li'),'initial tree');
 const doc=frame.contentDocument,root=doc.querySelector('#tree li');
 const refresh=()=>frame.contentWindow.postMessage({type:'relu-search-tree-refresh'},'*');
 refresh();await wait(()=>doc.querySelector('#treeSnapshotLoader')?.contentDocument?.querySelector('script'),'first snapshot');
 await new Promise(r=>setTimeout(r,50));const before=await get('__test_counts');
 let sharedTicks=0;
 frame.contentWindow.addEventListener('message',event=>{if(event.source===window&&event.data?.type==='relu-search-tree-refresh')sharedTicks++;});
 refresh();await wait(()=>sharedTicks>0,'shared image/tree refresh tick');
 for(let i=0;i<4;i++){
   refresh();await new Promise(r=>setTimeout(r,60));
 }
 const after=await get('__test_counts');
 assert(after['GET tree.snapshot.html']===before['GET tree.snapshot.html'],'unchanged tree snapshot is not parsed through an iframe again');
 assert(after['HEAD tree.snapshot.html']>before['HEAD tree.snapshot.html'],'tree polling remains active');
 assert(doc.querySelector('#tree li')===root,'unchanged tree preserves its DOM');
 const originalFetch=frame.contentWindow.fetch.bind(frame.contentWindow);let release,held=false;
 frame.contentWindow.fetch=(...args)=>{
   if(!held&&args[1]?.method==='HEAD'){
     held=true;return originalFetch(...args).then(response=>new Promise(resolve=>{release=()=>resolve(response);}));
   }
   return originalFetch(...args);
 };
 refresh();await wait(()=>Boolean(release),'held update check');
 await get('__test_advance');
 frame.contentWindow.postMessage({type:'relu-search-tree-refresh',final:true},'*');
 await new Promise(r=>setTimeout(r,20));release();
 await wait(()=>doc.querySelector('#count').textContent.includes('가지 2개'),'queued final snapshot');
 assert(frame.contentDocument===doc,'updated snapshot keeps controls and document');
 let stalled=false;
 frame.contentWindow.fetch=(...args)=>{
   if(!stalled&&args[1]?.method==='HEAD'){stalled=true;return new Promise(()=>{});}
   return originalFetch(...args);
 };
 await get('__test_advance');refresh();
 await wait(()=>doc.querySelector('#count').textContent.includes('가지 3개'),'automatic recovery from stalled refresh');
 out.textContent='PASS';
}catch(error){out.textContent='FAIL '+error.message;}document.body.append(out);
})();</script>''')


if __name__ == '__main__':
    unittest.main()
