"""Chart selections fill local prompts without sending any LLM request."""
import json
from http.client import HTTPConnection
from pathlib import Path
import re
import subprocess
import tempfile
import threading
import unittest

from Automation.LogAnalysisServer import AnalysisApp, make_server
from Automation.tests.test_reactive_tree_browser import chromium_path
from visualization.AnalysisSelection import write_selection_map
from visualization.RealtimeSplitVisualization import SplitLogEvent, draw_realtime_split_dashboard
from visualization.ReluNeuronHover import write_relu_hover_map
from visualization.ReluSearchTree import write_relu_search_tree_html
from visualization.SolveTrace import SolveTrace, TraceEvent
from visualization.SolverMechanisms import write_solver_mechanisms_html
from visualization.SolverProgress import build_solver_feedback, draw_dpll_theory_panel, draw_simplex_panel
from visualization.UnifiedRealtimeDashboard import UnifiedRealtimeDashboard


def feedback_fixture():
    events = [{"event": "round_start", "round": 3},
              {"event": "theory_result", "round": 3, "result": "CONFLICT"},
              {"event": "round_end", "round": 3, "result": "UNSAT", "duration_seconds": 25}]
    for number in range(1, 16):
        events.append({"event": "simplex_call", "call_id": f"simplex-{100 + number}",
                       "call_number": 200 + number, "round": 3, "depth": number % 4,
                       "result": "UNSAT" if number == 12 else "SAT", "duration_seconds": number / 2,
                       "pivots": number * 7, "iterations": number * 7 + 1})
    feedback = build_solver_feedback(events)
    feedback["top_violated_constraints"] = [{"slack_name": "s7", "constraint_id": "c7", "count": 12}]
    return feedback


def read_map(path):
    text = path.read_text(encoding="utf-8")
    return json.loads(text[len("window.receiveAnalysisMap?.(document.currentScript,"):-2])


class AnalysisSelectionTests(unittest.TestCase):
    @unittest.skipUnless(chromium_path(), "Chromium browser is not installed")
    def test_one_click_works_during_metadata_loading_and_pointer_spanning_refresh(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            image = root / 'plot.svg'
            image.write_text('<svg xmlns="http://www.w3.org/2000/svg" width="1050" height="720"><rect width="1050" height="720" fill="white"/></svg>', encoding='utf-8')
            payload = {'items': [{'scope': {'topic': 'simplex', 'call_id': 'simplex-old'}, 'text': '기존 스냅샷'}],
                       'regions': [{'box': [0, 0, 1, 1], 'positions': [], 'direction': 'x', 'items': [0], 'label': '호출'}]}
            image.with_suffix('.selection.js').write_text('window.receiveAnalysisMap?.(document.currentScript,' + json.dumps(payload) + ');', encoding='utf-8')
            dashboard = root / 'dashboard.html'
            UnifiedRealtimeDashboard({'simplex': image}, dashboard).write()
            harness = r'''<script>(async()=>{
const $=id=>document.getElementById(id),assert=(p,m)=>{if(!p)throw Error(m);};
const wait=async(p,label)=>{for(let i=0;i<200;i++){if(p())return;await new Promise(r=>setTimeout(r,10));}throw Error('timeout '+label);};
const out=document.createElement('pre');out.id='single-click-test';
try{
 const panel=panels[0];await wait(()=>!panel.pending&&$('simplex-dashboard').style.cursor==='pointer','initial map');
 clearInterval(imageTimer);
 const image=$('simplex-dashboard'),src=image.src,version=image.dataset.renderVersion,box=image.getBoundingClientRect();
 const point={clientX:box.left+box.width/2,clientY:box.top+box.height/2,bubbles:true,button:0};
 let selections=0;window.addEventListener('log-analysis-select',()=>selections++);
 const click=()=>image.dispatchEvent(new MouseEvent('click',point));
 const original=window.loadAnalysisSelection;let release;
 window.loadAnalysisSelection=()=>new Promise(resolve=>{release=()=>resolve(__NEXT__);});
 refreshPanel(panel);await wait(()=>Boolean(release),'delayed metadata');
 assert($('simplex-dashboard')===image&&image.src===src,'old frame stays usable until metadata is ready');
 click();assert($('llm-call').value==='simplex-old'&&selections===1,'one click during loading is processed immediately');
 image.dispatchEvent(new PointerEvent('pointerdown',{...point,pointerId:7,pointerType:'mouse'}));
 release();await wait(()=>!panel.pending&&image.dataset.renderVersion!==version,'new frame');
 assert($('simplex-dashboard')===image,'pointer target is not replaced');
 image.dispatchEvent(new PointerEvent('pointerup',{...point,pointerId:7,pointerType:'mouse'}));click();
 assert($('llm-call').value==='simplex-old'&&selections===2,'refresh during press keeps the originally pressed item');
 click();assert($('llm-call').value==='simplex-new'&&selections===3,'next single click uses the new snapshot');
 const freshVersion=image.dataset.renderVersion;
 window.loadAnalysisSelection=()=>Promise.resolve(null);refreshPanel(panel);
 await wait(()=>!panel.pending,'failed metadata update');
 assert(image.dataset.renderVersion===freshVersion,'failed metadata does not replace the clickable snapshot');
 click();assert($('llm-call').value==='simplex-new'&&selections===4,'transient update failure never creates a dead click window');
 image.dispatchEvent(new PointerEvent('pointerdown',{...point,pointerId:8,pointerType:'mouse'}));
 image.dispatchEvent(new MouseEvent('click',{...point,clientX:point.clientX+30}));
 assert(selections===4,'dragging is not an item selection');
 window.loadAnalysisSelection=original;
 assert($('llm-start').disabled,'click still cannot upload without explicit analysis');
 out.textContent='PASS';
}catch(error){out.textContent='FAIL '+error.message;}document.body.appendChild(out);
})();</script>'''
            fresh = {'items': [{'scope': {'topic': 'simplex', 'call_id': 'simplex-new'}, 'text': '새 스냅샷'}], 'regions': payload['regions']}
            harness = harness.replace('__NEXT__', json.dumps(fresh))
            dashboard.write_text(dashboard.read_text(encoding='utf-8').replace('</body>', harness + '</body>'), encoding='utf-8')
            result = subprocess.run([chromium_path(), '--headless', '--no-sandbox', '--disable-gpu', '--no-first-run',
                f'--user-data-dir={root / "profile"}', '--virtual-time-budget=10000', '--dump-dom', dashboard.as_uri()], capture_output=True, timeout=30)
            dom = result.stdout.decode('utf-8', errors='replace')
            match = re.search(r'<pre id="single-click-test">([^<]+)</pre>', dom)
            self.assertIsNotNone(match, dom[-1500:])
            self.assertEqual(match[1], 'PASS')

    def test_empty_new_render_clears_previous_click_targets(self):
        import matplotlib.pyplot as plt
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'simplex.png'
            for feedback in (feedback_fixture(), build_solver_feedback([])):
                figure, _ = draw_simplex_panel(feedback)
                try:
                    figure.savefig(path, dpi=100)
                    payload = read_map(write_selection_map(figure, path))
                finally:
                    plt.close(figure)
            self.assertEqual(payload, {'items': [], 'regions': []})

    def test_map_preserves_plot_order_explicit_ids_and_all_quadrants(self):
        import matplotlib.pyplot as plt
        with tempfile.TemporaryDirectory() as directory:
            for renderer in (draw_simplex_panel, draw_dpll_theory_panel):
                figure, _ = renderer(feedback_fixture())
                try:
                    for cropped in (False, True):
                        path = Path(directory) / "plot.png"
                        figure.savefig(path, dpi=100, **({"bbox_inches": "tight"} if cropped else {}))
                        data = read_map(write_selection_map(figure, path, cropped=cropped))
                        self.assertEqual(len(data["regions"]), 4)
                        for region in data["regions"]:
                            left, top, right, bottom = region["box"]
                            self.assertTrue(0 <= left < right <= 1)
                            self.assertTrue(0 <= top < bottom <= 1)
                            self.assertTrue(region["targets"])
                        if renderer == draw_simplex_panel:
                            region = data["regions"][0]
                            item = data["items"][region["items"][11]]
                            self.assertEqual(item["scope"], {"topic": "simplex", "round": 3, "call_id": "simplex-112"})
                            self.assertIn("그래프 12번째", item["text"])
                            self.assertIn("피벗 횟수: 84", item["text"])
                            self.assertNotIn("simplex-12\n", item["text"])
                            self.assertEqual(data["regions"][1]["items"], region["items"])
                finally:
                    plt.close(figure)

    def test_renderer_writes_map_in_the_normal_realtime_output_path(self):
        from visualization.RenderProcess import _replace_figure
        with tempfile.TemporaryDirectory() as directory:
            image = Path(directory) / "simplex.png"
            figure, _ = draw_simplex_panel(feedback_fixture())
            _replace_figure(figure, str(image), 100)
            self.assertTrue(image.exists())
            self.assertTrue(image.with_suffix('.selection.js').exists())

    def test_counterexample_text_carries_its_actual_formula_input_and_output(self):
        import matplotlib.pyplot as plt
        feedback = feedback_fixture()
        feedback['solver'] = {'status': 'SAT', 'model': {'x': 0.25, 'y': 1.5},
            'witness_context': {'inputs': {'X_0': {'coeffs': {'x': 1}, 'constant': 0}}, 'outputs': {'Y_0': 'y'}}}
        feedback['theory_links'] = [{'result': 'THEORY_SAT', 'inequalities': [{'coeffs': {'y': 1}, 'lower': 1}]}]
        with tempfile.TemporaryDirectory() as directory:
            figure, _ = draw_dpll_theory_panel(feedback)
            try:
                path = Path(directory) / 'dpll.png'
                figure.savefig(path, dpi=150, bbox_inches='tight')
                data = read_map(write_selection_map(figure, path, cropped=True))
                text = '\n'.join(item['text'] for item in data['items'])
                self.assertIn('VNNLIB\n(Y_0 >= 1)', text)
                self.assertIn('Counter example input\nX_0 = 0.25', text)
                self.assertIn('Network output\nY_0 = 1.5', text)
            finally:
                plt.close(figure)

    def test_selection_map_cache_revalidates_and_updates_without_api_calls(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            file = root / 'plot.selection.js'
            file.write_text('first', encoding='utf-8')
            app = AnalysisApp(root, output_dir=root / 'analysis', client_factory=lambda _progress: None)
            server = make_server(app)
            thread = threading.Thread(target=server.serve_forever, daemon=True)
            thread.start()
            def get(etag=None):
                connection = HTTPConnection('127.0.0.1', server.server_port, timeout=5)
                try:
                    connection.request('GET', f'/s/{app.token}/plot.selection.js', headers={'If-None-Match': etag} if etag else {})
                    response = connection.getresponse()
                    return response.status, dict(response.getheaders()), response.read()
                finally:
                    connection.close()
            try:
                status, headers, body = get()
                self.assertEqual((status, body), (200, b'first'))
                self.assertEqual(headers['Cache-Control'], 'private, no-cache')
                self.assertEqual(get(headers['ETag'])[::2], (304, b''))
                file.write_text('second snapshot', encoding='utf-8')
                status, updated, body = get(headers['ETag'])
                self.assertEqual((status, body), (200, b'second snapshot'))
                self.assertNotEqual(headers['ETag'], updated['ETag'])
                self.assertFalse(app.jobs)
            finally:
                server.shutdown();server.server_close();thread.join(timeout=5);app.close()

    @unittest.skipUnless(chromium_path(), "Chromium browser is not installed")
    def test_clicks_fill_prompt_keep_manual_question_and_do_not_upload(self):
        import matplotlib.pyplot as plt
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            images, maps = {}, {}
            for name, renderer in (("simplex", draw_simplex_panel), ("dpll-theory", draw_dpll_theory_panel)):
                figure, _ = renderer(feedback_fixture())
                try:
                    images[name] = root / f"{name}.png"
                    figure.savefig(images[name], dpi=100)
                    maps[name] = read_map(write_selection_map(figure, images[name]))
                finally:
                    plt.close(figure)
            events = [SplitLogEvent('', 1000000000, 'a', 1, 3, 'z1_3', '+')]
            figure, _ = draw_realtime_split_dashboard(root / 'events.jsonl', events=events,
                current_time_ns=2000000000, round_starts=[{"round": 3, "time_ns": 0}])
            try:
                images['relu'] = root / 'relu.png'
                figure.savefig(images['relu'], dpi=100)
                hover = write_relu_hover_map(figure, images['relu'])
                maps['relu'] = read_map(write_selection_map(figure, images['relu']))
                points = json.loads(hover.read_text(encoding='utf-8')[len('window.postMessage('):-len(',"*");')])['points']
            finally:
                plt.close(figure)
            tree = root / 'tree.html'
            trace = SolveTrace(events=[TraceEvent('reluplex_branch', 1, 2, 1, 'z1_3',
                {'phase': '+', 'result': 'UNSAT', 'layer': 1, 'index': 3, 'round_index': 3, 'split_number': 9})])
            write_relu_search_tree_html(trace, tree, overall_status='UNSAT', mechanism_panels=[])
            record = {"kind": "stable", "sequence": 17, "id": 17, "variable": "z1_3", "round_index": 3,
                      "depth": 2, "source": "search", "state": "active", "before": [-1, 2], "after": [0, 2]}
            numeric = {"kind": "numerical", "sequence": 22, "round_index": 3, "depth": 2, "action": "check",
                       "drift": 0.01, "residual": 0.02, "outcome": "within_tolerance"}
            report = {"available": True, "settings": [{"backjump": False, "drift_threshold": 0.1}], "totals": {"conflict_LP": 2},
                "boundRounds": [{"round_index": 3, "ranges": [record]}], "boundHistory": [record],
                "eventsByKind": {"bounds": [record], "numerical": [numeric]}, "numericalChecks": [numeric]}
            mechanisms = root / 'mechanisms.html'
            write_solver_mechanisms_html(report, mechanisms, status='UNSAT')
            dashboard = root / 'dashboard.html'
            UnifiedRealtimeDashboard(images, dashboard, embedded_pages={"relu-tree": tree, "solver-mechanisms": mechanisms}).write()
            harness = r'''<script>(async()=>{
const $=id=>document.getElementById(id),assert=(p,m)=>{if(!p)throw Error(m);};
const wait=async(p,label)=>{for(let i=0;i<200;i++){if(p())return;await new Promise(r=>setTimeout(r,20));}throw Error('timeout '+label);};
const out=document.createElement('pre');out.id='selection-test';let requests=0;
let selections=0;window.addEventListener('log-analysis-select',()=>selections++);
const fetchOriginal=window.fetch;window.fetch=(url,options)=>{if(options?.method==='POST')requests++;return fetchOriginal(url,options);};
const map=__MAP__,dpllMap=__DPLL_MAP__,reluMap=__RELU_MAP__,points=__POINTS__;
const click=(id,x,y)=>{
 const image=$(id),box=visualizationImageBox(image);
 const event=new PointerEvent('click');
 // The synthetic constructor truncates CSS coordinates, which can miss subpixel strokes.
 Object.defineProperties(event,{clientX:{value:box.left+box.width*x},clientY:{value:box.top+box.height*y}});
 image.dispatchEvent(event);
};
const hit=(id,region,index,verticalFraction=0.5)=>{
 const [,,x0,y0,x1,y1]=region.targets.findLast(target=>target[0]===index);
 click(id,(x0+x1)/2,y0+(y1-y0)*verticalFraction);
};
try{
 await wait(()=>$('simplex-dashboard').dataset.renderVersion,'Simplex hit map');
 $('llm-multi-select').click();
 const selectedText=()=>[...$('llm-selected-items').querySelectorAll('span')].map(span=>span.title).join('\n');
 $('llm-question').value='왜 오래 걸렸나요?';
 hit('simplex-dashboard',map.regions[0],11);
 assert($('llm-question').value.includes('왜 오래 걸렸나요?'),'manual question preserved');
 assert(selectedText().includes('12번째')&&selectedText().includes('84'),'correct clicked call metrics');
 assert($('llm-topic').value==='simplex'&&$('llm-call').value==='simplex-112'&&$('llm-round').value==='3','explicit call ID and round, not ordinal');
 const short=map.regions[0].targets.find(target=>target[0]===0),next=map.regions[0].targets.find(target=>target[0]===1);
 const work=map.regions[1].box;
 const selected=selectedText();
 click('simplex-dashboard',short[2],(map.regions[0].box[1]+short[5])/2);
 click('simplex-dashboard',(short[2]+next[2])/2,short[3]+(short[5]-short[3])/2);
 click('simplex-dashboard',(work[0]+work[2])/2,work[1]+0.002);
 assert(selectedText()===selected&&$('llm-call').value==='simplex-112','white space above and between marks is ignored');
 const once=selectedText();hit('simplex-dashboard',map.regions[0],11);
 assert(selectedText()===once,'repeated click must not duplicate context');
 document.querySelector('.simplex-section').style.width='500px';hit('simplex-dashboard',map.regions[1],12);
 assert($('llm-call').value==='simplex-113'&&selectedText().includes('13번째'),'responsive coordinates');
 assert(!selectedText().includes('12번째'),'old generated context replaced');
 const previous=$('simplex-dashboard'),version=previous.dataset.renderVersion;refreshPanel(panels.find(panel=>panel.id==='simplex-dashboard'),true);
 await wait(()=>previous.dataset.renderVersion!==version,'refreshed hit map');
 assert($('simplex-dashboard')===previous,'refresh keeps the clickable element stable');
 hit('simplex-dashboard',map.regions[0],11);assert($('llm-call').value==='simplex-112','refresh preserves click support');
 hit('simplex-dashboard',map.regions[2],1);assert(selectedText().includes('UNSAT'),'Simplex result bar');
 hit('simplex-dashboard',map.regions[3],0);assert(selectedText().includes('경계 위반 반복'),'repeated bound variable');
 await wait(()=>$('dpll-theory-dashboard').dataset.renderVersion,'DPLL map');
 hit('dpll-theory-dashboard',dpllMap.regions[0],0,0.15);assert($('llm-topic').value==='dpll'&&$('llm-round').value==='3','DPLL round');
 const afterBar=selections;
 const badge=dpllMap.regions[0].targets.find(target=>target[0]===-1);
 click('dpll-theory-dashboard',(badge[2]+badge[4])/2,(badge[3]+badge[5])/2);
 assert(selections===afterBar,'white annotation box is ignored');
 hit('dpll-theory-dashboard',dpllMap.regions[3],0);assert(selectedText().includes('selected atoms'),'round theory flow');
 await wait(()=>$('relu-dashboard').dataset.renderVersion,'ReLU history map');
 hit('relu-dashboard',reluMap.regions[0],0);assert($('llm-topic').value==='relu'&&$('llm-round').value==='3','ReLU timeline event');
 const relu=$('relu-dashboard'),box=relu.getBoundingClientRect(),point=points[0];
 const beforePoint=selections;
 click('relu-dashboard',point.x+point.rx+0.001,point.y);
 assert(selections===beforePoint,'white space beside a ReLU square is ignored');
 await wait(()=>{relu.dispatchEvent(new MouseEvent('click',{clientX:box.left+box.width*point.x,clientY:box.top+box.height*point.y}));return $('llm-topic').value==='relu';},'ReLU point');
 assert($('llm-layer').value==='1'&&$('llm-neuron').value==='3','neuron selected');
 const frame=$('relu-tree-dashboard');await wait(()=>frame.contentDocument?.querySelector('li summary'),'tree');
 frame.contentDocument.querySelector('li summary').click();await wait(()=>$('llm-topic').value==='tree','tree prompt');
 assert(selectedText().includes('노드 #')&&$('llm-layer').value==='1','tree context');
 const mechanism=$('solver-mechanisms-dashboard');await wait(()=>mechanism.contentDocument?.querySelector('.chart svg'),'mechanisms');
 mechanism.contentDocument.querySelector('.chart svg').dispatchEvent(new MouseEvent('click',{bubbles:true}));
 await new Promise(r=>setTimeout(r,30));assert($('llm-topic').value==='tree','empty SVG area does not replace the selection');
 mechanism.contentDocument.querySelector('.chart g[role="button"] line').dispatchEvent(new MouseEvent('click',{bubbles:true}));
 await wait(()=>$('llm-topic').value==='mechanisms','mechanisms prompt');
 assert(document.querySelector('[data-mechanism="bounds"]').getAttribute('aria-checked')==='true'&&$('llm-neuron').value==='3','bound mechanism and neuron');
 assert(selectedText().includes('변경 전')&&selectedText().includes('17'),'selected bound change');
 mechanism.contentDocument.querySelectorAll('.card')[2].querySelector('circle[role="button"]').dispatchEvent(new MouseEvent('click',{bubbles:true}));
 await wait(()=>document.querySelector('[data-mechanism="numeric"]').getAttribute('aria-checked')==='true','numeric prompt');
 assert(selectedText().includes('22')&&selectedText().includes('0.01'),'numeric sample');
 mechanism.contentDocument.querySelectorAll('.card')[1].querySelector('[role="button"]').dispatchEvent(new MouseEvent('click',{bubbles:true}));
 await wait(()=>document.querySelector('[data-mechanism="backjump"]').getAttribute('aria-checked')==='true','conflict bar');
 assert(selectedText().includes('2건'),'conflict count');
 mechanism.contentDocument.querySelectorAll('.card')[3].querySelector('.chart g[role="button"] line')
  .dispatchEvent(new MouseEvent('click',{bubbles:true}));
 await wait(()=>document.querySelector('[data-mechanism="underapprox"]').getAttribute('aria-checked')==='true','underapprox preview');
 assert(selectedText().includes('실제 솔버 적용 결과가 아닙니다'),'preview not execution evidence');
 mechanism.contentDocument.getElementById('occurrenceRecords').querySelector('summary').click();
 await wait(()=>mechanism.contentDocument.querySelector('#events .event'),'occurrence rows');
 mechanism.contentDocument.querySelector('#events .event').click();
 await wait(()=>document.querySelector('[data-mechanism="bounds"]').getAttribute('aria-checked')==='true','occurrence prompt');
 assert(selectedText().includes('17'),'occurrence sequence');
 assert($('llm-question').value==='왜 오래 걸렸나요?','all chart clicks leave the question untouched');
 const before=selectedText();
 window.postMessage({type:'log-analysis-select',scope:{topic:'simplex',call_id:'bad'},text:'untrusted'},'*');
 await new Promise(r=>setTimeout(r,50));assert(selectedText()===before,'unknown message source rejected');
 window.dispatchEvent(new CustomEvent('log-analysis-select',{detail:{scope:{topic:'simplex',call_id:'../../bad'},text:'bad'}}));
 assert(selectedText()===before,'invalid scope rejected');
 $('llm-question').value='x'.repeat(2000);hit('simplex-dashboard',map.regions[0],11);
 assert($('llm-question').value==='x'.repeat(2000),'long manual text never overwritten');
 assert(selectedText().includes('12번째')&&$('llm-call').value==='simplex-112','full question does not block selection');
 assert(requests===0&&$('llm-start').disabled,'click never sends API requests');
 out.textContent='PASS';
}catch(error){out.textContent='FAIL '+error.message;}document.body.appendChild(out);
})();</script>'''.replace('__MAP__', json.dumps(maps['simplex'])).replace('__POINTS__', json.dumps(points)).replace('__DPLL_MAP__', json.dumps(maps['dpll-theory'])).replace('__RELU_MAP__', json.dumps(maps['relu']))
            dashboard.write_text(dashboard.read_text(encoding='utf-8').replace('</body>',harness+'</body>'),encoding='utf-8')
            def no_client(_progress):
                self.fail('Selection must not start an LLM request')
            app = AnalysisApp(root, dashboard=dashboard.name, output_dir=root / 'analysis', client_factory=no_client)
            server = make_server(app)
            thread = threading.Thread(target=server.serve_forever, daemon=True)
            thread.start()
            try:
                process = subprocess.run([chromium_path(),'--headless','--no-sandbox','--disable-gpu','--no-first-run',
                    f'--user-data-dir={root / "profile"}','--virtual-time-budget=12000','--dump-dom',
                    f'http://127.0.0.1:{server.server_port}/s/{app.token}/'], capture_output=True,timeout=35)
                self.assertFalse(app.jobs)
            finally:
                server.shutdown();server.server_close();thread.join(timeout=5);app.close()
            dom = process.stdout.decode('utf-8',errors='replace')
            match = re.search(r'<pre id="selection-test">([^<]+)</pre>',dom)
            self.assertIsNotNone(match,dom[-1500:])
            self.assertEqual(match[1],'PASS')


if __name__ == '__main__':
    unittest.main()
