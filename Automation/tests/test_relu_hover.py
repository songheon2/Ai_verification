"""Verify raster neuron hit maps and responsive file:// hover labels."""
import json
from pathlib import Path
import re
import subprocess
import tempfile
import unittest

from Automation.tests.test_reactive_tree_browser import chromium_path
from visualization.RealtimeSplitVisualization import SplitLogEvent, draw_realtime_split_dashboard
from visualization.ReluNeuronHover import write_relu_hover_map
from visualization.UnifiedRealtimeDashboard import UnifiedRealtimeDashboard


class ReluHoverTests(unittest.TestCase):
    @unittest.skipUnless(chromium_path(), 'Chromium is not installed')
    def test_hover_names_survive_resizing_and_image_refresh(self):
        import matplotlib.pyplot as plt
        with tempfile.TemporaryDirectory(prefix='relu-hover-') as directory:
            root = Path(directory)
            image = root / 'relu.png'
            events = [SplitLogEvent('', 1000000000, 'a', 1, 3, 'z1_3', '+'),
                      SplitLogEvent('', 2000000000, 'b', 2, 7, 'z2_7', '+'),
                      SplitLogEvent('', 3000000000, 'c', 1, 3, 'z1_3', '+'),
                      SplitLogEvent('', 4000000000, 'a', 1, 3, 'z1_3', '-')]
            figure, _ = draw_realtime_split_dashboard(root/'events.jsonl', events=events,
                        round_starts=[{'round': 1, 'time_ns': 0}, {'round': 2, 'time_ns': 2500000000}])
            try:
                figure.savefig(image, dpi=100)
                sidecar = write_relu_hover_map(figure, image)
            finally:
                plt.close(figure)
            payload = json.loads(sidecar.read_text(encoding='utf-8')[len('window.postMessage('):-len(',"*");')])
            self.assertEqual([(p['layer'], p['neuron']) for p in payload['points']], [(1, 3), (2, 7)])
            self.assertEqual([p['roundSplits'] for p in payload['points']],
                             [[{'round': 1, 'count': 1}, {'round': 2, 'count': 1}],
                              [{'round': 1, 'count': 1}, {'round': 2, 'count': 0}]])
            for point in payload['points']:
                self.assertTrue(0 < point['x'] < 1 and 0 < point['y'] < 1)
            dashboard = root/'dashboard.html'
            UnifiedRealtimeDashboard({'relu': image}, dashboard).write()
            harness = r'''<script>
(async()=>{
 const wait=async test=>{for(let i=0;i<100;i++){if(test())return;await new Promise(r=>setTimeout(r,20));}throw Error('hover timed out');};
 const out=document.createElement('pre');out.id='browser-test-result';
 try {
  const points=__POINTS__;
  const check=async()=>{
   const image=document.getElementById('relu-dashboard');
   for(const point of points){
    await wait(()=>{const box=image.getBoundingClientRect();image.dispatchEvent(new MouseEvent('mousemove',{clientX:box.left+box.width*point.x,clientY:box.top+box.height*point.y}));const label=document.getElementById('relu-neuron-tooltip').textContent;return label.includes(`Layer ${point.layer} · Neuron ${point.neuron}`)&&point.roundSplits.every(item=>item.count>0?label.includes(`R${item.round} split: ${item.count}회`):!label.includes(`R${item.round} split: 0회`));});
    const tooltip=document.getElementById('relu-neuron-tooltip');
    if(tooltip.hidden)throw Error('custom tooltip must appear immediately without native hover delay');
    if(image.hasAttribute('title'))throw Error('native tooltip must not duplicate the custom tooltip');
    if(tooltip.textContent.includes('0부터'))throw Error('removed numbering note remains');
   }
   image.dispatchEvent(new MouseEvent('mouseleave'));if(image.title)throw Error('hover label remains after leaving');
  };
  await wait(()=>document.getElementById('relu-dashboard').naturalWidth>0 && !panels[0].pending);await check();
  const hoverImage=document.getElementById('relu-dashboard'),readBounds=hoverImage.getBoundingClientRect.bind(hoverImage);
  const hoverBox=readBounds(),hoverPoint=points[0];let boundsReads=0;
  hoverImage.getBoundingClientRect=()=>{boundsReads++;return readBounds();};
  for(let i=0;i<100;i++)hoverImage.dispatchEvent(new MouseEvent('mousemove',{clientX:hoverBox.left+hoverBox.width*hoverPoint.x,clientY:hoverBox.top+hoverBox.height*hoverPoint.y}));
  if(boundsReads>3)throw Error('rapid mouse events must not measure layout one hundred times');
  hoverImage.dispatchEvent(new MouseEvent('mouseleave'));
  await new Promise(resolve=>setTimeout(resolve,30));
  if(!document.getElementById('relu-neuron-tooltip').hidden)throw Error('queued hover must not reappear after mouse leave');
  hoverImage.getBoundingClientRect=readBounds;
  document.querySelector('.relu-section').style.width='600px';await check();
  const previous=document.getElementById('relu-dashboard'),version=previous.dataset.renderVersion,last=points[points.length-1],box=previous.getBoundingClientRect();
  previous.dispatchEvent(new MouseEvent('mousemove',{clientX:box.left+box.width*last.x,clientY:box.top+box.height*last.y}));
  refreshPanel(panels[0]);
  await wait(()=>previous.dataset.renderVersion!==version);
  if(document.getElementById('relu-dashboard')!==previous)throw Error('refresh must keep the hover and click target stable');
  await wait(()=>!document.getElementById('relu-neuron-tooltip').hidden);
  await check();
  out.textContent='PASS: neuron hover follows responsive images';
 }catch(error){out.textContent='FAIL: '+error.message;}
 document.body.append(out);
})();</script>'''.replace('__POINTS__', json.dumps(payload['points']))
            dashboard.write_text(dashboard.read_text(encoding='utf-8').replace('</body>',harness+'</body>'),encoding='utf-8')
            result = subprocess.run([chromium_path(),'--headless','--no-sandbox','--disable-gpu',
                                     '--no-first-run',f'--user-data-dir={root / "profile"}',
                                     '--virtual-time-budget=6000','--dump-dom',dashboard.as_uri()],
                                    capture_output=True,timeout=30)
            match = re.search(r'<pre id="browser-test-result">([^<]+)</pre>',result.stdout.decode('utf-8',errors='replace'))
            self.assertIsNotNone(match)
            self.assertTrue(match.group(1).startswith('PASS:'),match.group(1))


if __name__=='__main__':
    unittest.main()
