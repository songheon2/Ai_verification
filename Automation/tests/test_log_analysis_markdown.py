"""Safe local Markdown report rendering and layout checks, without an LLM call."""
import json
from pathlib import Path
import re
import subprocess
import tempfile
import unittest

from Automation.tests.test_reactive_tree_browser import chromium_path
from visualization.LogAnalysisPanel import install_panel, panel_markup


SAMPLE = '''# Simplex 분석 보고서

## 1. 관찰된 패턴

**첫 호출에 작업이 집중**되어 있습니다. `simplex-1`의 반복·피벗 기록을 우선 확인해야 합니다.
근거: [run:solver_progress.jsonl:3]

| 항목 | 관찰 | 해석 |
| :--- | ---: | :--- |
| Simplex 호출 | **12회** | 선택한 R1 범위 |
| 최대 피벗 | 320 | `simplex-1` |
| 종료 상태 | UNSAT | 선형 제약의 불가능 |

## 2. 원인 가설

- **경계 위반 반복**: 동일한 변수가 반복해서 나타납니다.
  - Layer 2 · Neuron 49의 경계 변경을 비교합니다.
  - 수정 전후의 기록이 있어야 인과를 판단할 수 있습니다.
- *수치 오차 영향*은 아직 확인되지 않았습니다.

> 관찰된 상관관계만으로 병목의 원인을 확정할 수 없습니다.

### 다음 확인 순서

1. 해당 호출의 진입·이탈 변수를 확인합니다.
2. 직전 분기와 경계 변경 기록을 비교합니다.

```text
R1 | simplex-1 | pivots=320
z2_49 < lower_bound
```

---

분석 범위는 **선택한 항목에 한정**됩니다. [공식 문서](https://example.com/reference)
'''


def render_fixture(directory, width=1280, screenshot=None):
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    malicious = '<img src=x onerror="window.pwned=1"><script>window.pwned=1</script>\n\n[x](javascript:alert(1)) [x](data:text/html,evil)'
    harness = r'''<script>
const result=document.getElementById('llm-result');
const output=document.createElement('pre');output.id='markdown-test-result';
try{
 const assert=(ok,message)=>{if(!ok)throw Error(message);};
 const primary=document.getElementById('llm-start'),wasDisabled=primary.disabled;
 const question=document.getElementById('llm-question');
 assert(question.getBoundingClientRect().height>=180,'question input is taller');
 assert(getComputedStyle(question).resize==='vertical','question remains manually resizable');
 assert(primary.querySelector('svg[aria-hidden="true"]'),'decorative action icon');
 assert(parseFloat(getComputedStyle(primary).borderRadius)>=20,'pill-shaped buttons');
 primary.disabled=false;
 assert(getComputedStyle(primary).color==='rgb(255, 255, 255)','blue primary action has white label');
 primary.disabled=wasDisabled;
 assert(document.getElementById('llm-quota-link').rel==='noopener noreferrer','safe external quota link');
 document.getElementById('llm-topic').value='simplex';
 document.getElementById('llm-topic').dispatchEvent(new Event('change'));
 assert(getComputedStyle(document.getElementById('llm-layer-label')).display==='none','irrelevant controls stay hidden');
 assert(getComputedStyle(document.getElementById('llm-call-label')).display!=='none','selected call filter remains visible');
 window.renderLogAnalysisMarkdown(result,__MALICIOUS__);
 assert(!result.querySelector('img,script,iframe,svg'),'HTML must remain text');
 assert(!result.querySelector('a'),'unsafe URL schemes must not become links');
 assert(!window.pwned,'untrusted HTML cannot execute');
 const legacy='simplex_end duration_seconds=1.25 pivots=12 simplex-23 z2_49_vnnlib_199 h2_49_vnnlib_200 THEORY_CONFLICT BOOLEAN_UNSAT';
 const readable=window.formatLogAnalysisText(legacy);
 assert(readable.includes('소요 시간(초)=1.25') && readable.includes('피벗 횟수=12'),'numbers and metric meanings preserved');
 assert(readable.includes('Simplex 호출 ID 23'),'call ID is not invented graph order');
 assert(readable.includes('L2:N49 (z · ReLU 적용 전)') && readable.includes('L2:N49 (h · ReLU 출력)'),'neuron role and coordinates preserved');
 assert(!readable.includes('vnnlib_') && !readable.includes('simplex_end'),'raw names removed from display');
 assert(readable.includes('CONFLICT(이론 제약 충돌)') && readable.includes('DPLL 후보 소진(UNSAT)'),'theory and overall Boolean status remain distinct');
 assert(window.formatLogAnalysisText('[문서](https://example.com/duration_seconds)')==='[문서](https://example.com/duration_seconds)','URLs remain unchanged');
 window.renderLogAnalysisMarkdown(result,__SAMPLE__);
 assert(result.querySelector('h1')?.textContent==='Simplex 분석 보고서','Markdown title');
 assert(result.querySelectorAll('h2').length===2,'section headings');
 assert(result.querySelector('strong')?.textContent==='첫 호출에 작업이 집중','bold text');
 assert(result.querySelectorAll('table thead th').length===3,'table header');
 assert(result.querySelectorAll('table tbody tr').length===3,'table rows');
 assert(result.querySelector('ul ul li'),'nested bullet list');
 assert(result.querySelectorAll('ol li').length===2,'numbered list');
 assert(result.querySelector('blockquote'),'callout');
 assert(result.querySelector('pre code')?.textContent.includes('L2:N49 (z · ReLU 적용 전) < 하한'),'readable variable and bound names');
 assert(result.querySelector('.llm-citation')?.textContent==='솔버 진행 · 기록 3','readable citation chip');
 assert(!result.textContent.includes('solver_progress.jsonl'),'raw source name stays out of report');
 assert(result.querySelector('a')?.rel==='noopener noreferrer','safe external link');
 assert(document.documentElement.scrollWidth<=innerWidth,'report fits viewport');
 const style=getComputedStyle(result);assert(style.whiteSpace==='normal','no preformatted paragraph layout');
 document.getElementById('llm-result-tag').textContent='분석 완료';
 document.getElementById('llm-result-tag').dataset.status='complete';
 document.getElementById('llm-connection').textContent='gemini-3.5-flash-lite';
 document.getElementById('llm-progress').textContent='complete · API 4회';
 output.textContent='PASS';
}catch(error){output.textContent='FAIL: '+error.message;}
document.body.appendChild(output);
</script>'''.replace('__SAMPLE__', json.dumps(SAMPLE, ensure_ascii=False).replace('<', '\\u003c')).replace(
    '__MALICIOUS__', json.dumps(malicious).replace('<', '\\u003c'))
    html = '<!doctype html><html lang="ko"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><body style="margin:0;padding:24px;background:#0f172a;font-family:system-ui"><main style="max-width:1080px;margin:auto"><img id="simplex-dashboard" hidden>' + panel_markup() + '</main>' + harness + '</body></html>'
    path = directory / 'report.html'
    path.write_text(html, encoding='utf-8')
    command = [chromium_path(), '--headless', '--no-sandbox', '--disable-gpu', '--no-first-run',
               f'--user-data-dir={directory / "profile"}', f'--window-size={width},1900',
               '--virtual-time-budget=1500', '--dump-dom']
    if screenshot is not None:
        command.append('--screenshot=' + str(Path(screenshot).resolve()))
    command.append(path.as_uri())
    process = subprocess.run(command, capture_output=True, timeout=30)
    return process.stdout.decode('utf-8', errors='replace')


class MarkdownReportTests(unittest.TestCase):
    def test_old_and_current_dashboard_panels_are_refreshed_without_duplication(self):
        for old in ('<main><section id="llm-analysis-panel">old</section><script>OLD_UI</script></main>',
                    '<main><!-- LLM_ANALYSIS_PANEL_START -->old<!-- LLM_ANALYSIS_PANEL_END --></main>', '<main></main>'):
            document=install_panel(old)
            self.assertEqual(document.count('id="llm-analysis-panel"'),1)
            self.assertIn('Markdown 분석 보고서',document)
            self.assertNotIn('OLD_UI',document)
            self.assertNotIn('id="llm-evidence"',document)
            self.assertNotIn('조회 근거',document)
            self.assertNotIn('LLM 분석은 검증 판정이나 증명이 아닙니다.',document)
            self.assertEqual(install_panel(document),document)

    @unittest.skipUnless(chromium_path(), 'Chromium is not installed')
    def test_markdown_is_formatted_safe_and_responsive(self):
        for width in (1280, 480):
            with self.subTest(width=width), tempfile.TemporaryDirectory() as directory:
                dom = render_fixture(Path(directory).resolve(), width)
                match = re.search(r'<pre id="markdown-test-result">([^<]+)</pre>', dom)
                self.assertIsNotNone(match, dom[-1000:])
                self.assertEqual(match[1], 'PASS')


if __name__ == '__main__':
    unittest.main()
