"""Offline browser and graph font checks; no solver run or external requests."""
from functools import partial
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
import re
import subprocess
import tempfile
import threading
import unittest

from Automation.tests.test_reactive_tree_browser import chromium_path
from visualization.Fonts import (
    FONT_ASSETS, FONT_DIRECTORY, FONT_FAMILY, apply_web_font, configure_matplotlib_fonts,
)


class VisualizationFontTests(unittest.TestCase):
    def test_graph_fonts_resolve_to_bundled_regular_and_bold_with_korean_glyphs(self):
        from matplotlib.font_manager import FontProperties, findfont
        from matplotlib.ft2font import FT2Font
        configure_matplotlib_fonts()
        for weight, name in ((400, FONT_ASSETS[0]), (700, FONT_ASSETS[1])):
            font = Path(findfont(FontProperties(family=FONT_FAMILY, weight=weight)))
            self.assertEqual(font.resolve(), (FONT_DIRECTORY / name).resolve())
            self.assertTrue(set(map(ord, "한글경계유도ReLUSimplex")) <= FT2Font(str(font)).get_charmap().keys())
        self.assertEqual(Path(findfont(FontProperties(family="monospace"))).name, FONT_ASSETS[0])

    def test_reports_share_one_local_font_copy_without_rewriting_it(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            page = root / "report.html"
            document = apply_web_font("<html><head></head><body>한글</body></html>", page)
            assets = root / "_fonts"
            before = {p.name: p.stat().st_mtime_ns for p in assets.iterdir()}
            self.assertEqual(set(before), set(FONT_ASSETS))
            self.assertEqual(apply_web_font(document, page), document)
            self.assertEqual(before, {p.name: p.stat().st_mtime_ns for p in assets.iterdir()})
            self.assertEqual(document.count("data-visualization-font"), 1)
            self.assertNotIn("https://", document)
            self.assertLess(len(document), 2000)

    @unittest.skipUnless(chromium_path(), "Chromium browser is not installed")
    def test_bundled_web_font_loads_from_file_and_local_http(self):
        harness = '''<script>
(async()=>{const output=document.createElement('pre');output.id='font-test-result';
try{
 const faces=await document.fonts.load('400 16px "Noto Sans KR"','한글');
 const bold=await document.fonts.load('700 16px "Noto Sans KR"','한글');
 if(faces.length!==1||bold.length!==1)throw Error('bundled font did not load');
 for(const element of document.querySelectorAll('body,button,input,select,textarea,svg,text,code,pre')){
   if(!getComputedStyle(element).fontFamily.includes('Noto Sans KR'))throw Error('mixed font: '+element.tagName);
 }
 output.textContent='PASS: local regular/bold Korean fonts loaded';
}catch(error){output.textContent='FAIL: '+error.message;}document.body.append(output);})();
</script>'''
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            page = root / "font.html"
            document = apply_web_font('<!doctype html><html><head></head><body>경계 유도 ReLU'
                '<button>분석</button><input><select><option>선택</option></select><textarea>한글</textarea>'
                '<svg><text x="0" y="20">뉴런</text></svg><code>값</code><pre>기록</pre>'
                + harness + '</body></html>', page)
            page.write_text(document, encoding="utf-8")
            class Handler(SimpleHTTPRequestHandler):
                def log_message(self, *_args):
                    pass
            server = ThreadingHTTPServer(("127.0.0.1", 0), partial(Handler, directory=directory))
            worker = threading.Thread(target=server.serve_forever, daemon=True)
            worker.start()
            try:
                for scheme, url in (("file", page.as_uri()),
                                    ("http", f"http://127.0.0.1:{server.server_port}/font.html")):
                    with self.subTest(scheme=scheme):
                        result = subprocess.run([chromium_path(), "--headless", "--no-sandbox", "--disable-gpu",
                            "--no-first-run", "--no-default-browser-check", f"--user-data-dir={root / scheme}",
                            "--virtual-time-budget=10000", "--dump-dom", url], capture_output=True, timeout=40)
                        match = re.search(r'<pre id="font-test-result">([^<]+)</pre>', result.stdout.decode("utf-8", errors="replace"))
                        self.assertIsNotNone(match, result.stderr.decode("utf-8", errors="replace")[-2000:])
                        self.assertTrue(match.group(1).startswith("PASS:"), match.group(1))
            finally:
                server.shutdown()
                server.server_close()
                worker.join(timeout=5)


if __name__ == "__main__":
    unittest.main()
