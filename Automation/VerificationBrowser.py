"""Track only the tabs opened by one standalone verification launcher."""
from http.client import HTTPConnection, HTTPException
import json
import os
from pathlib import Path
import shutil
import subprocess
from tempfile import mkdtemp
from threading import Lock
from time import monotonic, sleep
from urllib.parse import quote, urlsplit


def chromium_executables():
    candidates = []
    for variable in ("ProgramFiles", "ProgramFiles(x86)", "LOCALAPPDATA"):
        root = os.environ.get(variable)
        if root:
            for relative in ("Google/Chrome/Application/chrome.exe", "Microsoft/Edge/Application/msedge.exe"):
                path = Path(root) / relative
                if path.is_file():
                    candidates.append(str(path))
    for name in ("chrome", "msedge", "google-chrome", "chromium", "microsoft-edge"):
        if executable := shutil.which(name):
            candidates.append(executable)
    seen = set()
    for executable in candidates:
        key = os.path.normcase(os.path.realpath(executable))
        if key not in seen:
            seen.add(key)
            yield executable


class VerificationBrowser:
    def __init__(self, executable):
        self.executable = executable
        self.profile = None
        self.process = None
        self.port = None
        self.lock = Lock()
        self.tabs = []
        self.closed = False

    def _request(self, path, method="GET"):
        connection = HTTPConnection("127.0.0.1", self.port, timeout=2)
        try:
            connection.request(method, path)
            response = connection.getresponse()
            if response.status != 200:
                raise OSError(f"Browser request failed ({response.status})")
            return response.read(1024 * 1024)
        finally:
            connection.close()

    def _pages(self):
        return json.loads(self._request("/json/list"))

    def start(self, url, *, extra_args=()):
        try:
            self.profile = Path(mkdtemp(prefix="verification-browser-")).resolve()
            command = [self.executable, f"--user-data-dir={self.profile}", "--remote-debugging-port=0",
                       "--no-first-run", "--no-default-browser-check", "--disable-background-mode",
                       "--start-maximized", *extra_args, "--new-window", "about:blank"]
            environment = {key: value for key, value in os.environ.items() if key != "GEMINI_API_KEY"}
            self.process = subprocess.Popen(command, stdin=subprocess.DEVNULL, stdout=subprocess.DEVNULL,
                                            stderr=subprocess.DEVNULL, env=environment)
            deadline = monotonic() + 12
            while self.process.poll() is None and monotonic() < deadline:
                try:
                    lines = (self.profile / "DevToolsActivePort").read_text(encoding="utf-8").splitlines()
                    self.port = int(lines[0])
                    blank_tabs = [page["id"] for page in self._pages()
                                  if page.get("type") == "page" and page.get("url") == "about:blank"]
                    if self.open(url):
                        for tab in blank_tabs:
                            try:
                                self._request("/json/close/" + quote(tab, safe=""))
                            except (OSError, HTTPException):
                                self.tabs.insert(0, tab)
                        return True
                except (OSError, ValueError, IndexError, KeyError, HTTPException):
                    pass
                sleep(0.05)
        except OSError:
            pass
        if self.process is not None and self.process.poll() is None:
            self.process.terminate()
            try:
                self.process.wait(timeout=3)
            except subprocess.TimeoutExpired:
                pass
        self._clean_profile()
        return False

    def open(self, url):
        target = urlsplit(url)
        if not (target.scheme == "file" and not target.netloc
                or target.scheme == "http" and target.hostname == "127.0.0.1"):
            return False
        with self.lock:
            if self.closed or self.port is None or self.process.poll() is not None:
                return False
            try:
                page = json.loads(self._request("/json/new?" + quote(url, safe=""), method="PUT"))
                self.tabs.append(page["id"])
            except (OSError, ValueError, KeyError, HTTPException):
                return False
            return True

    def close(self):
        with self.lock:
            self.closed = True
            for tab in reversed(self.tabs):
                try:
                    self._request("/json/close/" + quote(tab, safe=""))
                except (OSError, HTTPException):
                    # Tabs may already have been closed by the user.
                    continue
            self.tabs.clear()
        if self.process is not None:
            try:
                self.process.wait(timeout=1)
            except subprocess.TimeoutExpired:
                # A user-opened tab can keep this browser session alive.
                pass
        self._clean_profile()

    def _clean_profile(self):
        if (self.profile is not None and self.profile.resolve() == self.profile
                and (self.process is None or self.process.poll() is not None)):
            shutil.rmtree(self.profile, ignore_errors=True)
