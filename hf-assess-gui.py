#!/usr/bin/env python3
"""One-box operator: paste or drop anything, the rest is probed and filled."""

from __future__ import annotations

import importlib.util
import json
import os
import queue
import shutil
import subprocess
import sys
import threading
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

try:
    import tkinter as tk
    from tkinter import filedialog, messagebox, ttk
    from tkinter.scrolledtext import ScrolledText
except ImportError as exc:
    raise SystemExit("tkinter is required. On macOS: brew install python-tk") from exc

SCRIPT_DIR = Path(__file__).resolve().parent
ANALYZER = SCRIPT_DIR / "hf_assess.py"
DEFAULT_VENV = SCRIPT_DIR / ".hf-assess-venv"
RESULTS_DIR = SCRIPT_DIR / "hf-assess-results"
HF_HUB = Path.home() / ".cache" / "huggingface" / "hub"

PREFERRED_MODELS = [
    "HuggingFaceTB/SmolLM2-360M-Instruct",
    "Qwen/Qwen2.5-0.5B-Instruct",
    "microsoft/DialoGPT-medium",
]


def load_analyzer():
    spec = importlib.util.spec_from_file_location("hf_assess", ANALYZER)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load {ANALYZER}")
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)
    return mod


def venv_python() -> Path:
    candidate = DEFAULT_VENV / "bin" / "python"
    return candidate if candidate.is_file() else Path(sys.executable)


def model_is_cached(repo: str) -> bool:
    path = HF_HUB / f"models--{repo.replace('/', '--')}"
    return path.is_dir() and any(path.rglob("config.json"))


def pick_ready_model(preferred: Optional[str] = None) -> Optional[str]:
    if preferred and model_is_cached(preferred):
        return preferred
    for repo in PREFERRED_MODELS:
        if model_is_cached(repo):
            return repo
    return preferred if preferred else None


def _redact_cmd(cmd: List[str]) -> str:
    redacted: List[str] = []
    hide_next = False
    for part in cmd:
        if hide_next:
            redacted.append("<redacted>")
            hide_next = False
            continue
        if part in {"--hf-token", "--operator"}:
            redacted.append(part)
            hide_next = True
            continue
        redacted.append(part)
    return " ".join(redacted)


def default_operator() -> Optional[str]:
    """Never infer the local account, git email, or machine name."""
    value = (os.getenv("HF_ASSESS_OPERATOR") or "").strip()
    return value or None


class EasyOperator(tk.Tk):
    def __init__(self, initial: str = "", preselect_repo: Optional[str] = None) -> None:
        super().__init__()
        self.analyzer = load_analyzer()
        self.title("Assess")
        self.geometry("920x720")
        self.minsize(760, 560)
        self.msg_q: queue.Queue[str] = queue.Queue()
        self.busy = False
        self.last_output: Optional[Path] = None
        self.probe = None
        self.chosen_model = pick_ready_model(preselect_repo)
        self.operator = default_operator()
        self._build()
        if initial.strip():
            self.inbox.insert("1.0", initial.strip())
        self.after(150, self._reprobe)
        self.after(200, self._drain_queue)

    def _build(self) -> None:
        self.columnconfigure(0, weight=1)
        self.rowconfigure(2, weight=1)
        self.rowconfigure(4, weight=3)

        head = ttk.Frame(self, padding=(16, 14, 16, 6))
        head.grid(row=0, column=0, sticky="ew")
        ttk.Label(head, text="Paste anything. Then run.", font=("Helvetica", 20, "bold")).pack(anchor="w")
        ttk.Label(
            head,
            text="IP, hostname, URL, nmap output, XML, JSON, CVE list, a scan file path, or a Hugging Face model id.",
            wraplength=880,
        ).pack(anchor="w")

        box = ttk.Frame(self, padding=(16, 4, 16, 4))
        box.grid(row=1, column=0, sticky="nsew")
        box.columnconfigure(0, weight=1)
        self.inbox = ScrolledText(box, height=8, wrap="word", font=("Menlo", 13), undo=True)
        self.inbox.grid(row=0, column=0, sticky="nsew")
        self.inbox.bind("<KeyRelease>", lambda _e: self._schedule_probe())
        self.inbox.bind("<<Paste>>", lambda _e: self.after(30, self._reprobe))
        self.inbox.focus_set()

        actions = ttk.Frame(box)
        actions.grid(row=1, column=0, sticky="ew", pady=(8, 0))
        ttk.Button(actions, text="Open file…", command=self._open_file).pack(side="left")
        ttk.Button(actions, text="Paste clipboard", command=self._paste_clipboard).pack(side="left", padx=6)
        ttk.Button(actions, text="Clear", command=self._clear).pack(side="left")
        self.run_btn = ttk.Button(actions, text="Run", command=self.run_analysis)
        self.run_btn.pack(side="right")
        self.auth_var = tk.BooleanVar(value=False)
        self.auth_chk = ttk.Checkbutton(
            actions,
            text="I am authorized for this public target",
            variable=self.auth_var,
        )
        self.nmap_var = tk.BooleanVar(value=False)
        self.nmap_chk = ttk.Checkbutton(actions, text="Also run nmap", variable=self.nmap_var)

        detect = ttk.LabelFrame(self, text="What I will do", padding=10)
        detect.grid(row=2, column=0, sticky="nsew", padx=16, pady=(4, 8))
        detect.columnconfigure(0, weight=1)
        self.detect = ScrolledText(detect, height=7, wrap="word", font=("Helvetica", 13), background="#f4f4f4")
        self.detect.grid(row=0, column=0, sticky="nsew")
        self.detect.configure(state="disabled")

        self.status = tk.StringVar(value=self._model_status_line())
        ttk.Label(self, textvariable=self.status, padding=(16, 0)).grid(row=3, column=0, sticky="w")

        notes = ttk.Notebook(self)
        notes.grid(row=4, column=0, sticky="nsew", padx=16, pady=(0, 12))
        self.summary = ScrolledText(notes, wrap="word", font=("Helvetica", 13))
        self.findings = ScrolledText(notes, wrap="word", font=("Menlo", 12))
        self.model_out = ScrolledText(notes, wrap="word", font=("Menlo", 12))
        self.log = ScrolledText(notes, wrap="word", font=("Menlo", 11))
        notes.add(self.summary, text="Result")
        notes.add(self.findings, text="Findings")
        notes.add(self.model_out, text="Model")
        notes.add(self.log, text="Log")
        for widget in (self.summary, self.findings, self.model_out, self.log):
            widget.configure(state="disabled")

        self._probe_job = None

    def _model_status_line(self) -> str:
        nmap = "nmap found" if shutil.which("nmap") else "nmap not installed"
        if self.chosen_model:
            return f"Model ready: {self.chosen_model}   •   {nmap}"
        return f"No local model on disk yet — Run will download SmolLM2 360M Instruct   •   {nmap}"

    def _schedule_probe(self) -> None:
        if self._probe_job is not None:
            self.after_cancel(self._probe_job)
        self._probe_job = self.after(180, self._reprobe)

    def _inbox_text(self) -> str:
        return self.inbox.get("1.0", "end").strip()

    def _reprobe(self) -> None:
        self._probe_job = None
        text = self._inbox_text()
        self.probe = self.analyzer.probe_input(text, persist_dir=RESULTS_DIR / "inbox")
        if self.probe.model_repo:
            self.chosen_model = pick_ready_model(self.probe.model_repo) or self.probe.model_repo
        if self.probe.operator:
            self.operator = self.probe.operator
        self._render_probe()

    def _render_probe(self) -> None:
        probe = self.probe
        lines: List[str] = []
        if not probe or not self._inbox_text():
            lines.append("Waiting. Drop a file, paste a scan, or type a host.")
        else:
            lines.append(probe.summary)
            if probe.target:
                info = self.analyzer.classify_target(probe.target)
                extra = f"  (+{len(probe.extra_targets)} more)" if probe.extra_targets else ""
                lines.append(f"Host: {probe.target} ({info.kind}){extra}")
            if probe.scan_file:
                count = len((probe.parsed_scan or {}).get("findings") or [])
                lines.append(f"Evidence: {Path(probe.scan_file).name}  ({count} finding(s) extracted)")
            if probe.cves:
                lines.append("CVEs: " + ", ".join(probe.cves[:12]))
            if probe.tools:
                lines.append("Tools mentioned: " + ", ".join(probe.tools))
            if probe.model_repo:
                on = "on disk" if model_is_cached(probe.model_repo) else "will download if you run"
                lines.append(f"Model override: {probe.model_repo} ({on})")
            if probe.warnings:
                lines.extend("Note: " + w for w in probe.warnings[:3])
            if not probe.ready:
                lines.append("I still need a host or a scan before I can run.")
        if probe and probe.needs_authorization:
            self.auth_chk.pack(side="right", padx=10)
        else:
            self.auth_chk.pack_forget()
            self.auth_var.set(False)
        show_nmap = bool(
            probe
            and probe.target
            and not probe.scan_file
            and shutil.which("nmap")
        )
        if show_nmap:
            self.nmap_chk.pack(side="right", padx=10)
        else:
            self.nmap_chk.pack_forget()
            self.nmap_var.set(False)
        self._set_text(self.detect, "\n".join(lines))
        self.status.set(self._model_status_line())

    def _open_file(self) -> None:
        path = filedialog.askopenfilename(
            title="Open scan or evidence file",
            filetypes=[
                ("Anything useful", "*.xml *.json *.txt *.gnmap *.nmap *.csv"),
                ("All files", "*.*"),
            ],
        )
        if path:
            current = self._inbox_text()
            addition = path if not current else current + "\n" + path
            self.inbox.delete("1.0", "end")
            self.inbox.insert("1.0", addition)
            self._reprobe()

    def _paste_clipboard(self) -> None:
        try:
            clip = self.clipboard_get()
        except tk.TclError:
            return
        if clip:
            self.inbox.insert("end", ("\n" if self._inbox_text() else "") + clip)
            self._reprobe()

    def _clear(self) -> None:
        self.inbox.delete("1.0", "end")
        self.probe = None
        self._render_probe()

    def run_analysis(self) -> None:
        if self.busy:
            return
        self._reprobe()
        probe = self.probe
        if not probe or not probe.ready:
            messagebox.showinfo("Need input", "Paste a host, URL, or scan first.")
            return
        if probe.needs_authorization and not self.auth_var.get():
            messagebox.showinfo(
                "Public target",
                f"{probe.target} looks public. Check the authorization box if you are allowed to analyze it.",
            )
            return

        model = self.chosen_model or "HuggingFaceTB/SmolLM2-360M-Instruct"
        if not model_is_cached(model) and not (probe.remote_url):
            if not messagebox.askyesno(
                "Download model",
                f"No local model is ready.\nDownload {model} now and then run?",
            ):
                return
            self._start_download(model, then_run=True)
            return
        self._launch(probe, model)

    def _launch(self, probe: Any, model: str) -> None:
        RESULTS_DIR.mkdir(parents=True, exist_ok=True)
        stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
        host = probe.target or "assessment"
        safe = "".join(ch if ch.isalnum() or ch in ".-" else "_" for ch in host)[:40]
        output = RESULTS_DIR / f"{safe}-{stamp}.json"
        cmd = [
            str(venv_python()),
            str(ANALYZER),
            "--json",
            "--output",
            str(output),
            "--engagement-id",
            f"ENG-{stamp}",
            "--model",
            model,
        ]
        if self.operator:
            cmd.extend(["--operator", self.operator])
        if probe.target:
            cmd.extend(["--target", probe.target])
        tools = probe.tools or (["nmap"] if self.nmap_var.get() else ["imported-scan"])
        cmd.append("--tools")
        cmd.extend(tools)
        if probe.scan_file:
            cmd.extend(["--scan-file", probe.scan_file])
        if probe.hf_token:
            cmd.extend(["--hf-token", probe.hf_token])
        if probe.remote_url:
            cmd.extend(["--remote-model", probe.remote_url])
        if probe.needs_authorization or self.auth_var.get():
            cmd.append("--authorized")
        if self.nmap_var.get():
            cmd.append("--run-tools")
        if probe.cves:
            cmd.append("--enrich-cves")
        if probe.notes:
            cmd.extend(["--notes", "; ".join(probe.notes)])
        if probe.extra_targets:
            cmd.extend(["--notes", "Also mentioned: " + ", ".join(probe.extra_targets[:20])])

        self.busy = True
        self.run_btn.state(["disabled"])
        self.status.set("Working…")
        self._log(_redact_cmd(cmd))
        threading.Thread(target=self._run_worker, args=(cmd, output), daemon=True).start()

    def _start_download(self, repo: str, then_run: bool = False) -> None:
        self.busy = True
        self.run_btn.state(["disabled"])
        self.status.set(f"Downloading {repo}…")
        self._log(f"Downloading {repo}")
        threading.Thread(target=self._download_worker, args=(repo, then_run), daemon=True).start()

    def _download_worker(self, repo: str, then_run: bool) -> None:
        env = os.environ.copy()
        token = (self.probe.hf_token if self.probe else None) or os.getenv("HF_TOKEN") or ""
        if token:
            env["HF_TOKEN"] = token
        code = (
            "import os,sys\n"
            "from huggingface_hub import snapshot_download\n"
            "print(snapshot_download(repo_id=sys.argv[1], token=sys.argv[2] or None))\n"
        )
        try:
            proc = subprocess.Popen(
                [str(venv_python()), "-u", "-c", code, repo, token],
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                env=env,
            )
            assert proc.stdout is not None
            for line in proc.stdout:
                self.msg_q.put(line.rstrip())
            rc = proc.wait()
            self.msg_q.put(f"DOWNLOAD {'OK' if rc == 0 else 'FAIL'} {repo} {int(then_run)}")
        except Exception as exc:
            self.msg_q.put(f"DOWNLOAD FAIL {repo} 0")
            self.msg_q.put(str(exc))

    def _run_worker(self, cmd: List[str], output: Path) -> None:
        env = os.environ.copy()
        try:
            proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, env=env)
            assert proc.stdout is not None
            chunks: List[str] = []
            for line in proc.stdout:
                chunks.append(line)
                self.msg_q.put(line.rstrip())
            rc = proc.wait()
            self.msg_q.put(f"RUN {rc} {output}")
        except Exception as exc:
            self.msg_q.put(f"RUN FAIL {exc}")

    def _drain_queue(self) -> None:
        try:
            while True:
                item = self.msg_q.get_nowait()
                if item.startswith("DOWNLOAD OK "):
                    _, _, repo, then_run = item.split(" ", 3)
                    self.chosen_model = repo
                    self.busy = False
                    self.run_btn.state(["!disabled"])
                    self.status.set(f"Downloaded {repo}")
                    if then_run == "1":
                        self.run_analysis()
                elif item.startswith("DOWNLOAD FAIL"):
                    self.busy = False
                    self.run_btn.state(["!disabled"])
                    self.status.set("Download failed")
                    messagebox.showerror("Download failed", item)
                elif item.startswith("RUN FAIL"):
                    self.busy = False
                    self.run_btn.state(["!disabled"])
                    self.status.set("Run failed")
                    messagebox.showerror("Run failed", item)
                elif item.startswith("RUN "):
                    parts = item.split(" ", 2)
                    rc = int(parts[1])
                    output = Path(parts[2])
                    self.busy = False
                    self.run_btn.state(["!disabled"])
                    if rc != 0:
                        self.status.set(f"Analyzer failed (exit {rc})")
                        messagebox.showerror("Analyzer failed", "See the Log tab.")
                    else:
                        self._show_output(output)
                else:
                    self._log(item)
        except queue.Empty:
            pass
        self.after(200, self._drain_queue)

    def _show_output(self, output: Path) -> None:
        self.last_output = output
        data: Dict[str, Any] = {}
        if output.is_file():
            try:
                data = json.loads(output.read_text(encoding="utf-8"))
            except json.JSONDecodeError:
                data = {}
        raw = data.get("raw_results") or {}
        analysis = data.get("ai_analysis") or {}
        risk = data.get("risk_score") or {}
        findings = raw.get("findings") or []
        recs = analysis.get("recommendations") or []
        lines = [
            f"{risk.get('label', 'Done')}  ({risk.get('score', '?')}/100)",
            f"Target  {raw.get('target')}",
            f"Saved   {output}",
            "",
            str(analysis.get("summary") or "The model did not return a summary field. Open the Model tab."),
            "",
        ]
        if recs:
            lines.append("Recommendations")
            lines.extend(f"• {rec}" for rec in recs)
        self._set_text(self.summary, "\n".join(lines))
        if findings:
            blocks = []
            for finding in findings:
                blocks.append(
                    f"[{str(finding.get('severity') or '?').upper()}] "
                    f"{finding.get('title') or finding.get('cve') or finding.get('id')}\n"
                    f"    {finding.get('cve') or ''}  port={finding.get('port')}"
                )
            self._set_text(self.findings, "\n\n".join(blocks))
        else:
            self._set_text(self.findings, "No extracted findings. The model still analyzed whatever you pasted.")
        self._set_text(self.model_out, str(analysis.get("model_raw") or analysis.get("summary") or ""))
        self.status.set(f"Done — {risk.get('label', 'report')}  {output.name}")
        md = output.with_suffix(".md")
        if md.is_file():
            self._log(f"Markdown: {md}")

    def _log(self, line: str) -> None:
        self.log.configure(state="normal")
        self.log.insert("end", f"{time.strftime('%H:%M:%S')}  {line}\n")
        self.log.see("end")
        self.log.configure(state="disabled")

    @staticmethod
    def _set_text(widget: ScrolledText, text: str) -> None:
        widget.configure(state="normal")
        widget.delete("1.0", "end")
        widget.insert("1.0", text)
        widget.configure(state="disabled")


def main(argv: Optional[List[str]] = None) -> int:
    args = list(argv if argv is not None else sys.argv[1:])
    repo = None
    if args and args[0] == "--repo" and len(args) > 1:
        repo = args[1]
        args = args[2:]
    if args and args[0] in {"-h", "--help"}:
        print("Usage: hf-assess-gui.py [--repo org/model] [file-or-target ...]")
        return 0
    initial_parts: List[str] = []
    for item in args:
        path = Path(item).expanduser()
        initial_parts.append(str(path.resolve()) if path.exists() else item)
    app = EasyOperator(initial="\n".join(initial_parts), preselect_repo=repo)
    app.mainloop()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
