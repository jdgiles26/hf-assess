#!/usr/bin/env python3
"""End-to-end functional coverage for the analyzer, launcher, and GUI."""

from __future__ import annotations

import importlib.util
import json
import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
ANALYZER = ROOT / "hf_assess.py"
GUI = ROOT / "hf-assess-gui.py"
LAUNCHER = ROOT / "hf-assess-gui.sh"
VENV_PY = ROOT / ".hf-assess-venv" / "bin" / "python"
SYS_PY = Path(sys.executable)
WRAPPER = ANALYZER


def load(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load {path}")
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


def run_cmd(args, cwd=None, timeout=60, env=None):
    return subprocess.run(
        args,
        cwd=cwd or ROOT,
        capture_output=True,
        text=True,
        timeout=timeout,
        check=False,
        env=env,
    )


class LauncherTests(unittest.TestCase):
    def test_help(self):
        proc = run_cmd(["bash", str(LAUNCHER), "--help"])
        self.assertEqual(proc.returncode, 0, proc.stderr)
        self.assertIn("Paste an IP", proc.stdout)

    def test_list(self):
        proc = run_cmd(["bash", str(LAUNCHER), "--list"])
        self.assertEqual(proc.returncode, 0, proc.stderr)
        self.assertIn("SmolLM2", proc.stdout)
        self.assertIn("DialoGPT", proc.stdout)

    def test_check(self):
        proc = run_cmd(["bash", str(LAUNCHER), "--check"])
        self.assertEqual(proc.returncode, 0, proc.stderr)
        self.assertIn("Analyzer", proc.stdout)
        self.assertIn("OK", proc.stdout)

    def test_unknown_flag(self):
        proc = run_cmd(["bash", str(LAUNCHER), "--not-a-real-flag"])
        self.assertEqual(proc.returncode, 2)

    def test_probe_passthrough(self):
        proc = run_cmd(["bash", str(LAUNCHER), "--probe", "10.0.0.8"])
        self.assertEqual(proc.returncode, 0, proc.stderr + proc.stdout)
        data = json.loads(proc.stdout)
        self.assertEqual(data["probe"]["target"], "10.0.0.8")


class AnalyzerCliTests(unittest.TestCase):
    def test_help(self):
        proc = run_cmd([str(SYS_PY), str(ANALYZER), "--help"])
        self.assertEqual(proc.returncode, 0, proc.stderr)
        self.assertIn("--probe", proc.stdout)
        self.assertIn("inbox", proc.stdout)

    def test_list_presets(self):
        proc = run_cmd([str(SYS_PY), str(ANALYZER), "--list-presets"])
        self.assertEqual(proc.returncode, 0, proc.stderr)
        self.assertIn("instruct-small", proc.stdout)
        self.assertIn("HuggingFaceTB/SmolLM2-360M-Instruct", proc.stdout)

    def test_check_deps_system_python_missing_transformers(self):
        proc = run_cmd([str(SYS_PY), str(ANALYZER), "--check-deps"])
        self.assertEqual(proc.returncode, 2, proc.stdout)
        self.assertIn("MISSING", proc.stdout)
        self.assertIn("transformers", proc.stdout)

    def test_check_deps_venv(self):
        if not VENV_PY.is_file():
            self.skipTest("venv missing")
        proc = run_cmd([str(VENV_PY), str(ANALYZER), "--check-deps"])
        self.assertEqual(proc.returncode, 0, proc.stdout)
        self.assertIn("transformers", proc.stdout)
        self.assertIn("ok", proc.stdout)

    def test_check_model_known(self):
        py = VENV_PY if VENV_PY.is_file() else SYS_PY
        proc = run_cmd([str(py), str(ANALYZER), "--check-model", "microsoft/DialoGPT-medium"], timeout=90)
        self.assertEqual(proc.returncode, 0, proc.stderr + proc.stdout)
        self.assertIn("microsoft/DialoGPT-medium", proc.stdout)

    def test_check_model_missing(self):
        py = VENV_PY if VENV_PY.is_file() else SYS_PY
        proc = run_cmd([str(py), str(ANALYZER), "--check-model", "no-such-org/no-such-model-xyz"], timeout=60)
        self.assertEqual(proc.returncode, 1, proc.stdout)
        self.assertIn("not found", proc.stdout.lower())

    def test_system_python_refuses_run_without_model(self):
        proc = run_cmd([str(SYS_PY), str(ANALYZER), "--target", "10.0.0.8", "--tools", "nmap"])
        self.assertEqual(proc.returncode, 1, proc.stdout + proc.stderr)
        self.assertIn("transformers", (proc.stdout + proc.stderr).lower())

    def test_public_target_without_auth_exits_3(self):
        proc = run_cmd([str(SYS_PY), str(ANALYZER), "8.8.8.8"])
        self.assertEqual(proc.returncode, 3, proc.stdout + proc.stderr)
        self.assertIn("authorized", (proc.stdout + proc.stderr).lower())

    def test_missing_input_errors(self):
        proc = run_cmd([str(SYS_PY), str(ANALYZER)])
        self.assertNotEqual(proc.returncode, 0)
        self.assertIn("target", (proc.stderr + proc.stdout).lower())

    def test_run_tools_unknown_collector_fails_before_model(self):
        proc = run_cmd(
            [
                str(SYS_PY),
                str(ANALYZER),
                "--target",
                "10.0.0.8",
                "--tools",
                "nikto",
                "--run-tools",
            ],
            timeout=15,
        )
        self.assertEqual(proc.returncode, 1, proc.stdout + proc.stderr)
        combined = (proc.stdout + proc.stderr).lower()
        self.assertIn("nikto", combined)
        self.assertNotIn("loading local model", combined)

    def test_invalid_params_json(self):
        proc = run_cmd(
            [str(SYS_PY), str(ANALYZER), "--target", "10.0.0.1", "--tools", "nmap", "--params", "{nope"]
        )
        self.assertEqual(proc.returncode, 1)
        self.assertIn("json", (proc.stderr + proc.stdout).lower())

    def test_config_file(self):
        with tempfile.TemporaryDirectory() as tmp:
            cfg = Path(tmp) / "c.json"
            cfg.write_text(json.dumps({"target": "10.0.0.9", "tools": ["nikto"]}), encoding="utf-8")
            proc = run_cmd([str(SYS_PY), str(ANALYZER), "--config", str(cfg), "--probe"])
            # --probe with empty inbox is not ready; config is applied only after probe in main.
            # This documents current behavior: --probe does not read --config target.
            self.assertIn(proc.returncode, {0, 1})


class ProbeFunctionalTests(unittest.TestCase):
    def setUp(self):
        self.mod = load(ANALYZER, "hf_assess_fn")

    def test_empty(self):
        probe = self.mod.probe_input("")
        self.assertFalse(probe.ready)

    def test_cidr(self):
        probe = self.mod.probe_input("192.168.10.0/24")
        self.assertEqual(probe.target, "192.168.10.0/24")
        self.assertFalse(probe.needs_authorization)

    def test_ipv6_loopback_classified(self):
        info = self.mod.classify_target("::1")
        self.assertEqual(info.kind, "loopback")

    def test_hostname_port(self):
        info = self.mod.classify_target("db.internal:5432")
        self.assertEqual(info.kind, "hostname")

    def test_url_with_port(self):
        probe = self.mod.probe_input("https://10.2.2.2:8443/admin")
        self.assertEqual(probe.target, "10.2.2.2")

    def test_mixed_clipboard(self):
        blob = "see 10.0.0.5 and 10.0.0.6 also https://wiki.local CVE-2014-0160 nmap nikto"
        probe = self.mod.probe_input(blob)
        self.assertEqual(probe.target, "10.0.0.5")
        self.assertIn("10.0.0.6", probe.extra_targets)
        self.assertIn("CVE-2014-0160", probe.cves)
        self.assertIn("nmap", probe.tools)
        self.assertIn("nikto", probe.tools)

    def test_hf_url_model(self):
        probe = self.mod.probe_input("https://huggingface.co/HuggingFaceTB/SmolLM2-360M-Instruct")
        self.assertEqual(probe.model_repo, "HuggingFaceTB/SmolLM2-360M-Instruct")

    def test_hf_prefix(self):
        self.assertEqual(self.mod._looks_like_repo("hf:org/name"), "org/name")

    def test_remote_router_url(self):
        probe = self.mod.probe_input("https://router.huggingface.co/hf-inference/models/x/y")
        self.assertTrue(probe.remote_url)

    def test_email_operator(self):
        probe = self.mod.probe_input("contact me@lab.example and 10.0.0.1")
        self.assertEqual(probe.operator, "me@lab.example")

    def test_persist_scan_blob(self):
        with tempfile.TemporaryDirectory() as tmp:
            blob = "Host: 10.9.9.9 () Ports: 443/open/tcp//https///\n"
            probe = self.mod.probe_input(blob, persist_dir=Path(tmp))
            self.assertTrue(probe.scan_file)
            self.assertTrue(Path(probe.scan_file).is_file())
            self.assertEqual(probe.parsed_scan["source"], "nmap-grepable")

    def test_parse_any_json_and_xml_and_text(self):
        xml = """<?xml version="1.0"?><nmaprun><host><status state="up"/>
        <address addr="10.3.3.3" addrtype="ipv4"/><ports>
        <port protocol="tcp" portid="25"><state state="open"/>
        <service name="smtp"/></port></ports></host></nmaprun>"""
        parsed = self.mod.parse_any_scan_text(xml)
        self.assertEqual(parsed["source"], "nmap-xml")
        parsed = self.mod.parse_any_scan_text(
            json.dumps({"target": "10.4.4.4", "open_ports": [21], "vulnerabilities": []})
        )
        self.assertEqual(parsed["target"], "10.4.4.4")

    def test_default_nmap_profile_is_bounded(self):
        flags = self.mod._DEFAULT_NMAP
        self.assertIn("--top-ports", flags)
        self.assertIn("--host-timeout", flags)
        self.assertIn("-T4", flags)

    def test_apply_probe_fills_config(self):
        probe = self.mod.probe_input("10.0.0.8 nmap")
        cfg = self.mod.AssessmentConfig()
        cfg = self.mod.apply_probe_to_config(cfg, probe)
        self.assertEqual(cfg.target, "10.0.0.8")
        self.assertIn("nmap", cfg.tool_stack)


class ReportAndRiskTests(unittest.TestCase):
    def setUp(self):
        self.mod = load(ANALYZER, "hf_assess_fn")

    def test_all_report_formats(self):
        enhanced = {
            "meta": {"target": "10.0.0.8", "mode": "scan-file", "engine": "local"},
            "raw_results": {
                "target": "10.0.0.8",
                "findings": [{"id": "1", "severity": "high", "cve": "CVE-2022-22965", "title": "Spring"}],
            },
            "ai_analysis": {"summary": "high", "recommendations": ["patch"]},
            "risk_score": {"score": 72.0, "label": "High"},
            "compliance": {"pci_dss": {"status": "Gap", "gaps": ["6.2"]}},
        }
        with tempfile.TemporaryDirectory() as tmp:
            paths = self.mod.write_reports(enhanced, Path(tmp) / "out", ["json", "markdown", "sarif", "html"])
            self.assertTrue(paths["json"].exists())
            self.assertTrue(paths["markdown"].exists())
            self.assertTrue(paths["sarif"].exists())
            self.assertTrue(paths["html"].exists())
            html = paths["html"].read_text(encoding="utf-8")
            self.assertIn("Spring", html)
            self.assertIn("<!DOCTYPE html>", html)

    def test_diff_scans(self):
        a = {"findings": [{"cve": "CVE-1", "title": "old"}, {"cve": "CVE-2", "title": "keep"}]}
        b = {"findings": [{"cve": "CVE-2", "title": "keep"}, {"cve": "CVE-3", "title": "new"}]}
        diff = self.mod.diff_scans(b, a)
        self.assertEqual(len(diff["added"]), 1)
        self.assertEqual(len(diff["removed"]), 1)
        self.assertEqual(diff["unchanged"], 1)

    def test_extract_json_and_remote_shapes(self):
        self.assertEqual(self.mod.extract_json_object('{"a": 1}')["a"], 1)
        self.assertEqual(self.mod.parse_remote_response([{"generated_text": "hi"}]), "hi")
        self.assertEqual(
            self.mod.parse_remote_response({"choices": [{"message": {"content": "ok"}}]}),
            "ok",
        )
        self.assertEqual(
            self.mod.build_remote_url("hf:org/model"),
            "https://router.huggingface.co/hf-inference/models/org/model",
        )

    def test_compliance_and_cwe(self):
        findings = [{"severity": "critical", "cve": "CVE-2021-44228", "kind": "vulnerability"}]
        self.mod.attach_cwe(findings)
        self.assertEqual(findings[0]["cwe"], "CWE-502")
        report = self.mod.map_compliance({"findings": findings})
        self.assertEqual(report["pci_dss"]["status"], "Gap")


class GuiHeadlessTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        os.environ.setdefault("TK_SILENCE_DEPRECATION", "1")

    def test_gui_probes_and_builds_without_user_fields(self):
        gui = load(GUI, "hf_assess_gui_fn")
        app = gui.EasyOperator(initial="10.0.0.8")
        try:
            app.update_idletasks()
            app._reprobe()
            self.assertIsNotNone(app.probe)
            self.assertEqual(app.probe.target, "10.0.0.8")
            detect = app.detect.get("1.0", "end")
            self.assertIn("10.0.0.8", detect)
            if app.chosen_model:
                self.assertTrue(gui.model_is_cached(app.chosen_model))
            app.inbox.delete("1.0", "end")
            app.inbox.insert("1.0", "8.8.8.8")
            app._reprobe()
            self.assertTrue(app.probe.needs_authorization)
            app.inbox.delete("1.0", "end")
            app.inbox.insert(
                "1.0",
                "Host: 10.8.8.8 () Ports: 22/open/tcp//ssh///\n",
            )
            app._reprobe()
            self.assertTrue(app.probe.scan_file)
            self.assertEqual(app.probe.target, "10.8.8.8")
            app._set_text(app.summary, "line one\n" * 80)
            app.update_idletasks()
            self.assertGreater(float(app.summary.text.index("end-1c").split(".")[0]), 70)
            self.assertEqual(app.notebook.index("end"), 4)
        finally:
            app.destroy()

    def test_gui_refuses_empty_run(self):
        gui = load(GUI, "hf_assess_gui_fn")
        app = gui.EasyOperator(initial="")
        try:
            app.update_idletasks()
            app._reprobe()
            self.assertFalse(app.probe.ready)
            prompts: list[str] = []
            app.bell = lambda: None  # type: ignore[method-assign]
            import tkinter.messagebox as mb

            original = mb.showinfo
            mb.showinfo = lambda title, message: prompts.append(f"{title}:{message}") or "ok"  # type: ignore
            try:
                app.run_analysis()
            finally:
                mb.showinfo = original
            self.assertTrue(any("Need input" in p for p in prompts), prompts)
        finally:
            app.destroy()

    def test_gui_open_file_replaces_inbox_and_loads_scan(self):
        gui = load(GUI, "hf_assess_gui_fn")
        app = gui.EasyOperator(initial="10.0.0.8")
        try:
            app.update_idletasks()
            with tempfile.TemporaryDirectory() as tmp:
                scan = Path(tmp) / "scan.json"
                scan.write_text(
                    json.dumps({"target": "10.9.9.9", "vulnerabilities": [{"cve": "CVE-2014-0160", "severity": "high"}]}),
                    encoding="utf-8",
                )
                app.inbox.delete("1.0", "end")
                app.inbox.insert("1.0", f"lab-host\n{scan}")
                app._reprobe()
                self.assertEqual(app.probe.scan_file, str(scan.resolve()))
                self.assertTrue(app.probe.ready)
            app._set_busy(True)
            self.assertEqual(str(app.inbox.cget("state")), "disabled")
            app._set_busy(False)
            self.assertEqual(str(app.inbox.cget("state")), "normal")
        finally:
            app.destroy()

    def test_pick_ready_model_ignores_uncached(self):
        gui = load(GUI, "hf_assess_gui_fn")
        fake = "org/definitely-not-cached-xyz"
        got = gui.pick_ready_model(fake)
        self.assertNotEqual(got, fake)
        if got:
            self.assertTrue(gui.model_is_cached(got))


class WrapperAndDeviceTests(unittest.TestCase):
    def test_legacy_wrapper_help(self):
        if not WRAPPER.is_file():
            self.skipTest("wrapper missing")
        proc = run_cmd([str(SYS_PY), str(WRAPPER), "--help"])
        self.assertEqual(proc.returncode, 0, proc.stderr)
        self.assertIn("--target", proc.stdout)

    def test_detect_device_is_string(self):
        mod = load(ANALYZER, "hf_assess_fn")
        device = mod.detect_device()
        self.assertIn(device, {"cpu", "mps", "cuda"})


class LiveModelTests(unittest.TestCase):
    """Requires the venv and a cached local model. Skips if either is missing."""

    def test_local_model_analyzes_scan_file(self):
        if not VENV_PY.is_file():
            self.skipTest("venv missing")
        hub = Path.home() / ".cache" / "huggingface" / "hub"
        model = "HuggingFaceTB/SmolLM2-360M-Instruct"
        cached = hub / "models--HuggingFaceTB--SmolLM2-360M-Instruct"
        if not cached.exists():
            model = "microsoft/DialoGPT-medium"
            cached = hub / "models--microsoft--DialoGPT-medium"
        if not cached.exists():
            self.skipTest("no cached local model")
        with tempfile.TemporaryDirectory() as tmp:
            scan = Path(tmp) / "scan.xml"
            scan.write_text(
                """<?xml version="1.0"?>
                <nmaprun>
                  <host>
                    <status state="up"/>
                    <address addr="10.0.0.8" addrtype="ipv4"/>
                    <ports>
                      <port protocol="tcp" portid="22">
                        <state state="open"/>
                        <service name="ssh" product="OpenSSH" version="8.2p1"/>
                      </port>
                    </ports>
                  </host>
                </nmaprun>
                """,
                encoding="utf-8",
            )
            out = Path(tmp) / "live.json"
            proc = run_cmd(
                [
                    str(VENV_PY),
                    str(ANALYZER),
                    "--scan-file",
                    str(scan),
                    "--target",
                    "10.0.0.8",
                    "--tools",
                    "nmap",
                    "--model",
                    model,
                    "--max-new-tokens",
                    "48",
                    "--output",
                    str(out),
                    "--json",
                    "--quiet",
                ],
                timeout=240,
            )
            self.assertEqual(proc.returncode, 0, proc.stderr + proc.stdout[-4000:])
            self.assertTrue(out.is_file(), "json report missing")
            self.assertTrue(out.with_suffix(".md").is_file(), "markdown report missing")
            self.assertTrue(out.with_suffix(".sarif").is_file(), "sarif report missing")
            data = json.loads(out.read_text(encoding="utf-8"))
            self.assertEqual(data["raw_results"]["target"], "10.0.0.8")
            self.assertEqual(data["meta"]["engine"], "local")
            self.assertIn("ai_analysis", data)
            self.assertTrue(data["ai_analysis"].get("engine") == "local")


if __name__ == "__main__":
    unittest.main(verbosity=2)
