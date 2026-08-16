#!/usr/bin/env python3
"""Tests for the Hugging Face authorized assessment analyzer."""

from __future__ import annotations

import importlib.util
import json
import sys
import tempfile
import unittest
from pathlib import Path

SCRIPT = Path(__file__).resolve().parent.parent / "hf_assess.py"


def load_module():
    spec = importlib.util.spec_from_file_location("hf_assess", SCRIPT)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load {SCRIPT}")
    mod = importlib.util.module_from_spec(spec)
    # Python 3.14 dataclasses look up sys.modules[__name__] while the class
    # body is executed. Register before exec_module or @dataclass crashes.
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)
    return mod


class ImportWithoutTransformersTests(unittest.TestCase):
    def test_module_imports_even_if_transformers_missing(self):
        # The original script imported transformers at module top-level, so
        # `python script.py --help` crashed with ModuleNotFoundError.
        mod = load_module()
        self.assertTrue(hasattr(mod, "main"))
        self.assertTrue(hasattr(mod, "probe_dependencies"))

    def test_probe_dependencies_reports_missing_or_present(self):
        mod = load_module()
        report = mod.probe_dependencies()
        self.assertIn("transformers", report)
        self.assertIn("torch", report)
        self.assertIn("requests", report)
        self.assertIsInstance(report["transformers"].available, bool)

    def test_missing_transformers_message_is_actionable(self):
        mod = load_module()
        msg = mod.install_hint()
        self.assertIn("pip install", msg)
        self.assertIn("transformers", msg)


class ScopeAndAuthTests(unittest.TestCase):
    def setUp(self):
        self.mod = load_module()

    def test_classifies_private_cidr(self):
        info = self.mod.classify_target("192.168.1.0/24")
        self.assertEqual(info.kind, "private")
        self.assertTrue(info.is_rfc1918)

    def test_classifies_localhost(self):
        info = self.mod.classify_target("127.0.0.1")
        self.assertEqual(info.kind, "loopback")

    def test_classifies_hostname(self):
        info = self.mod.classify_target("example.com")
        self.assertEqual(info.kind, "hostname")

    def test_classifies_public_ip(self):
        info = self.mod.classify_target("8.8.8.8")
        self.assertEqual(info.kind, "public")
        self.assertFalse(info.is_rfc1918)

    def test_authorization_required_for_public_target(self):
        cfg = self.mod.AssessmentConfig(
            target="8.8.8.8",
            tool_stack=["nmap"],
            authorized=False,
        )
        with self.assertRaises(self.mod.AuthorizationError):
            self.mod.require_authorization(cfg)

    def test_authorization_ok_when_flag_set(self):
        cfg = self.mod.AssessmentConfig(
            target="10.0.0.1",
            tool_stack=["nmap"],
            authorized=True,
            operator="tester",
            engagement_id="ENG-1",
        )
        self.mod.require_authorization(cfg)

    def test_private_target_does_not_require_flag(self):
        cfg = self.mod.AssessmentConfig(
            target="10.0.0.8",
            tool_stack=["nmap"],
            authorized=False,
        )
        self.mod.require_authorization(cfg)

    def test_wide_cidr_is_not_rfc1918(self):
        for cidr in ("0.0.0.0/0", "8.0.0.0/5", "172.0.0.0/8", "192.0.0.0/8", "10.0.0.0/7"):
            info = self.mod.classify_target(cidr)
            self.assertEqual(info.kind, "public", cidr)
            self.assertFalse(info.is_rfc1918, cidr)

    def test_rfc1918_subnet_stays_private(self):
        info = self.mod.classify_target("10.20.0.0/16")
        self.assertEqual(info.kind, "private")
        self.assertTrue(info.is_rfc1918)

    def test_authorization_required_for_wide_cidr(self):
        cfg = self.mod.AssessmentConfig(target="0.0.0.0/0", tool_stack=["nmap"], authorized=False)
        with self.assertRaises(self.mod.AuthorizationError):
            self.mod.require_authorization(cfg)

    def test_authorization_required_for_public_hostname(self):
        cfg = self.mod.AssessmentConfig(target="dns.google", tool_stack=["nmap"], authorized=False)
        with self.assertRaises(self.mod.AuthorizationError):
            self.mod.require_authorization(cfg)

    def test_local_hostname_does_not_require_flag(self):
        for host in ("printer.local", "db.internal", "localhost"):
            cfg = self.mod.AssessmentConfig(target=host, tool_stack=["nmap"], authorized=False)
            self.mod.require_authorization(cfg)

    def test_probe_marks_public_hostname_and_wide_cidr(self):
        self.assertTrue(self.mod.probe_input("example.com").needs_authorization)
        self.assertTrue(self.mod.probe_input("0.0.0.0/0").needs_authorization)
        self.assertFalse(self.mod.probe_input("nas.local").needs_authorization)


class RiskAndComplianceTests(unittest.TestCase):
    def setUp(self):
        self.mod = load_module()

    def test_empty_findings_score_near_zero(self):
        score = self.mod.calculate_risk_score({"findings": []})
        self.assertLess(score.score, 10)
        self.assertEqual(score.label, "Informational")

    def test_critical_cve_raises_score(self):
        results = {
            "findings": [
                {
                    "id": "f1",
                    "severity": "critical",
                    "cve": "CVE-2021-44228",
                    "cvss": 10.0,
                    "title": "Log4Shell",
                }
            ]
        }
        score = self.mod.calculate_risk_score(results)
        self.assertGreaterEqual(score.score, 70)
        self.assertIn(score.label, {"High", "Critical"})

    def test_compliance_is_derived_not_hardcoded_pass(self):
        results = {
            "findings": [
                {"id": "f1", "severity": "critical", "cve": "CVE-2021-44228", "title": "RCE"}
            ]
        }
        report = self.mod.map_compliance(results)
        self.assertNotEqual(report["pci_dss"]["status"], "Compliant")
        self.assertIn("gaps", report["pci_dss"])


class ParserTests(unittest.TestCase):
    def setUp(self):
        self.mod = load_module()

    def test_parse_generic_json_scan(self):
        payload = {
            "target": "10.0.0.5",
            "open_ports": [22, 80],
            "vulnerabilities": [
                {"port": 80, "cve": "CVE-2021-44228", "severity": "Critical"}
            ],
        }
        parsed = self.mod.parse_scan_payload(payload)
        self.assertEqual(parsed["target"], "10.0.0.5")
        self.assertGreaterEqual(len(parsed["findings"]), 1)
        self.assertEqual(parsed["findings"][0]["cve"], "CVE-2021-44228")

    def test_parse_nmap_xml(self):
        xml = """<?xml version="1.0"?>
        <nmaprun>
          <host>
            <status state="up"/>
            <address addr="10.0.0.8" addrtype="ipv4"/>
            <ports>
              <port protocol="tcp" portid="22">
                <state state="open"/>
                <service name="ssh" product="OpenSSH" version="8.2p1"/>
              </port>
              <port protocol="tcp" portid="80">
                <state state="open"/>
                <service name="http" product="Apache" version="2.4.41"/>
              </port>
            </ports>
          </host>
        </nmaprun>
        """
        parsed = self.mod.parse_nmap_xml(xml)
        ports = {f["port"] for f in parsed["findings"] if f.get("kind") == "open_port"}
        self.assertEqual(ports, {22, 80})
        self.assertEqual(parsed["hosts"][0]["address"], "10.0.0.8")

    def test_extract_json_from_messy_llm_text(self):
        text = """Here you go:\n```json\n{\"risk_level\": \"High\", \"summary\": \"ok\"}\n```\nThanks"""
        data = self.mod.extract_json_object(text)
        self.assertEqual(data["risk_level"], "High")


class ReportTests(unittest.TestCase):
    def setUp(self):
        self.mod = load_module()

    def test_markdown_and_sarif_reports_write(self):
        enhanced = {
            "meta": {"target": "10.0.0.8", "mode": "scan-file"},
            "raw_results": {
                "target": "10.0.0.8",
                "findings": [
                    {
                        "id": "f1",
                        "severity": "high",
                        "cve": "CVE-2022-22965",
                        "title": "Spring4Shell",
                    }
                ],
            },
            "ai_analysis": {"summary": "High risk host", "recommendations": ["Patch"]},
            "risk_score": {"score": 72.0, "label": "High"},
            "compliance": {"pci_dss": {"status": "Gap", "gaps": ["6.2"]}},
        }
        with tempfile.TemporaryDirectory() as tmp:
            paths = self.mod.write_reports(enhanced, Path(tmp) / "out")
            self.assertTrue(paths["json"].exists())
            self.assertTrue(paths["markdown"].exists())
            self.assertTrue(paths["sarif"].exists())
            sarif = json.loads(paths["sarif"].read_text(encoding="utf-8"))
            self.assertEqual(sarif["version"], "2.1.0")
            md = paths["markdown"].read_text(encoding="utf-8")
            self.assertIn("Spring4Shell", md)


class ProbeTests(unittest.TestCase):
    def setUp(self):
        self.mod = load_module()

    def test_probes_ipv4(self):
        probe = self.mod.probe_input("10.0.0.8")
        self.assertEqual(probe.target, "10.0.0.8")
        self.assertTrue(probe.ready)
        self.assertFalse(probe.needs_authorization)

    def test_probes_public_ip_needs_auth(self):
        probe = self.mod.probe_input("8.8.8.8")
        self.assertEqual(probe.target, "8.8.8.8")
        self.assertTrue(probe.needs_authorization)

    def test_probes_url_host(self):
        probe = self.mod.probe_input("https://intranet.local/login")
        self.assertEqual(probe.target, "intranet.local")
        self.assertTrue(probe.ready)

    def test_probes_nmap_grepable(self):
        blob = "Host: 10.0.0.8 (gw) Ports: 22/open/tcp//ssh///, 80/open/tcp//http///\n"
        probe = self.mod.probe_input(blob)
        self.assertEqual(probe.target, "10.0.0.8")
        self.assertTrue(probe.parsed_scan)
        ports = {f["port"] for f in probe.parsed_scan["findings"] if f.get("kind") == "open_port"}
        self.assertEqual(ports, {22, 80})

    def test_probes_nmap_text(self):
        blob = """Nmap scan report for 10.0.0.8
PORT   STATE SERVICE
22/tcp open  ssh
443/tcp open  ssl/http
"""
        parsed = self.mod.parse_nmap_text(blob)
        self.assertEqual(parsed["target"], "10.0.0.8")
        ports = {f["port"] for f in parsed["findings"]}
        self.assertEqual(ports, {22, 443})

    def test_probes_cves_and_token_and_model(self):
        blob = "hf_abcdefghijklmnopqrstuvwxyz12 check microsoft/DialoGPT-medium CVE-2021-44228"
        probe = self.mod.probe_input(blob)
        self.assertEqual(probe.hf_token, "hf_abcdefghijklmnopqrstuvwxyz12")
        self.assertIn("CVE-2021-44228", probe.cves)
        self.assertEqual(probe.model_repo, "microsoft/DialoGPT-medium")

    def test_probes_existing_file(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "scan.json"
            path.write_text(
                json.dumps(
                    {
                        "target": "10.1.1.4",
                        "vulnerabilities": [{"cve": "CVE-2022-22965", "severity": "High"}],
                    }
                ),
                encoding="utf-8",
            )
            probe = self.mod.probe_input(str(path))
            self.assertEqual(probe.scan_file, str(path.resolve()))
            self.assertEqual(probe.target, "10.1.1.4")
            self.assertTrue(probe.ready)

    def test_cli_probe_does_not_need_transformers(self):
        import subprocess

        proc = subprocess.run(
            [sys.executable, str(SCRIPT), "--probe", "10.0.0.8"],
            capture_output=True,
            text=True,
            check=False,
        )
        self.assertEqual(proc.returncode, 0, proc.stderr + proc.stdout)
        data = json.loads(proc.stdout)
        self.assertEqual(data["probe"]["target"], "10.0.0.8")


class CliSmokeTests(unittest.TestCase):
    def test_help_does_not_require_transformers(self):
        import subprocess

        proc = subprocess.run(
            [sys.executable, str(SCRIPT), "--help"],
            capture_output=True,
            text=True,
            check=False,
        )
        self.assertEqual(proc.returncode, 0, proc.stderr)
        self.assertIn("--target", proc.stdout)
        self.assertIn("--model", proc.stdout)
        self.assertNotIn("--heuristic-only", proc.stdout)
        self.assertNotIn("--demo", proc.stdout)

    def test_refuses_to_run_without_model(self):
        import subprocess

        proc = subprocess.run(
            [
                sys.executable,
                str(SCRIPT),
                "--target",
                "10.0.0.8",
                "--tools",
                "nikto",
            ],
            capture_output=True,
            text=True,
            check=False,
        )
        combined = proc.stderr + proc.stdout
        self.assertNotEqual(proc.returncode, 0, combined)
        self.assertTrue(
            "transformers" in combined.lower() or "model" in combined.lower(),
            combined,
        )


if __name__ == "__main__":
    unittest.main()
