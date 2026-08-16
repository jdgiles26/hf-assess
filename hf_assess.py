#!/usr/bin/env python3
"""
AI-enhanced security analysis using a real Hugging Face model.

A local `transformers` model or a remote Hugging Face endpoint is required.
`--help`, `--check-deps`, `--check-model`, and `--bootstrap` can run first
so a missing package does not hide the real error.

This tool does not invent scan results and does not run exploits.
"""

from __future__ import annotations

import argparse
import hashlib
import ipaddress
import json
import logging
import os
import re
import sys
import time
import xml.etree.ElementTree as ET
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple
from urllib.parse import urlparse

SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_VENV = SCRIPT_DIR / ".hf-assess-venv"
DEFAULT_REQUIREMENTS = SCRIPT_DIR / "hf_assess_requirements.txt"
LOGGER_NAME = "hf_assess"

logger = logging.getLogger(LOGGER_NAME)

# ---------------------------------------------------------------------------
# Errors / small value types
# ---------------------------------------------------------------------------


class AssessmentError(Exception):
    """Base error for this tool."""


class AuthorizationError(AssessmentError):
    """Operator has not attested authorization for this target."""


class DependencyError(AssessmentError):
    """Optional heavy dependency is missing or broken."""


class ModelError(AssessmentError):
    """Local or remote model could not be used."""


class ParseError(AssessmentError):
    """Scan input could not be parsed."""


@dataclass
class DepStatus:
    name: str
    available: bool
    version: Optional[str] = None
    error: Optional[str] = None
    path: Optional[str] = None


@dataclass
class TargetInfo:
    raw: str
    kind: str
    is_rfc1918: bool = False
    is_loopback: bool = False
    is_link_local: bool = False
    is_multicast: bool = False
    is_public: bool = False
    network: Optional[str] = None


@dataclass
class RiskScore:
    score: float
    label: str
    breakdown: Dict[str, Any] = field(default_factory=dict)


@dataclass
class AssessmentConfig:
    """Runtime configuration for an authorized assessment analysis."""

    target: str = "unspecified"
    tool_stack: List[str] = field(default_factory=list)
    parameters: Dict[str, Any] = field(default_factory=dict)
    intrusive_mode: bool = False
    model_name: str = "microsoft/DialoGPT-medium"
    remote_url: Optional[str] = None
    use_remote: bool = False
    output_file: Optional[str] = None
    authorized: bool = False
    operator: Optional[str] = None
    engagement_id: Optional[str] = None
    scan_file: Optional[str] = None
    diff_scan: Optional[str] = None
    enrich_cves: bool = False
    offline: bool = False
    cache_dir: Optional[str] = None
    hf_token: Optional[str] = None
    max_new_tokens: int = 400
    temperature: float = 0.3
    timeout: int = 45
    formats: List[str] = field(default_factory=lambda: ["json", "markdown", "sarif"])
    quiet: bool = False
    verbose: bool = False
    json_stdout: bool = False
    notes: str = ""
    run_tools: bool = False


# Backward-compatible name used by the original script.
IntrusionConfig = AssessmentConfig


# ---------------------------------------------------------------------------
# Dependency probing (never import transformers at module import time)
# ---------------------------------------------------------------------------

_OPTIONAL_MODULES = ("transformers", "torch", "huggingface_hub", "requests", "accelerate")


def probe_dependencies(names: Sequence[str] = _OPTIONAL_MODULES) -> Dict[str, DepStatus]:
    """Return availability of optional packages without raising."""
    report: Dict[str, DepStatus] = {}
    for name in names:
        try:
            module = __import__(name)
            version = getattr(module, "__version__", None)
            path = getattr(module, "__file__", None)
            report[name] = DepStatus(name=name, available=True, version=version, path=path)
        except Exception as exc:  # noqa: BLE001 — probe must never crash
            report[name] = DepStatus(name=name, available=False, error=str(exc))
    return report


def install_hint() -> str:
    venv = DEFAULT_VENV
    req = DEFAULT_REQUIREMENTS
    req_arg = str(req) if req.exists() else "transformers accelerate safetensors huggingface_hub requests"
    pip_req = f"-r {req}" if req.exists() else req_arg
    return (
        "The Hugging Face `transformers` package is not installed in this Python "
        f"({sys.executable}, {sys.version.split()[0]}).\n\n"
        "Homebrew Python blocks `pip install --user` (PEP 668). Use a venv:\n\n"
        f"  python3 -m venv {venv}\n"
        f"  source {venv}/bin/activate\n"
        f"  pip install {pip_req}\n\n"
        "Then rerun this script with that interpreter:\n"
        f"  {venv}/bin/python {Path(__file__).name} --check-deps\n\n"
        "Or skip local weights and call a hosted model:\n"
        f"  {venv}/bin/python {Path(__file__).name} --target HOST --tools nmap "
        "--remote-model hf:HuggingFaceTB/SmolLM2-360M-Instruct\n"
    )


def resolve_hf_token(explicit: Optional[str] = None) -> Optional[str]:
    if explicit:
        return explicit.strip() or None
    for key in ("HF_TOKEN", "HUGGINGFACE_API_KEY", "HUGGING_FACE_HUB_TOKEN", "HUGGINGFACEHUB_API_TOKEN"):
        value = os.getenv(key)
        if value:
            return value.strip()
    return None


# ---------------------------------------------------------------------------
# Target classification + authorization
# ---------------------------------------------------------------------------


def classify_target(raw: str) -> TargetInfo:
    value = (raw or "").strip()
    if not value:
        return TargetInfo(raw=raw, kind="unknown")

    host = value
    if "://" in host:
        host = urlparse(host).hostname or host
    host = host.split("%")[0].strip().strip("[]")
    if host.count(":") == 1 and not host.startswith(":"):
        # hostname:port or ipv4:port
        left, right = host.rsplit(":", 1)
        if right.isdigit():
            host = left

    try:
        network = ipaddress.ip_network(host, strict=False)
        kind = _network_kind(network)
        return TargetInfo(
            raw=raw,
            kind=kind,
            is_rfc1918=_is_rfc1918(network),
            is_loopback=network.is_loopback,
            is_link_local=network.is_link_local,
            is_multicast=network.is_multicast,
            is_public=kind == "public",
            network=str(network),
        )
    except ValueError:
        pass

    try:
        ip = ipaddress.ip_address(host)
        kind = _ip_kind(ip)
        return TargetInfo(
            raw=raw,
            kind=kind,
            is_rfc1918=_is_rfc1918(ip),
            is_loopback=ip.is_loopback,
            is_link_local=ip.is_link_local,
            is_multicast=ip.is_multicast,
            is_public=kind == "public",
            network=str(ip),
        )
    except ValueError:
        return TargetInfo(raw=raw, kind="hostname")


_RFC1918 = (
    ipaddress.ip_network("10.0.0.0/8"),
    ipaddress.ip_network("172.16.0.0/12"),
    ipaddress.ip_network("192.168.0.0/16"),
)
_LOCAL_HOST_SUFFIXES = (".local", ".internal", ".lan", ".home", ".corp", ".intranet", ".localhost")
_LOCAL_HOST_LABELS = {"localhost", "localhost.localdomain"}


def _is_rfc1918(obj: ipaddress._BaseNetwork | ipaddress._BaseAddress) -> bool:
    """True only for addresses/networks entirely inside RFC1918. Supernets are not private."""
    if isinstance(obj, ipaddress.IPv4Network):
        return any(obj == net or obj.subnet_of(net) for net in _RFC1918)
    if isinstance(obj, ipaddress.IPv4Address):
        return any(obj in net for net in _RFC1918)
    return False


def is_local_hostname(raw: str) -> bool:
    host = (raw or "").strip().lower().rstrip(".")
    if "://" in host:
        host = (urlparse(host).hostname or host).lower()
    if host.count(":") == 1 and not host.startswith("["):
        left, right = host.rsplit(":", 1)
        if right.isdigit():
            host = left
    host = host.strip("[]")
    if host in _LOCAL_HOST_LABELS:
        return True
    if "." not in host:
        return True
    return any(host.endswith(suffix) for suffix in _LOCAL_HOST_SUFFIXES)


def target_requires_authorization(info: TargetInfo) -> bool:
    if info.kind == "public":
        return True
    if info.kind == "hostname":
        return not is_local_hostname(info.raw)
    return False


def _network_kind(network: ipaddress._BaseNetwork) -> str:
    if network.is_loopback:
        return "loopback"
    if network.is_link_local:
        return "link_local"
    if network.is_multicast:
        return "multicast"
    if _is_rfc1918(network) or network.is_private:
        return "private"
    if network.is_reserved or network.is_unspecified:
        return "reserved"
    return "public"


def _ip_kind(ip: ipaddress._BaseAddress) -> str:
    if ip.is_loopback:
        return "loopback"
    if ip.is_link_local:
        return "link_local"
    if ip.is_multicast:
        return "multicast"
    if _is_rfc1918(ip) or ip.is_private:
        return "private"
    if ip.is_reserved or ip.is_unspecified:
        return "reserved"
    return "public"


def require_authorization(config: AssessmentConfig) -> None:
    """Refuse live analysis of public IPs, wide CIDRs, and public DNS names without attestation."""
    if config.authorized:
        return
    if config.scan_file and not config.run_tools:
        return
    info = classify_target(config.target)
    if not target_requires_authorization(info):
        return
    raise AuthorizationError(
        f"Refusing public target {config.target!r} ({info.kind}) without --authorized.\n"
        "Pass --authorized --operator NAME --engagement-id ID, or analyze a --scan-file."
    )


# ---------------------------------------------------------------------------
# Scan parsers
# ---------------------------------------------------------------------------

_CVE_RE = re.compile(r"CVE-\d{4}-\d{4,7}", re.IGNORECASE)
_SEVERITY_ALIASES = {
    "crit": "critical",
    "critical": "critical",
    "high": "high",
    "med": "medium",
    "medium": "medium",
    "moderate": "medium",
    "low": "low",
    "info": "informational",
    "informational": "informational",
    "none": "informational",
}


def normalize_severity(value: Any) -> str:
    if value is None:
        return "informational"
    key = str(value).strip().lower()
    return _SEVERITY_ALIASES.get(key, key if key in _SEVERITY_ALIASES.values() else "informational")


def _finding_id(parts: Iterable[Any]) -> str:
    blob = "|".join(str(p) for p in parts if p is not None)
    return hashlib.sha1(blob.encode("utf-8")).hexdigest()[:12]


def parse_scan_payload(payload: Any) -> Dict[str, Any]:
    """Normalize JSON-ish scan data (native format or loosely structured)."""
    if isinstance(payload, str):
        payload = json.loads(payload)
    if not isinstance(payload, dict):
        raise ParseError("scan payload must be a JSON object")

    if "findings" in payload and isinstance(payload["findings"], list):
        findings = [_normalize_finding(item, idx) for idx, item in enumerate(payload["findings"])]
        result = dict(payload)
        result["findings"] = _dedupe_findings(findings)
        result.setdefault("target", payload.get("target") or "unknown")
        result.setdefault("hosts", payload.get("hosts") or [])
        return result

    scan = payload.get("scan_results") if isinstance(payload.get("scan_results"), dict) else payload
    findings: List[Dict[str, Any]] = []

    for idx, vuln in enumerate(scan.get("vulnerabilities") or payload.get("vulnerabilities") or []):
        findings.append(_normalize_finding(vuln, idx))

    for port in scan.get("open_ports") or payload.get("open_ports") or []:
        findings.append(
            {
                "id": _finding_id(("port", port)),
                "kind": "open_port",
                "title": f"Open port {port}",
                "port": int(port) if str(port).isdigit() else port,
                "severity": "informational",
            }
        )

    for service in scan.get("services") or payload.get("services") or []:
        findings.append(
            {
                "id": _finding_id(("service", service)),
                "kind": "service",
                "title": f"Service banner: {service}",
                "severity": "informational",
                "evidence": str(service),
            }
        )

    return {
        "target": payload.get("target") or scan.get("target") or "unknown",
        "tools_used": payload.get("tools_used") or payload.get("tools") or [],
        "hosts": payload.get("hosts") or [],
        "findings": _dedupe_findings(findings),
        "status": payload.get("status") or "parsed",
        "source": payload.get("source") or "json",
    }


def _normalize_finding(item: Any, idx: int) -> Dict[str, Any]:
    if isinstance(item, str):
        cves = [c.upper() for c in _CVE_RE.findall(item)]
        return {
            "id": _finding_id(("str", item, idx)),
            "kind": "note",
            "title": item[:120],
            "severity": "informational",
            "cve": cves[0] if cves else None,
            "cves": cves,
        }
    if not isinstance(item, dict):
        return {
            "id": _finding_id(("raw", idx)),
            "kind": "note",
            "title": str(item),
            "severity": "informational",
        }

    text = " ".join(str(item.get(k, "")) for k in ("title", "name", "description", "cve", "id", "output"))
    cves = [c.upper() for c in _CVE_RE.findall(text)]
    if item.get("cve"):
        cves.insert(0, str(item["cve"]).upper())
    # unique preserve order
    seen = set()
    ordered = []
    for cve in cves:
        if cve not in seen:
            seen.add(cve)
            ordered.append(cve)

    title = item.get("title") or item.get("name") or (ordered[0] if ordered else f"Finding {idx + 1}")
    severity = normalize_severity(item.get("severity") or item.get("risk") or item.get("cvss_severity"))
    cvss = item.get("cvss") or item.get("cvss_score")
    try:
        cvss_val = float(cvss) if cvss is not None else None
    except (TypeError, ValueError):
        cvss_val = None

    finding = {
        "id": str(item.get("id") or _finding_id((title, item.get("port"), ordered, idx))),
        "kind": item.get("kind") or ("vulnerability" if ordered else "finding"),
        "title": str(title),
        "severity": severity,
        "port": item.get("port"),
        "cve": ordered[0] if ordered else None,
        "cves": ordered,
        "cvss": cvss_val,
        "evidence": item.get("evidence") or item.get("description") or item.get("output"),
        "service": item.get("service"),
        "host": item.get("host") or item.get("address"),
    }
    return finding


def _dedupe_findings(findings: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    seen: set[str] = set()
    out: List[Dict[str, Any]] = []
    for finding in findings:
        key = finding.get("id") or _finding_id(finding.values())
        if key in seen:
            continue
        seen.add(str(key))
        finding["id"] = str(key)
        out.append(finding)
    return out


def parse_nmap_xml(xml_text: str) -> Dict[str, Any]:
    try:
        root = ET.fromstring(xml_text)
    except ET.ParseError as exc:
        raise ParseError(f"invalid nmap XML: {exc}") from exc

    hosts: List[Dict[str, Any]] = []
    findings: List[Dict[str, Any]] = []
    primary_target = None

    for host in root.findall("host"):
        status = (host.findtext("status") or "")
        state_el = host.find("status")
        state = state_el.get("state") if state_el is not None else "unknown"
        addr_el = host.find("address")
        address = addr_el.get("addr") if addr_el is not None else "unknown"
        if primary_target is None:
            primary_target = address
        hostname_el = host.find("hostnames/hostname")
        hostname = hostname_el.get("name") if hostname_el is not None else None
        host_rec = {
            "address": address,
            "hostname": hostname,
            "state": state,
            "status": status,
            "ports": [],
        }
        ports_el = host.find("ports")
        if ports_el is not None:
            for port_el in ports_el.findall("port"):
                try:
                    port_id = int(port_el.get("portid", "0"))
                except ValueError:
                    continue
                proto = port_el.get("protocol") or "tcp"
                pstate = (port_el.find("state").get("state") if port_el.find("state") is not None else "unknown")
                service_el = port_el.find("service")
                product = service_el.get("product") if service_el is not None else None
                version = service_el.get("version") if service_el is not None else None
                name = service_el.get("name") if service_el is not None else None
                banner = " ".join(p for p in (name, product, version) if p)
                host_rec["ports"].append(
                    {
                        "port": port_id,
                        "protocol": proto,
                        "state": pstate,
                        "service": banner or name,
                    }
                )
                if pstate == "open":
                    findings.append(
                        {
                            "id": _finding_id(("open", address, proto, port_id)),
                            "kind": "open_port",
                            "title": f"Open {proto.upper()} {port_id}" + (f" ({banner})" if banner else ""),
                            "severity": "informational",
                            "port": port_id,
                            "protocol": proto,
                            "service": banner or name,
                            "host": address,
                        }
                    )
                for script in port_el.findall("script"):
                    output = script.get("output") or ""
                    for cve in _CVE_RE.findall(output):
                        findings.append(
                            {
                                "id": _finding_id(("script", address, port_id, cve)),
                                "kind": "vulnerability",
                                "title": f"{cve.upper()} via {script.get('id')}",
                                "severity": "high",
                                "cve": cve.upper(),
                                "cves": [cve.upper()],
                                "port": port_id,
                                "host": address,
                                "evidence": output[:2000],
                            }
                        )
        hosts.append(host_rec)

    return {
        "target": primary_target or "unknown",
        "hosts": hosts,
        "findings": _dedupe_findings(findings),
        "source": "nmap-xml",
        "status": "parsed",
    }


def parse_nmap_grepable(text: str) -> Dict[str, Any]:
    hosts: List[Dict[str, Any]] = []
    findings: List[Dict[str, Any]] = []
    primary = None
    for line in text.splitlines():
        if not line.startswith("Host:"):
            continue
        host_m = re.search(r"Host:\s+(\S+)", line)
        if not host_m:
            continue
        address = host_m.group(1)
        if primary is None:
            primary = address
        ports: List[Dict[str, Any]] = []
        ports_m = re.search(r"Ports:\s+(.*)$", line)
        if ports_m:
            for chunk in ports_m.group(1).split(","):
                parts = chunk.strip().split("/")
                if len(parts) < 3:
                    continue
                try:
                    port_id = int(parts[0])
                except ValueError:
                    continue
                state = parts[1]
                proto = parts[2] or "tcp"
                service = parts[4] if len(parts) > 4 else ""
                ports.append({"port": port_id, "protocol": proto, "state": state, "service": service})
                if state == "open":
                    findings.append(
                        {
                            "id": _finding_id(("open", address, proto, port_id)),
                            "kind": "open_port",
                            "title": f"Open {proto.upper()} {port_id}" + (f" ({service})" if service else ""),
                            "severity": "informational",
                            "port": port_id,
                            "protocol": proto,
                            "service": service,
                            "host": address,
                        }
                    )
        hosts.append({"address": address, "ports": ports, "state": "up"})
    if not hosts:
        raise ParseError("no greppable nmap hosts found")
    return {
        "target": primary or "unknown",
        "hosts": hosts,
        "findings": _dedupe_findings(findings),
        "source": "nmap-grepable",
        "status": "parsed",
    }


def parse_nmap_text(text: str) -> Dict[str, Any]:
    if "Nmap scan report" not in text and not re.search(r"^\d+/tcp\s+\S+", text, re.M):
        raise ParseError("not nmap normal output")
    hosts: List[Dict[str, Any]] = []
    findings: List[Dict[str, Any]] = []
    current = None
    for line in text.splitlines():
        report = re.match(r"Nmap scan report for (.+)$", line.strip())
        if report:
            token = report.group(1).strip()
            addr_m = re.search(r"\((\d+\.\d+\.\d+\.\d+)\)", token)
            address = addr_m.group(1) if addr_m else token.split()[0]
            current = {"address": address, "ports": [], "state": "up"}
            hosts.append(current)
            continue
        port_m = re.match(r"(\d+)\/(tcp|udp)\s+(\S+)\s+(\S+)(?:\s+(.*))?$", line.strip())
        if port_m and current is not None:
            port_id = int(port_m.group(1))
            proto = port_m.group(2)
            state = port_m.group(3)
            service = (port_m.group(5) or port_m.group(4) or "").strip()
            current["ports"].append({"port": port_id, "protocol": proto, "state": state, "service": service})
            if state == "open":
                findings.append(
                    {
                        "id": _finding_id(("open", current["address"], proto, port_id)),
                        "kind": "open_port",
                        "title": f"Open {proto.upper()} {port_id}" + (f" ({service})" if service else ""),
                        "severity": "informational",
                        "port": port_id,
                        "protocol": proto,
                        "service": service,
                        "host": current["address"],
                    }
                )
            continue
        for cve in _CVE_RE.findall(line):
            host = current["address"] if current else None
            findings.append(
                {
                    "id": _finding_id(("cve", host, cve)),
                    "kind": "vulnerability",
                    "title": cve.upper(),
                    "severity": "high",
                    "cve": cve.upper(),
                    "cves": [cve.upper()],
                    "host": host,
                    "evidence": line.strip()[:2000],
                }
            )
    if not hosts and not findings:
        raise ParseError("no nmap hosts or findings")
    return {
        "target": hosts[0]["address"] if hosts else "unknown",
        "hosts": hosts,
        "findings": _dedupe_findings(findings),
        "source": "nmap-text",
        "status": "parsed",
    }


def parse_loose_evidence(text: str) -> Dict[str, Any]:
    findings: List[Dict[str, Any]] = []
    for idx, cve in enumerate(_CVE_RE.findall(text)):
        findings.append(
            {
                "id": _finding_id(("cve", cve, idx)),
                "kind": "vulnerability",
                "title": cve.upper(),
                "severity": "high",
                "cve": cve.upper(),
                "cves": [cve.upper()],
                "evidence": cve.upper(),
            }
        )
    for match in re.finditer(r"\bport(?:s)?\s*[:=]?\s*(\d{1,5})\b", text, re.I):
        port_id = int(match.group(1))
        if 1 <= port_id <= 65535:
            findings.append(
                {
                    "id": _finding_id(("port-mention", port_id)),
                    "kind": "open_port",
                    "title": f"Mentioned port {port_id}",
                    "severity": "informational",
                    "port": port_id,
                }
            )
    ips = extract_targets(text)
    return {
        "target": ips[0] if ips else "unknown",
        "hosts": [{"address": ip} for ip in ips],
        "findings": _dedupe_findings(findings),
        "source": "loose-text",
        "status": "parsed",
    }


def parse_any_scan_text(text: str) -> Dict[str, Any]:
    stripped = text.lstrip()
    if stripped.startswith("<?xml") or "<nmaprun" in stripped[:400]:
        return parse_nmap_xml(text)
    if stripped.startswith("Host:") or "\nHost:" in text:
        try:
            return parse_nmap_grepable(text)
        except ParseError:
            pass
    if stripped[:1] in "{[":
        try:
            return parse_scan_payload(json.loads(text))
        except (json.JSONDecodeError, ParseError):
            pass
    try:
        return parse_nmap_text(text)
    except ParseError:
        pass
    loose = parse_loose_evidence(text)
    if loose["findings"] or (loose["target"] and loose["target"] != "unknown"):
        return loose
    raise ParseError("could not recognize scan or evidence format")


def load_scan_file(path: Path) -> Dict[str, Any]:
    text = path.read_text(encoding="utf-8", errors="replace")
    parsed = parse_any_scan_text(text)
    parsed["source_file"] = str(path)
    return parsed


# ---------------------------------------------------------------------------
# Input probing — accept almost anything and classify it
# ---------------------------------------------------------------------------

_IPV4_RE = re.compile(r"\b(?:(?:25[0-5]|2[0-4]\d|[01]?\d\d?)\.){3}(?:25[0-5]|2[0-4]\d|[01]?\d\d?)(?:/\d{1,2})?\b")
_URL_RE = re.compile(r"https?://[^\s<>\"']+", re.I)
_HF_TOKEN_RE = re.compile(r"\bhf_[A-Za-z0-9]{16,}\b")
_HF_REPO_RE = re.compile(r"\b([A-Za-z0-9](?:[A-Za-z0-9._-]{0,38}[A-Za-z0-9])?)/([A-Za-z0-9][A-Za-z0-9._-]{1,120})\b")
_EMAIL_RE = re.compile(r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}\b")
_HOSTNAME_RE = re.compile(r"\b(?:[a-z0-9](?:[a-z0-9-]{0,61}[a-z0-9])?\.)+[a-z]{2,}\b", re.I)
_TOOL_WORDS = {
    "nmap": "nmap",
    "masscan": "masscan",
    "nikto": "nikto",
    "nuclei": "nuclei",
    "nessus": "nessus",
    "openvas": "openvas",
    "burp": "burp",
    "zap": "zap",
    "sqlmap": "sqlmap",
}


@dataclass
class ProbeResult:
    target: Optional[str] = None
    extra_targets: List[str] = field(default_factory=list)
    scan_file: Optional[str] = None
    parsed_scan: Optional[Dict[str, Any]] = None
    tools: List[str] = field(default_factory=list)
    model_repo: Optional[str] = None
    hf_token: Optional[str] = None
    remote_url: Optional[str] = None
    operator: Optional[str] = None
    cves: List[str] = field(default_factory=list)
    urls: List[str] = field(default_factory=list)
    kinds: List[str] = field(default_factory=list)
    notes: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)
    needs_authorization: bool = False
    ready: bool = False
    summary: str = ""

    def as_dict(self) -> Dict[str, Any]:
        return asdict(self)


def extract_targets(text: str) -> List[str]:
    found: List[str] = []

    def add(item: str) -> None:
        item = item.strip().strip(".,;()[]")
        if item and item not in found:
            found.append(item)

    for match in _IPV4_RE.findall(text):
        add(match)
    for match in _URL_RE.findall(text):
        host = urlparse(match).hostname
        if host:
            add(host)
    for match in _HOSTNAME_RE.findall(text):
        lower = match.lower()
        if lower.endswith((".png", ".jpg", ".json", ".xml", ".txt", ".md", ".py", ".sh")):
            continue
        if lower in {"github.com", "huggingface.co", "www.huggingface.co"}:
            continue
        add(match)
    return found


def _looks_like_repo(text: str) -> Optional[str]:
    value = text.strip()
    if value.startswith("hf:"):
        value = value[3:]
    if value.startswith("https://huggingface.co/"):
        parts = urlparse(value).path.strip("/").split("/")
        if parts and parts[0] == "models":
            parts = parts[1:]
        if len(parts) >= 2:
            return f"{parts[0]}/{parts[1]}"
    match = _HF_REPO_RE.search(value)
    if not match:
        return None
    left, right = match.group(1), match.group(2)
    if "." in left:
        return None
    if right.lower().endswith((".xml", ".json", ".txt", ".csv", ".html", ".py")):
        return None
    return f"{left}/{right}"


def _existing_path(token: str) -> Optional[Path]:
    raw = token.strip().strip("'\"")
    if raw.startswith("file://"):
        raw = urlparse(raw).path
    path = Path(raw).expanduser()
    try:
        if path.is_file():
            return path.resolve()
    except OSError:
        return None
    return None


def probe_input(raw: str, persist_dir: Optional[Path] = None) -> ProbeResult:
    """Classify pasted text, file paths, URLs, models, tokens, and scan blobs."""
    result = ProbeResult()
    blob = (raw or "").strip()
    if not blob:
        result.summary = "Waiting for a target, file, or pasted scan."
        return result

    path_tokens = []
    for line in re.split(r"[\n\r]+", blob):
        candidate = _existing_path(line.strip())
        if candidate:
            path_tokens.append(candidate)
    if len(path_tokens) == 1 and blob.count("\n") <= 2 and _existing_path(blob.splitlines()[0]):
        path = path_tokens[0]
        result.scan_file = str(path)
        result.kinds.append("file")
        try:
            parsed = load_scan_file(path)
            result.parsed_scan = parsed
            result.target = parsed.get("target") if parsed.get("target") not in (None, "unknown") else None
            result.tools = list(parsed.get("tools_used") or [])
            result.kinds.append(parsed.get("source") or "scan")
            result.notes.append(f"Loaded {path.name}")
        except ParseError as exc:
            result.warnings.append(str(exc))
            text = path.read_text(encoding="utf-8", errors="replace")
            blob = text
            result.notes.append(f"Read {path.name} as text")

    token_m = _HF_TOKEN_RE.search(blob)
    if token_m:
        result.hf_token = token_m.group(0)
        result.kinds.append("hf-token")

    repo = _looks_like_repo(blob) if "\n" not in blob.strip() or "huggingface.co" in blob else None
    if repo is None:
        for match in _HF_REPO_RE.finditer(blob):
            if "." not in match.group(1):
                repo = f"{match.group(1)}/{match.group(2)}"
                break
    if repo and any(key in blob.lower() for key in ("huggingface", "hf:", "instruct", "gpt", "llama", "qwen", "dialogpt", "smolm")):
        result.model_repo = repo
        result.kinds.append("model")
    elif repo and blob.strip() in {repo, f"hf:{repo}"} or (blob.strip().startswith("https://huggingface.co/") and repo):
        result.model_repo = repo
        result.kinds.append("model")

    for url in _URL_RE.findall(blob):
        result.urls.append(url)
        if "huggingface.co" in url and not result.model_repo:
            maybe = _looks_like_repo(url)
            if maybe:
                result.model_repo = maybe
                result.kinds.append("model")
        if "api-inference.huggingface" in url or "router.huggingface.co" in url:
            result.remote_url = url
            result.kinds.append("remote-endpoint")

    email = _EMAIL_RE.search(blob)
    if email:
        result.operator = email.group(0)
        result.kinds.append("operator")

    for word, tool in _TOOL_WORDS.items():
        if re.search(rf"\b{re.escape(word)}\b", blob, re.I):
            if tool not in result.tools:
                result.tools.append(tool)

    result.cves = [c.upper() for c in dict.fromkeys(_CVE_RE.findall(blob))]
    if result.cves:
        result.kinds.append("cves")

    looks_like_scan = any(
        marker in blob
        for marker in ("<nmaprun", "Nmap scan report", "Host: ", "PORT     STATE", "open_ports", "vulnerabilities")
    ) or blob.lstrip()[:1] in "{["
    if looks_like_scan and not result.parsed_scan:
        try:
            parsed = parse_any_scan_text(blob)
            result.parsed_scan = parsed
            result.kinds.append(parsed.get("source") or "scan")
            if persist_dir is not None:
                persist_dir.mkdir(parents=True, exist_ok=True)
                suffix = ".xml" if parsed.get("source") == "nmap-xml" else ".txt"
                digest = hashlib.sha1(blob.encode("utf-8", errors="replace")).hexdigest()[:10]
                saved = persist_dir / f"inbox-{digest}{suffix}"
                saved.write_text(blob, encoding="utf-8")
                result.scan_file = str(saved)
        except ParseError as exc:
            result.warnings.append(str(exc))

    if result.cves and not result.parsed_scan:
        try:
            result.parsed_scan = parse_loose_evidence(blob)
            result.kinds.append("loose-text")
            if persist_dir is not None:
                persist_dir.mkdir(parents=True, exist_ok=True)
                digest = hashlib.sha1(blob.encode("utf-8", errors="replace")).hexdigest()[:10]
                saved = persist_dir / f"inbox-{digest}.txt"
                saved.write_text(blob, encoding="utf-8")
                result.scan_file = str(saved)
        except ParseError:
            pass

    targets = extract_targets(blob)
    if result.parsed_scan and result.parsed_scan.get("target") not in (None, "unknown"):
        if result.parsed_scan["target"] not in targets:
            targets.insert(0, str(result.parsed_scan["target"]))
    if targets:
        result.target = result.target or targets[0]
        result.extra_targets = [t for t in targets if t != result.target]
        result.kinds.append("target")
        info = classify_target(result.target)
        result.needs_authorization = target_requires_authorization(info)

    if result.parsed_scan and not result.tools:
        src = result.parsed_scan.get("source") or ""
        if src.startswith("nmap"):
            result.tools = ["nmap"]

    result.kinds = list(dict.fromkeys(result.kinds))
    result.ready = bool(result.target or result.scan_file or result.parsed_scan)
    bits = []
    if result.target:
        bits.append(f"target {result.target}")
    if result.extra_targets:
        bits.append(f"+{len(result.extra_targets)} more host(s)")
    if result.scan_file:
        bits.append(f"scan {Path(result.scan_file).name}")
    if result.cves:
        bits.append(f"{len(result.cves)} CVE(s)")
    if result.model_repo:
        bits.append(f"model {result.model_repo}")
    if result.tools:
        bits.append("tools " + ", ".join(result.tools))
    if result.needs_authorization:
        bits.append("public target — authorization required")
    result.summary = "Detected: " + "; ".join(bits) if bits else "Nothing I can run yet. Paste an IP, hostname, URL, or drop a scan file."
    return result


def apply_probe_to_config(config: AssessmentConfig, probe: ProbeResult) -> AssessmentConfig:
    if probe.target and (not config.target or config.target == "unspecified"):
        config.target = probe.target
    if probe.scan_file and not config.scan_file:
        config.scan_file = probe.scan_file
    if probe.tools and not config.tool_stack:
        config.tool_stack = probe.tools
    if probe.hf_token and not config.hf_token:
        config.hf_token = probe.hf_token
    if probe.model_repo and config.model_name in {"microsoft/DialoGPT-medium", ""}:
        config.model_name = probe.model_repo
    if probe.remote_url and not config.remote_url:
        config.remote_url = probe.remote_url
        config.use_remote = True
    if probe.operator and not config.operator:
        config.operator = probe.operator
    if probe.cves and not config.notes:
        config.notes = "CVEs mentioned in input: " + ", ".join(probe.cves)
    if probe.parsed_scan and not config.tool_stack:
        config.tool_stack = ["imported-scan"]
    return config


# ---------------------------------------------------------------------------
# Risk, compliance, enrichment
# ---------------------------------------------------------------------------

_SEV_POINTS = {
    "critical": 40.0,
    "high": 22.0,
    "medium": 10.0,
    "low": 4.0,
    "informational": 0.0,
}

_CWE_HINTS = {
    "CVE-2021-44228": "CWE-502",
    "CVE-2022-22965": "CWE-94",
    "CVE-2021-41773": "CWE-22",
    "CVE-2014-0160": "CWE-201",
    "CVE-2017-5638": "CWE-20",
}

_KNOWN_CVSS = {
    "CVE-2021-44228": 10.0,
    "CVE-2022-22965": 9.8,
    "CVE-2021-41773": 7.5,
    "CVE-2014-0160": 7.5,
}


def calculate_risk_score(results: Dict[str, Any]) -> RiskScore:
    findings = results.get("findings") or []
    if not findings:
        return RiskScore(score=0.0, label="Informational", breakdown={"findings": 0, "points": 0.0})

    points = 0.0
    counts: Dict[str, int] = {}
    for finding in findings:
        sev = normalize_severity(finding.get("severity"))
        counts[sev] = counts.get(sev, 0) + 1
        points += _SEV_POINTS.get(sev, 4.0)
        cvss = finding.get("cvss")
        if cvss is None and finding.get("cve"):
            cvss = _KNOWN_CVSS.get(str(finding["cve"]).upper())
        if cvss is not None:
            try:
                points += float(cvss) * 3.0
            except (TypeError, ValueError):
                pass

    score = max(0.0, min(100.0, points))
    if score < 10:
        label = "Informational"
    elif score < 40:
        label = "Low"
    elif score < 70:
        label = "Medium"
    elif score < 85:
        label = "High"
    else:
        label = "Critical"
    return RiskScore(score=round(score, 1), label=label, breakdown={"counts": counts, "points": round(points, 1)})


def map_compliance(results: Dict[str, Any]) -> Dict[str, Any]:
    """Indicative control mapping. Not a certification or audit opinion."""
    findings = results.get("findings") or []
    severities = {normalize_severity(f.get("severity")) for f in findings}
    has_vuln = any(f.get("kind") == "vulnerability" or f.get("cve") for f in findings)
    has_high = bool(severities & {"high", "critical"})
    open_mgmt = any(f.get("port") in {21, 23, 3389, 5900} and f.get("kind") == "open_port" for f in findings)

    def _status(gap: bool, gaps: List[str]) -> Dict[str, Any]:
        return {
            "status": "Gap" if gap else "Indicative — no mapped gaps in this sample",
            "gaps": gaps if gap else [],
            "certified": False,
        }

    pci_gaps = []
    iso_gaps = []
    nist_gaps = []
    cis_gaps = []
    if has_high or has_vuln:
        pci_gaps.extend(["6.2 — unpatched critical/high software", "11.3 — unresolved vulnerability findings"])
        iso_gaps.append("A.8.8 — technical vulnerabilities not addressed")
        nist_gaps.append("PR.IP-12 / ID.RA — vulnerability management")
        cis_gaps.append("CIS 7 — continuous vulnerability management")
    if open_mgmt:
        pci_gaps.append("1.3 — insecure remote-admin exposure")
        cis_gaps.append("CIS 4 — secure administration")
        nist_gaps.append("PR.AC — access control surface")

    return {
        "disclaimer": "Indicative mapping from observed findings only. Not an audit or certification.",
        "pci_dss": _status(bool(pci_gaps), pci_gaps),
        "iso_27001": _status(bool(iso_gaps), iso_gaps),
        "nist_csf": _status(bool(nist_gaps), nist_gaps),
        "cis_controls": _status(bool(cis_gaps), cis_gaps),
        "gdpr": {
            "status": "Review required" if has_high else "Not assessed — no personal-data context supplied",
            "gaps": ["Article 32 — security of processing"] if has_high else [],
            "certified": False,
        },
        "sox": {"status": "Not applicable unless in-scope financial systems", "gaps": [], "certified": False},
    }


def attach_cwe(findings: List[Dict[str, Any]]) -> None:
    for finding in findings:
        cve = (finding.get("cve") or "").upper()
        if cve and "cwe" not in finding:
            hint = _CWE_HINTS.get(cve)
            if hint:
                finding["cwe"] = hint


def enrich_cves(findings: List[Dict[str, Any]], timeout: int = 15, cache_dir: Optional[Path] = None) -> None:
    """Best-effort public CVE metadata (CIRCL). Failures are recorded, not fatal."""
    try:
        import requests
    except ImportError:
        logger.warning("requests not installed; skipping CVE enrichment")
        return

    cache: Dict[str, Any] = {}
    cache_path = None
    if cache_dir:
        cache_dir.mkdir(parents=True, exist_ok=True)
        cache_path = cache_dir / "cve-cache.json"
        if cache_path.exists():
            try:
                cache = json.loads(cache_path.read_text(encoding="utf-8"))
            except json.JSONDecodeError:
                cache = {}

    for finding in findings:
        cve = (finding.get("cve") or "").upper()
        if not cve:
            continue
        if cve in cache:
            finding["cve_meta"] = cache[cve]
            if finding.get("cvss") is None:
                finding["cvss"] = cache[cve].get("cvss")
            continue
        url = f"https://cve.circl.lu/api/cve/{cve}"
        try:
            resp = requests.get(url, timeout=timeout)
            if resp.status_code != 200:
                finding.setdefault("enrichment_error", f"{cve} HTTP {resp.status_code}")
                continue
            body = resp.json()
            meta = {
                "id": cve,
                "summary": body.get("summary") or (body.get("containers", {}).get("cna", {}).get("title")),
                "cvss": _extract_cvss(body),
                "source": "cve.circl.lu",
            }
            cache[cve] = meta
            finding["cve_meta"] = meta
            if finding.get("cvss") is None:
                finding["cvss"] = meta.get("cvss")
        except Exception as exc:  # noqa: BLE001
            finding.setdefault("enrichment_error", str(exc))

    if cache_path is not None:
        cache_path.write_text(json.dumps(cache, indent=2), encoding="utf-8")


def _extract_cvss(body: Dict[str, Any]) -> Optional[float]:
    for key in ("cvss", "cvss3"):
        val = body.get(key)
        if isinstance(val, (int, float)):
            return float(val)
        if isinstance(val, dict) and val.get("score") is not None:
            try:
                return float(val["score"])
            except (TypeError, ValueError):
                pass
    return None


# ---------------------------------------------------------------------------
# JSON extraction from model output
# ---------------------------------------------------------------------------


def extract_json_object(text: str) -> Dict[str, Any]:
    if not text or not str(text).strip():
        raise json.JSONDecodeError("empty", text or "", 0)
    blob = str(text).strip()
    fence = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", blob, flags=re.DOTALL | re.IGNORECASE)
    if fence:
        return json.loads(fence.group(1))
    try:
        parsed = json.loads(blob)
        if isinstance(parsed, dict):
            return parsed
    except json.JSONDecodeError:
        pass
    start = blob.find("{")
    end = blob.rfind("}")
    if start >= 0 and end > start:
        return json.loads(blob[start : end + 1])
    raise json.JSONDecodeError("no JSON object found", blob, 0)


# ---------------------------------------------------------------------------
# Model backends
# ---------------------------------------------------------------------------

MODEL_PRESETS = {
    "dialogpt": "microsoft/DialoGPT-medium",
    "instruct-small": "HuggingFaceTB/SmolLM2-360M-Instruct",
    "instruct-qwen": "Qwen/Qwen2.5-0.5B-Instruct",
    "tiny": "sshleifer/tiny-gpt2",
}


def detect_device() -> str:
    try:
        import torch
    except ImportError:
        return "cpu"
    if torch.cuda.is_available():
        return "cuda"
    mps = getattr(torch.backends, "mps", None)
    if mps is not None and mps.is_available():
        return "mps"
    return "cpu"


def build_remote_url(remote: str) -> str:
    remote = remote.strip()
    if remote.startswith(("http://", "https://")):
        return remote
    if remote.startswith("hf:"):
        remote = remote[3:]
    return f"https://router.huggingface.co/hf-inference/models/{remote}"


def parse_remote_response(payload: Any) -> str:
    if isinstance(payload, list) and payload:
        first = payload[0]
        if isinstance(first, dict):
            for key in ("generated_text", "summary_text", "translation_text"):
                if key in first:
                    return str(first[key])
            if "text" in first:
                return str(first["text"])
        return str(first)
    if isinstance(payload, dict):
        if "generated_text" in payload:
            return str(payload["generated_text"])
        choices = payload.get("choices")
        if isinstance(choices, list) and choices:
            choice = choices[0]
            if isinstance(choice, dict):
                if "text" in choice:
                    return str(choice["text"])
                message = choice.get("message")
                if isinstance(message, dict) and "content" in message:
                    return str(message["content"])
        if "error" in payload:
            raise ModelError(str(payload["error"]))
    return str(payload)


class AIAnalyzer:
    def __init__(self, config: AssessmentConfig):
        self.config = config
        self.backend = None
        self.pipeline = None
        self.tokenizer = None
        self.device = "cpu"
        if config.use_remote and config.remote_url:
            self.backend = "remote"
            logger.info("Remote model endpoint: %s", config.remote_url)
            return
        deps = probe_dependencies(("transformers", "torch"))
        if not deps["transformers"].available:
            raise DependencyError(install_hint())
        if not deps["torch"].available:
            raise DependencyError(
                "PyTorch is required to load a local model.\n"
                "Install it in the same venv as transformers, or pass --remote-model."
            )
        self._load_local()

    def _load_local(self) -> None:
        try:
            import torch
            from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
        except ImportError as exc:
            raise DependencyError(install_hint()) from exc

        model_name = self.config.model_name
        token = resolve_hf_token(self.config.hf_token)
        cache_dir = self.config.cache_dir
        self.device = detect_device()
        logger.info("Loading local model %s on %s", model_name, self.device)
        try:
            kwargs: Dict[str, Any] = {}
            if token:
                kwargs["token"] = token
            if cache_dir:
                kwargs["cache_dir"] = cache_dir
            tokenizer = AutoTokenizer.from_pretrained(model_name, **kwargs)
            dtype = torch.float32
            if self.device in {"cuda", "mps"}:
                dtype = torch.float16
            model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=dtype, **kwargs)
            if tokenizer.pad_token is None:
                tokenizer.pad_token = tokenizer.eos_token or tokenizer.unk_token
            device_arg: Any
            if self.device == "cuda":
                device_arg = 0
            elif self.device == "mps":
                device_arg = "mps"
            else:
                device_arg = -1
            self.pipeline = pipeline(
                "text-generation",
                model=model,
                tokenizer=tokenizer,
                device=device_arg,
            )
            self.tokenizer = tokenizer
            self.backend = "local"
            logger.info("Local model ready")
        except OSError as exc:
            # Typical: config.json / model.safetensors missing, bad cache, offline hub
            raise ModelError(
                f"Model files were not found or could not be opened for {model_name!r}: {exc}\n"
                "Common fixes:\n"
                "  • Check the repo id (huggingface.co/<org>/<model>)\n"
                f"  • huggingface-cli download {model_name}\n"
                "  • Pass --cache-dir to a writable folder\n"
                "  • Use --remote-model hf:<org>/<model> with HF_TOKEN\n"
                "  • huggingface-cli login if the repo is gated"
            ) from exc
        except Exception as exc:  # noqa: BLE001
            raise ModelError(f"Failed to load local model {model_name!r}: {exc}") from exc

    def generate(self, prompt: str) -> str:
        if self.backend == "remote":
            return self._call_remote(prompt)
        if self.backend == "local" and self.pipeline is not None:
            return self._call_local(prompt)
        raise ModelError("No model backend is loaded. Pass --model or --remote-model.")

    def _call_local(self, prompt: str) -> str:
        try:
            pad_id = None
            if self.tokenizer is not None:
                pad_id = self.tokenizer.eos_token_id or self.tokenizer.pad_token_id
            raw = self.pipeline(
                prompt,
                max_new_tokens=self.config.max_new_tokens,
                do_sample=self.config.temperature > 0,
                temperature=max(self.config.temperature, 0.01),
                num_return_sequences=1,
                pad_token_id=pad_id,
                return_full_text=False,
            )
            text = raw[0].get("generated_text", "")
            return str(text).replace(prompt, "").strip()
        except Exception as exc:  # noqa: BLE001
            raise ModelError(f"Local generation failed: {exc}") from exc

    def _call_remote(self, prompt: str) -> str:
        try:
            import requests
        except ImportError as exc:
            raise DependencyError("requests is required for --remote-model") from exc

        url = build_remote_url(self.config.remote_url or "")
        token = resolve_hf_token(self.config.hf_token)
        headers = {"Content-Type": "application/json"}
        if token:
            headers["Authorization"] = f"Bearer {token}"
        payload = {
            "inputs": prompt,
            "parameters": {
                "max_new_tokens": self.config.max_new_tokens,
                "temperature": self.config.temperature,
                "return_full_text": False,
            },
        }
        last_error = "remote call failed"
        for attempt in range(1, 4):
            try:
                resp = requests.post(url, json=payload, headers=headers, timeout=self.config.timeout)
                if resp.status_code in {429, 503, 529}:
                    time.sleep(1.5 * attempt)
                    last_error = f"HTTP {resp.status_code}: {resp.text[:300]}"
                    continue
                if resp.status_code >= 400:
                    raise ModelError(f"Remote API error {resp.status_code}: {resp.text[:500]}")
                return parse_remote_response(resp.json()).strip()
            except ModelError:
                raise
            except Exception as exc:  # noqa: BLE001
                last_error = str(exc)
                time.sleep(1.0 * attempt)
        raise ModelError(f"Remote model failed after retries: {last_error}")

    def analyze_parameters(self) -> Dict[str, Any]:
        info = classify_target(self.config.target)
        prompt = (
            "You are a defensive security reviewer. Reply with JSON only.\n"
            f"Target: {self.config.target} ({info.kind})\n"
            f"Tools in the engagement plan: {', '.join(self.config.tool_stack) or 'unspecified'}\n"
            f"Parameters: {json.dumps(self.config.parameters)}\n"
            f"High-impact testing requested: {self.config.intrusive_mode}\n"
            "Keys: risk_level, impact, precautions, legal, scope_notes.\n"
        )
        raw = self.generate(prompt)
        try:
            parsed = extract_json_object(raw)
        except json.JSONDecodeError:
            return {"engine": self.backend, "model_raw": raw}
        parsed["engine"] = self.backend
        parsed["model_raw"] = raw
        return parsed

    def enhance_results(self, raw_results: Dict[str, Any]) -> Dict[str, Any]:
        prompt = (
            "You are a defensive security analyst. Reply with JSON only.\n"
            "Keys: summary, vulnerabilities (list of {title,cve,severity}), "
            "recommendations (list of strings), residual_risk.\n"
            f"SCAN:\n{json.dumps(raw_results, indent=2)[:8000]}\n"
        )
        raw = self.generate(prompt)
        try:
            analysis = extract_json_object(raw)
        except json.JSONDecodeError:
            analysis = {"summary": raw, "model_raw": raw}
        analysis["engine"] = self.backend
        analysis["model_raw"] = raw
        return analysis


# ---------------------------------------------------------------------------
# Evidence collection + reports
# ---------------------------------------------------------------------------


def run_nmap(config: AssessmentConfig) -> Dict[str, Any]:
    """Run a real nmap service scan when the binary is installed."""
    import shutil
    import subprocess

    nmap = shutil.which("nmap")
    if not nmap:
        raise AssessmentError(
            "nmap is listed in --tools but is not installed on PATH. "
            "Install nmap, or pass an existing --scan-file."
        )
    extra = []
    if isinstance(config.parameters.get("nmap"), str):
        extra = config.parameters["nmap"].split()
    elif isinstance(config.parameters.get("nmap"), list):
        extra = [str(x) for x in config.parameters["nmap"]]
    cmd = [nmap, "-sV", "-oX", "-"]
    if extra:
        cmd = [nmap, *extra]
        if "-oX" not in cmd:
            cmd.extend(["-oX", "-"])
    cmd.append(config.target)
    logger.info("Running %s", " ".join(cmd))
    try:
        proc = subprocess.run(cmd, capture_output=True, text=True, check=False, timeout=config.timeout)
    except subprocess.TimeoutExpired as exc:
        raise AssessmentError(f"nmap timed out after {config.timeout}s") from exc
    if proc.returncode != 0:
        raise AssessmentError(f"nmap exited {proc.returncode}: {proc.stderr.strip() or proc.stdout[:500]}")
    parsed = parse_nmap_xml(proc.stdout)
    parsed["tools_used"] = ["nmap"]
    parsed["nmap_command"] = cmd
    return parsed


def engagement_context(config: AssessmentConfig) -> Dict[str, Any]:
    """Evidence bag for the model when no scan has been produced yet."""
    return {
        "target": config.target,
        "tools_used": config.tool_stack,
        "parameters": config.parameters,
        "findings": [],
        "status": "awaiting-scan",
        "source": "engagement-config",
        "notes": config.notes or "No scan file or nmap run. Model is analyzing the engagement configuration only.",
    }


def write_reports(enhanced: Dict[str, Any], output: Path, formats: Optional[Sequence[str]] = None) -> Dict[str, Path]:
    formats = list(formats or ["json", "markdown", "sarif"])
    if output.suffix.lower() in {".json", ".md", ".markdown", ".sarif", ".html"}:
        base = output.with_suffix("")
    else:
        base = output
    base.parent.mkdir(parents=True, exist_ok=True)
    paths: Dict[str, Path] = {}

    if "json" in formats:
        path = output if output.suffix.lower() == ".json" else base.with_suffix(".json")
        path.write_text(json.dumps(enhanced, indent=2, default=str), encoding="utf-8")
        paths["json"] = path
    if "markdown" in formats or "md" in formats:
        path = base.with_suffix(".md")
        path.write_text(render_markdown(enhanced), encoding="utf-8")
        paths["markdown"] = path
    if "sarif" in formats:
        path = base.with_suffix(".sarif")
        path.write_text(json.dumps(render_sarif(enhanced), indent=2), encoding="utf-8")
        paths["sarif"] = path
    if "html" in formats:
        path = base.with_suffix(".html")
        path.write_text(render_html(enhanced), encoding="utf-8")
        paths["html"] = path
    return paths


def render_markdown(enhanced: Dict[str, Any]) -> str:
    raw = enhanced.get("raw_results") or {}
    risk = enhanced.get("risk_score") or {}
    analysis = enhanced.get("ai_analysis") or {}
    findings = raw.get("findings") or []
    lines = [
        f"# Security assessment analysis — {raw.get('target', enhanced.get('meta', {}).get('target', 'unknown'))}",
        "",
        f"- Generated: {enhanced.get('meta', {}).get('generated_at', '')}",
        f"- Mode: {enhanced.get('meta', {}).get('mode', '')}",
        f"- Engine: {enhanced.get('meta', {}).get('engine', analysis.get('engine', ''))}",
        f"- Risk: {risk.get('label', '?')} ({risk.get('score', '?')}/100)",
        "",
        "## Summary",
        "",
        str(analysis.get("summary") or "No summary."),
        "",
        "## Findings",
        "",
    ]
    if not findings:
        lines.append("_No findings in this sample._")
    for finding in findings:
        title = finding.get("title") or finding.get("cve") or finding.get("id")
        sev = str(finding.get("severity") or "info").upper()
        extra = []
        if finding.get("cve"):
            extra.append(finding["cve"])
        if finding.get("port"):
            extra.append(f"port {finding['port']}")
        suffix = f" ({', '.join(str(x) for x in extra)})" if extra else ""
        lines.append(f"- **{sev}** — {title}{suffix}")
    lines.extend(["", "## Recommendations", ""])
    for rec in analysis.get("recommendations") or []:
        lines.append(f"- {rec}")
    if analysis.get("attack_paths"):
        lines.extend(["", "## Plausible paths (analytic, not exploit steps)", ""])
        for path in analysis["attack_paths"]:
            lines.append(f"- {path}")
    compliance = enhanced.get("compliance") or {}
    if compliance:
        lines.extend(["", "## Indicative control mapping", ""])
        if compliance.get("disclaimer"):
            lines.append(f"_{compliance['disclaimer']}_")
            lines.append("")
        for key, body in compliance.items():
            if key == "disclaimer" or not isinstance(body, dict):
                continue
            lines.append(f"- **{key}**: {body.get('status')}")
            for gap in body.get("gaps") or []:
                lines.append(f"  - {gap}")
    lines.extend(["", "---", "This report is model analysis of supplied scan evidence or engagement configuration."])
    return "\n".join(lines) + "\n"


def render_sarif(enhanced: Dict[str, Any]) -> Dict[str, Any]:
    raw = enhanced.get("raw_results") or {}
    target = str(raw.get("target") or "unknown")
    results = []
    for finding in raw.get("findings") or []:
        level = {
            "critical": "error",
            "high": "error",
            "medium": "warning",
            "low": "note",
            "informational": "note",
        }.get(normalize_severity(finding.get("severity")), "note")
        message = finding.get("title") or finding.get("cve") or "finding"
        results.append(
            {
                "ruleId": finding.get("cve") or finding.get("id") or "finding",
                "level": level,
                "message": {"text": str(message)},
                "locations": [
                    {
                        "physicalLocation": {
                            "artifactLocation": {"uri": f"assessment://{target}"},
                            "region": {"startLine": 1},
                        }
                    }
                ],
                "properties": {k: finding.get(k) for k in ("port", "severity", "kind", "cve") if finding.get(k) is not None},
            }
        )
    return {
        "version": "2.1.0",
        "$schema": "https://json.schemastore.org/sarif-2.1.0.json",
        "runs": [
            {
                "tool": {
                    "driver": {
                        "name": "hf-authorized-assessment-analyzer",
                        "informationUri": "https://huggingface.co",
                        "version": "2.0.0",
                    }
                },
                "results": results,
            }
        ],
    }


def render_html(enhanced: Dict[str, Any]) -> str:
    md = render_markdown(enhanced)
    escaped = (
        md.replace("&", "&amp;")
        .replace("<", "&lt;")
        .replace(">", "&gt;")
    )
    risk = (enhanced.get("risk_score") or {}).get("label", "")
    color = {
        "Critical": "#8b0000",
        "High": "#c0392b",
        "Medium": "#d35400",
        "Low": "#2980b9",
        "Informational": "#2c3e50",
    }.get(risk, "#2c3e50")
    return f"""<!DOCTYPE html>
<html lang="en"><head><meta charset="utf-8"/>
<title>Assessment analysis</title>
<style>
 body {{ font-family: ui-sans-serif, system-ui, sans-serif; margin: 2rem; background:#0f1419; color:#e6edf3; }}
 pre {{ white-space: pre-wrap; line-height: 1.4; }}
 .badge {{ display:inline-block; padding:.2rem .6rem; background:{color}; border-radius:999px; }}
</style></head>
<body>
<p class="badge">{risk or 'Report'}</p>
<pre>{escaped}</pre>
</body></html>
"""


def diff_scans(current: Dict[str, Any], previous: Dict[str, Any]) -> Dict[str, Any]:
    def key(item: Dict[str, Any]) -> str:
        return str(item.get("cve") or item.get("title") or item.get("id"))

    now = {key(f): f for f in current.get("findings") or []}
    old = {key(f): f for f in previous.get("findings") or []}
    return {
        "added": [now[k] for k in now.keys() - old.keys()],
        "removed": [old[k] for k in old.keys() - now.keys()],
        "unchanged": len(now.keys() & old.keys()),
    }


def collect_raw_results(config: AssessmentConfig) -> Dict[str, Any]:
    if config.scan_file:
        return load_scan_file(Path(config.scan_file))
    if config.run_tools and "nmap" in [t.lower() for t in config.tool_stack]:
        return run_nmap(config)
    if config.run_tools:
        missing = [t for t in config.tool_stack if t.lower() != "nmap"]
        raise AssessmentError(
            "No executable collector for: "
            + ", ".join(missing)
            + ". Pass --scan-file with that tool's output, or include nmap with --run-tools."
        )
    return engagement_context(config)


def run_assessment(config: AssessmentConfig) -> Dict[str, Any]:
    require_authorization(config)
    info = classify_target(config.target)
    raw = collect_raw_results(config)
    raw.setdefault("target", config.target)
    analyzer = AIAnalyzer(config)
    pre = analyzer.analyze_parameters()
    attach_cwe(raw.get("findings") or [])
    if config.enrich_cves and not config.offline:
        cache = Path(config.cache_dir) if config.cache_dir else SCRIPT_DIR / ".hf-assess-cache"
        enrich_cves(raw.get("findings") or [], timeout=config.timeout, cache_dir=cache)
    analysis = analyzer.enhance_results(raw)
    risk = calculate_risk_score(raw)
    compliance = map_compliance(raw)
    previous_diff = None
    if config.diff_scan:
        previous = load_scan_file(Path(config.diff_scan))
        previous_diff = diff_scans(raw, previous)

    enhanced = {
        "meta": {
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "target": raw.get("target"),
            "target_class": asdict(info),
            "mode": "scan-file" if config.scan_file else ("nmap" if raw.get("source") == "nmap-xml" else "model"),
            "engine": analyzer.backend,
            "operator": config.operator,
            "engagement_id": config.engagement_id,
            "authorized": bool(config.authorized),
            "tools": config.tool_stack,
            "model": config.model_name if analyzer.backend == "local" else config.remote_url,
            "device": analyzer.device if analyzer.backend == "local" else None,
        },
        "pre_analysis": pre,
        "raw_results": raw,
        "ai_analysis": analysis,
        "risk_score": asdict(risk),
        "compliance": compliance,
        "diff": previous_diff,
    }
    if config.output_file:
        write_reports(enhanced, Path(config.output_file), config.formats)
        logger.info("Reports written next to %s", config.output_file)
    return enhanced


def intrusive_hack(config: AssessmentConfig) -> Dict[str, Any]:
    """Original entry point. Requires a loaded Hugging Face model."""
    return run_assessment(config)


AIIntrusionAnalyzer = AIAnalyzer


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog=Path(__file__).name,
        description="Hugging Face model-backed security analysis. A real model is required.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s --check-deps
  %(prog)s --bootstrap
  %(prog)s --target 10.0.0.8 --tools nmap --model-preset instruct-small --output results.json
  %(prog)s --scan-file nmap.xml --target 10.0.0.8 --tools nmap --remote-model hf:HuggingFaceTB/SmolLM2-360M-Instruct
  %(prog)s --target 10.0.0.8 --tools nmap --run-tools --authorized --operator you@org --engagement-id ENG-1

A local transformers install or --remote-model is required. This tool does not invent findings.
""",
    )
    parser.add_argument(
        "inbox",
        nargs="*",
        help="Anything: IP, hostname, URL, scan file path, or pasted nmap/JSON/CVE text",
    )
    parser.add_argument("--target", help="Hostname, IP, or CIDR in scope (optional with --scan-file)")
    parser.add_argument("--tools", nargs="+", help="Tools in the engagement or that produced --scan-file")
    parser.add_argument("--probe", action="store_true", help="Classify inbox/input and exit")
    parser.add_argument("--model", default="microsoft/DialoGPT-medium", help="Local Hugging Face model id")
    parser.add_argument("--model-preset", choices=sorted(MODEL_PRESETS), help="Named local model preset")
    parser.add_argument("--remote-model", help="Remote URL or hf:<org>/<model>")
    parser.add_argument("--output", help="Output path (.json or prefix for sidecar reports)")
    parser.add_argument("--formats", default="json,markdown,sarif", help="Comma list: json,markdown,sarif,html")
    parser.add_argument("--non-intrusive", action="store_false", dest="intrusive", help="Mark engagement as non-intrusive")
    parser.add_argument("--params", help="JSON object of engagement parameters")
    parser.add_argument("--scan-file", help="Existing nmap XML or JSON findings to analyze")
    parser.add_argument("--run-tools", action="store_true", help="Run nmap if it is listed in --tools (must be installed)")
    parser.add_argument("--diff-scan", help="Previous scan file to diff against")
    parser.add_argument("--authorized", action="store_true", help="Attest you are authorized to analyze this target")
    parser.add_argument("--operator", help="Name or email of the person running the analysis")
    parser.add_argument("--engagement-id", help="Ticket / engagement identifier")
    parser.add_argument("--enrich-cves", action="store_true", help="Look up CVE metadata (needs network)")
    parser.add_argument("--offline", action="store_true", help="Do not make outbound HTTP calls")
    parser.add_argument("--cache-dir", help="Model / CVE cache directory")
    parser.add_argument("--hf-token", help="Hugging Face token (else HF_TOKEN / HUGGINGFACE_API_KEY)")
    parser.add_argument("--max-new-tokens", type=int, default=400)
    parser.add_argument("--temperature", type=float, default=0.3)
    parser.add_argument("--timeout", type=int, default=45)
    parser.add_argument("--notes", default="", help="Free-form engagement notes")
    parser.add_argument("--config", help="JSON config file; CLI flags override file values")
    parser.add_argument("--check-deps", action="store_true", help="Print dependency status and exit")
    parser.add_argument("--check-model", metavar="ID", help="Verify a Hugging Face model id is reachable, then exit")
    parser.add_argument("--list-presets", action="store_true", help="List model presets and exit")
    parser.add_argument("--bootstrap", action="store_true", help="Create .hf-assess-venv and install requirements")
    parser.add_argument("--json", action="store_true", dest="json_stdout", help="Print the JSON report to stdout")
    parser.add_argument("--quiet", action="store_true")
    parser.add_argument("--verbose", action="store_true")
    parser.set_defaults(intrusive=True)
    return parser


def configure_logging(verbose: bool, quiet: bool) -> None:
    if quiet:
        level = logging.ERROR
    elif verbose:
        level = logging.DEBUG
    else:
        level = logging.INFO
    logging.basicConfig(level=level, format="%(asctime)s %(levelname)s %(message)s")


def print_dep_report() -> int:
    report = probe_dependencies()
    print(f"Python: {sys.version.split()[0]}  ({sys.executable})")
    print(f"Device: {detect_device()}")
    width = max(len(k) for k in report) + 2
    for name, status in report.items():
        mark = "ok" if status.available else "MISSING"
        extra = status.version or status.error or ""
        print(f"  {name.ljust(width)} {mark:8} {extra}")
    if not report["transformers"].available:
        print()
        print(install_hint())
        return 2
    return 0


def bootstrap_venv() -> int:
    import subprocess
    import venv

    venv_dir = DEFAULT_VENV
    print(f"Creating venv at {venv_dir}")
    venv.EnvBuilder(with_pip=True).create(venv_dir)
    pip = venv_dir / "bin" / "pip"
    py = venv_dir / "bin" / "python"
    cmd = [str(pip), "install", "--upgrade", "pip"]
    subprocess.check_call(cmd)
    if DEFAULT_REQUIREMENTS.exists():
        subprocess.check_call([str(pip), "install", "-r", str(DEFAULT_REQUIREMENTS)])
    else:
        subprocess.check_call(
            [str(pip), "install", "transformers", "accelerate", "safetensors", "huggingface_hub", "requests"]
        )
    print(f"Done. Use: {py} {Path(__file__).name} --check-deps")
    return 0


def load_config_file(path: Path) -> Dict[str, Any]:
    text = path.read_text(encoding="utf-8")
    if path.suffix.lower() in {".yaml", ".yml"}:
        try:
            import yaml  # type: ignore
        except ImportError as exc:
            raise AssessmentError("PyYAML is required to read YAML config files") from exc
        data = yaml.safe_load(text)
    else:
        data = json.loads(text)
    if not isinstance(data, dict):
        raise AssessmentError("config file must contain an object")
    return data


def apply_config_file(args: argparse.Namespace) -> argparse.Namespace:
    """Fill empty CLI fields from a JSON/YAML config object."""
    if not getattr(args, "config", None):
        return args
    data = load_config_file(Path(args.config))
    aliases = {
        "output_file": "output",
        "model_name": "model",
        "remote_url": "remote_model",
        "tool_stack": "tools",
        "json": "json_stdout",
    }
    for key, value in data.items():
        dest = aliases.get(key, key.replace("-", "_"))
        if not hasattr(args, dest):
            continue
        current = getattr(args, dest)
        defaultish = current in (None, False, "", [], 0)
        if dest in {"max_new_tokens", "temperature", "timeout"}:
            # Keep argparse defaults unless the file sets them and the user
            # did not pass a distinct override — file wins only when the
            # current value is the parser default.
            parser_defaults = {"max_new_tokens": 400, "temperature": 0.3, "timeout": 45}
            if current == parser_defaults.get(dest):
                setattr(args, dest, value)
            continue
        if defaultish and value not in (None, ""):
            setattr(args, dest, value)
    return args


def check_model_id(model_id: str, token: Optional[str] = None, cache_dir: Optional[str] = None) -> int:
    """Confirm a Hub repo exists so 'file not found' is diagnosed before load."""
    model_id = MODEL_PRESETS.get(model_id, model_id)
    deps = probe_dependencies(("huggingface_hub",))
    if not deps["huggingface_hub"].available:
        print(install_hint())
        return 2
    try:
        from huggingface_hub import model_info
    except ImportError:
        print("huggingface_hub is not installed")
        return 2
    try:
        info = model_info(model_id, token=resolve_hf_token(token))
    except Exception as exc:  # noqa: BLE001
        print(f"Model {model_id!r} was not found or is not reachable: {exc}")
        print("Fixes: check the id, log in (`huggingface-cli login`), or use --remote-model.")
        return 1
    siblings = [s.rfilename for s in (info.siblings or [])][:12]
    has_weights = any(
        name.endswith((".safetensors", ".bin", ".gguf")) or name == "config.json" for name in siblings
    )
    print(f"id:       {info.id}")
    print(f"sha:      {getattr(info, 'sha', None)}")
    print(f"pipeline: {getattr(info, 'pipeline_tag', None)}")
    print(f"private:  {getattr(info, 'private', None)}")
    print(f"files:    {', '.join(siblings) or '(none listed)'}")
    if cache_dir:
        print(f"cache:    {cache_dir}")
    if not has_weights and siblings:
        print("warning: no obvious weight/config files in the first page of the repo listing")
    return 0


def config_from_args(args: argparse.Namespace) -> AssessmentConfig:
    parameters: Dict[str, Any] = {}
    if args.params:
        try:
            parameters = json.loads(args.params)
        except json.JSONDecodeError as exc:
            raise AssessmentError(f"invalid JSON in --params: {exc}") from exc
        if not isinstance(parameters, dict):
            raise AssessmentError("--params must be a JSON object")

    model = MODEL_PRESETS.get(args.model_preset, args.model) if args.model_preset else args.model
    target = args.target or ""
    if not target and args.scan_file:
        target = Path(args.scan_file).stem
    tools = args.tools or []
    formats = [part.strip().lower() for part in (args.formats or "json").split(",") if part.strip()]

    return AssessmentConfig(
        target=target or "unspecified",
        tool_stack=tools,
        parameters=parameters,
        intrusive_mode=bool(args.intrusive),
        model_name=model,
        remote_url=args.remote_model,
        use_remote=bool(args.remote_model),
        output_file=args.output,
        authorized=bool(args.authorized),
        operator=args.operator,
        engagement_id=args.engagement_id,
        scan_file=args.scan_file,
        diff_scan=args.diff_scan,
        enrich_cves=bool(args.enrich_cves),
        offline=bool(args.offline),
        cache_dir=args.cache_dir,
        hf_token=args.hf_token,
        max_new_tokens=args.max_new_tokens,
        temperature=args.temperature,
        timeout=args.timeout,
        formats=formats,
        quiet=bool(args.quiet),
        verbose=bool(args.verbose),
        json_stdout=bool(args.json_stdout),
        notes=args.notes or "",
        run_tools=bool(getattr(args, "run_tools", False)),
    )


def print_summary(results: Dict[str, Any]) -> None:
    raw = results.get("raw_results") or {}
    risk = results.get("risk_score") or {}
    analysis = results.get("ai_analysis") or {}
    findings = raw.get("findings") or []
    vulns = [f for f in findings if f.get("cve") or f.get("kind") == "vulnerability"]
    print("\n=== ASSESSMENT ANALYSIS ===")
    print(f"Target:        {raw.get('target')}")
    print(f"Mode:          {results.get('meta', {}).get('mode')}")
    print(f"Engine:        {results.get('meta', {}).get('engine')}")
    print(f"Risk:          {risk.get('label')} ({risk.get('score')}/100)")
    print(f"Findings:      {len(findings)}  (vulnerability-class: {len(vulns)})")
    summary = str(analysis.get("summary") or analysis.get("model_raw") or "")
    if len(summary) > 400:
        summary = summary[:400] + "..."
    print(f"\nSummary:\n{summary}")
    recs = analysis.get("recommendations") or []
    if recs:
        print("\nRecommendations:")
        for rec in recs[:8]:
            print(f"  - {rec}")


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    if getattr(args, "config", None):
        args = apply_config_file(args)

    if args.list_presets:
        for name, model in MODEL_PRESETS.items():
            print(f"{name:16} {model}")
        return 0
    if args.bootstrap:
        return bootstrap_venv()
    if args.check_deps:
        return print_dep_report()
    if args.check_model:
        return check_model_id(args.check_model, token=args.hf_token, cache_dir=args.cache_dir)

    configure_logging(args.verbose, args.quiet)

    inbox_text = "\n".join(args.inbox) if args.inbox else ""
    probe = probe_input(inbox_text, persist_dir=SCRIPT_DIR / "hf-assess-results") if inbox_text else None
    if args.probe:
        payload = {"inbox": inbox_text, "probe": probe.as_dict() if probe else None}
        print(json.dumps(payload, indent=2, default=str))
        return 0 if probe and probe.ready else 1

    if probe:
        if not args.target and probe.target:
            args.target = probe.target
        if not args.scan_file and probe.scan_file:
            args.scan_file = probe.scan_file
        if not args.tools and probe.tools:
            args.tools = probe.tools
        if not args.hf_token and probe.hf_token:
            args.hf_token = probe.hf_token
        if not args.remote_model and probe.remote_url:
            args.remote_model = probe.remote_url
        if not args.operator and probe.operator:
            args.operator = probe.operator
        if probe.model_repo and args.model == "microsoft/DialoGPT-medium" and not args.model_preset:
            args.model = probe.model_repo
        if probe.needs_authorization and not args.authorized:
            logger.error("Public target %s needs --authorized", probe.target)
            return 3
        if probe.notes and not args.notes:
            args.notes = "; ".join(probe.notes)

    if not args.scan_file and not args.target:
        parser.error("paste or pass a target, file path, URL, or scan — nothing usable was detected")
    if not args.tools and not args.scan_file:
        args.tools = ["imported-scan"] if args.scan_file else ["nmap"]

    try:
        config = config_from_args(args)
        if probe:
            config = apply_probe_to_config(config, probe)
        results = run_assessment(config)
    except AuthorizationError as exc:
        logger.error("%s", exc)
        return 3
    except (AssessmentError, OSError, json.JSONDecodeError) as exc:
        logger.error("%s", exc)
        return 1

    if args.json_stdout:
        print(json.dumps(results, indent=2, default=str))
    elif not args.quiet:
        print_summary(results)
    return 0


if __name__ == "__main__":
    sys.exit(main())
