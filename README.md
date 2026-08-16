# HF Assess

Local Hugging Face model analysis for **authorized** security assessments.

Paste a host, URL, nmap dump, or scan file. The tool classifies the input, loads a real local (or remote) model, and writes JSON / Markdown / SARIF / HTML reports.

It does **not** invent findings. It does **not** run exploits. A Hugging Face model is required.

## Requirements

- Python 3.10+
- macOS or Linux
- Optional: `nmap` on PATH if you want live service scans
- Optional: `HF_TOKEN` for gated Hub models or remote inference

Homebrew Python blocks global `pip install`. Use the venv below.

## Install

```bash
git clone https://github.com/jdgiles26/hf-assess.git
cd hf-assess
./start
```

That creates `.hf-assess-venv`, installs `requirements.txt`, offers a local model if none is cached, and opens the GUI.

## Run

```bash
./start
./start 10.0.0.8
./start path/to/nmap.xml
```

If a small instruct model is already in `~/.cache/huggingface/hub`, the launcher skips the download prompt.

CLI:

```bash
source .hf-assess-venv/bin/activate

# classify anything
python hf_assess.py --probe 'Host: 10.0.0.8 Ports: 22/open/tcp//ssh///'

# analyze an existing scan
python hf_assess.py --scan-file nmap.xml --target 10.0.0.8 --tools nmap \
  --model-preset instruct-small --output report.json

# live nmap (lab / authorized targets only)
python hf_assess.py --target 10.0.0.8 --tools nmap --run-tools --authorized \
  --operator you --engagement-id ENG-1 --model-preset instruct-small
```

Public IPs, wide CIDRs (`0.0.0.0/0`), and public DNS names require `--authorized`. RFC1918, loopback, and names like `*.local` / `*.internal` do not.

## Models

| Preset | Hub id | Notes |
|---|---|---|
| `instruct-small` | `HuggingFaceTB/SmolLM2-360M-Instruct` | Default recommendation |
| `instruct-qwen` | `Qwen/Qwen2.5-0.5B-Instruct` | Slightly larger |
| `dialogpt` | `microsoft/DialoGPT-medium` | Original conversational default |
| `tiny` | `sshleifer/tiny-gpt2` | Load test only |

```bash
./hf-assess-gui.sh --list
./hf-assess-gui.sh --download
python hf_assess.py --check-deps
python hf_assess.py --check-model HuggingFaceTB/SmolLM2-360M-Instruct
```

Remote instead of local weights:

```bash
python hf_assess.py --target 10.0.0.8 --tools nmap \
  --remote-model hf:HuggingFaceTB/SmolLM2-360M-Instruct
```

## Privacy

- No machine username, git email, or home-directory path is baked into the source.
- Reports do not get an operator name unless you pass `--operator` or set `HF_ASSESS_OPERATOR`.
- `.hf-assess-venv/`, `hf-assess-results/`, and `.hf-assess-cache/` are gitignored. Do not commit scan output.

Example addresses in docs and tests (`10.0.0.8`, `192.168.1.0/24`, `8.8.8.8`, `203.0.113.0/24` style) are documentation ranges, not anyone’s host.

## Tests

```bash
python3 -m unittest discover -s tests -v
```

Live model tests skip unless `.hf-assess-venv` exists and a Hub model is already cached.

## Layout

```
start                one-command bootstrap + GUI
hf_assess.py         CLI analyzer
hf-assess-gui.py     desktop operator
hf-assess-gui.sh     extra launcher flags (--check, --probe, --download)
requirements.txt
tests/
```

## License

MIT
