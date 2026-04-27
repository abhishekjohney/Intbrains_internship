import getpass
import json
import os
import socket
import urllib.error
import urllib.request
from datetime import datetime

import paramiko
from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER
from reportlab.lib.pagesizes import letter
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import inch
from reportlab.platypus import Paragraph, SimpleDocTemplate, Spacer


TITLE_REMOTE = "REMOTE SSH SYSTEM REPORT"
SECTION_HEADINGS = [
    "System Overview",
    "CPU Status",
    "Summary",
    "RAM/Memory Status",
    "Storage Status",
    "Network Status",
    "Observations",
]
META_PREFIXES = ("Generated:", "Target Host:", "Username:")
OLLAMA_API_URL = os.environ.get("OLLAMA_API_URL", "http://127.0.0.1:11434/api/generate")
LLAMA_MODEL = os.environ.get("LLAMA_MODEL", "llama3:latest")


def get_env_int(name, default):
    value = os.environ.get(name)
    if value is None:
        return default
    try:
        parsed = int(value)
        return parsed if parsed > 0 else default
    except ValueError:
        return default


OLLAMA_TIMEOUT_SEC = get_env_int("OLLAMA_TIMEOUT_SEC", 300)


def run_command(client, command):
    _, stdout, stderr = client.exec_command(command)
    output = stdout.read().decode("utf-8", errors="replace").strip()
    error = stderr.read().decode("utf-8", errors="replace").strip()
    return output if output else error


def run_first_success(client, commands):
    for command in commands:
        output = run_command(client, command)
        if not output:
            continue
        lowered = output.lower()
        if "is not recognized as an internal or external command" in lowered:
            continue
        if "command not found" in lowered:
            continue
        return output
    return "Unknown"


def truncate_text(text, max_chars=1200):
    if text is None:
        return "Unknown"
    if len(text) <= max_chars:
        return text
    return text[:max_chars] + "\n... [truncated]"


def format_ollama_error(exc):
    if isinstance(exc, TimeoutError):
        return (
            f"Ollama timed out after {OLLAMA_TIMEOUT_SEC} seconds. "
            f"The model '{LLAMA_MODEL}' may be too slow or Ollama may be busy."
        )

    if isinstance(exc, urllib.error.HTTPError):
        return f"Ollama returned HTTP {exc.code}: {exc.reason}"

    if isinstance(exc, urllib.error.URLError):
        reason = exc.reason
        if isinstance(reason, TimeoutError) or isinstance(reason, socket.timeout):
            return (
                f"Ollama timed out after {OLLAMA_TIMEOUT_SEC} seconds. "
                f"The model '{LLAMA_MODEL}' may be too slow or Ollama may be busy."
            )
        if isinstance(reason, ConnectionRefusedError):
            return (
                f"Could not connect to Ollama at {OLLAMA_API_URL}. "
                "Start Ollama and try again."
            )
        return f"Ollama request failed: {reason}"

    if isinstance(exc, json.JSONDecodeError):
        return "Ollama returned invalid JSON while generating the report."

    if isinstance(exc, OSError):
        return f"Network or OS error while contacting Ollama: {exc}"

    return str(exc)


def collect_remote_metrics(client, host, username, os_target, port):
    if os_target == "1":
        command_set = {
            "os_info": ["uname -a", "hostnamectl"],
            "hostname": ["hostname"],
            "fqdn": ["hostname -f", "hostname"],
            "boot_time": ["uptime -s", "who -b"],
            "uptime": ["uptime -p", "uptime"],
            "processes": ["ps -e --no-headers | wc -l"],
            "cpu_usage": ["top -bn1 | head -n 5", "cat /proc/loadavg", "uptime"],
            "cpu_cores": ["nproc", "getconf _NPROCESSORS_ONLN"],
            "cpu_model": ["lscpu | grep 'Model name' | cut -d ':' -f2", "uname -p"],
            "memory": ["free -h", "cat /proc/meminfo"],
            "storage": ["df -h", "lsblk"],
            "network": ["ip -s link", "cat /proc/net/dev"],
        }
        target_os = "Linux / macOS"
    else:
        command_set = {
            "os_info": ['systeminfo | findstr /C:"OS Name" /C:"OS Version"'],
            "hostname": ["hostname"],
            "fqdn": ['powershell -NoProfile -Command "[System.Net.Dns]::GetHostEntry($env:COMPUTERNAME).HostName"'],
            "boot_time": ['powershell -NoProfile -Command "(Get-CimInstance Win32_OperatingSystem).LastBootUpTime"'],
            "uptime": ['powershell -NoProfile -Command "(Get-Date) - (Get-CimInstance Win32_OperatingSystem).LastBootUpTime"'],
            "processes": ['powershell -NoProfile -Command "(Get-Process).Count"'],
            "cpu_usage": [
                "wmic cpu get loadpercentage",
                'powershell -NoProfile -Command "(Get-CimInstance Win32_Processor | Measure-Object -Property LoadPercentage -Average).Average"',
                'powershell -NoProfile -Command "(Get-Counter \\\"\\\\Processor(_Total)\\\\% Processor Time\\\").CounterSamples.CookedValue"',
            ],
            "cpu_cores": [
                'powershell -NoProfile -Command "(Get-CimInstance Win32_Processor | Measure-Object -Property NumberOfLogicalProcessors -Sum).Sum"'
            ],
            "cpu_model": ['powershell -NoProfile -Command "(Get-CimInstance Win32_Processor | Select-Object -First 1 -ExpandProperty Name)"'],
            "memory": ["wmic OS get FreePhysicalMemory,TotalVisibleMemorySize /Value", 'powershell -NoProfile -Command "Get-CimInstance Win32_OperatingSystem | Select TotalVisibleMemorySize,FreePhysicalMemory"'],
            "storage": ["wmic logicaldisk get caption,freespace,size", 'powershell -NoProfile -Command "Get-PSDrive -PSProvider FileSystem | Select Name,Used,Free"'],
            "network": ['powershell -NoProfile -Command "Get-NetAdapterStatistics"'],
        }
        target_os = "Windows"

    raw = {
        key: truncate_text(run_first_success(client, commands))
        for key, commands in command_set.items()
    }

    return {
        "title": TITLE_REMOTE,
        "generated": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "target_host": f"{host}:{port}",
        "username": username,
        "target_os": target_os,
        "raw_outputs": raw,
    }


def generate_report_with_llama(metrics):
    prompt = (
        "You are an expert system reliability analyst. "
        "Create a clean, standard, structured plain-text report (no markdown) with exactly these sections in this order: "
        "System Overview, CPU Status, Summary, RAM/Memory Status, Storage Status, Network Status, Observations. "
        "Use readable indentation with key: value pairs. Keep the report concise but complete. "
        "Do not invent values; if missing write Unknown. "
        "In Observations write exactly 3 bullet points for memory, storage, and CPU health.\n\n"
        f"Input metrics JSON:\n{json.dumps(metrics, ensure_ascii=True, indent=2)}"
    )

    payload = {
        "model": LLAMA_MODEL,
        "prompt": prompt,
        "stream": False,
        "options": {"temperature": 0.2, "num_predict": 500},
    }

    request = urllib.request.Request(
        OLLAMA_API_URL,
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )

    try:
        with urllib.request.urlopen(request, timeout=OLLAMA_TIMEOUT_SEC) as response:
            body = response.read().decode("utf-8", errors="replace")
    except (TimeoutError, urllib.error.URLError, urllib.error.HTTPError, json.JSONDecodeError, OSError) as exc:
        raise RuntimeError(format_ollama_error(exc)) from exc

    try:
        data = json.loads(body)
    except json.JSONDecodeError as exc:
        raise RuntimeError(format_ollama_error(exc)) from exc

    text = (data.get("response") or "").strip()
    if not text:
        raise RuntimeError("Ollama returned an empty report.")

    text = text.replace("AI-style Observations", "Observations")

    return "\n".join([
        "=" * 60,
        TITLE_REMOTE,
        "=" * 60,
        f"Generated: {metrics['generated']}",
        f"Target Host: {metrics['target_host']}",
        f"Username: {metrics['username']}",
        "",
        text,
        "=" * 60,
    ])


def get_pdf_styles(styles):
    title_style = ParagraphStyle(
        "CustomTitle",
        parent=styles["Heading1"],
        fontSize=18,
        textColor=colors.HexColor("#0f172a"),
        spaceAfter=16,
        spaceBefore=4,
        alignment=TA_CENTER,
        fontName="Helvetica-Bold",
    )

    meta_style = ParagraphStyle(
        "CustomMeta",
        parent=styles["Normal"],
        fontSize=9,
        leading=12,
        textColor=colors.HexColor("#475569"),
        fontName="Helvetica",
    )

    heading_style = ParagraphStyle(
        "CustomHeading",
        parent=styles["Heading2"],
        fontSize=13,
        textColor=colors.HexColor("#1e3a8a"),
        spaceAfter=6,
        spaceBefore=10,
        fontName="Helvetica-Bold",
    )

    normal_style = ParagraphStyle(
        "CustomNormal",
        parent=styles["Normal"],
        fontSize=10,
        leading=14,
        fontName="Helvetica",
    )

    bullet_style = ParagraphStyle(
        "CustomBullet",
        parent=normal_style,
        leftIndent=14,
        bulletIndent=8,
        spaceAfter=2,
    )

    return title_style, meta_style, heading_style, normal_style, bullet_style


def save_report_as_pdf(report, filename):
    doc = SimpleDocTemplate(
        filename,
        pagesize=letter,
        topMargin=0.7 * inch,
        bottomMargin=0.7 * inch,
        leftMargin=0.75 * inch,
        rightMargin=0.75 * inch,
    )
    story = []
    styles = getSampleStyleSheet()
    title_style, meta_style, heading_style, normal_style, bullet_style = get_pdf_styles(styles)

    for line in report.split("\n"):
        if line.startswith("="):
            continue

        if line.startswith(TITLE_REMOTE):
            story.append(Paragraph(TITLE_REMOTE, title_style))
            story.append(Spacer(1, 0.2 * inch))
            continue

        if line.startswith(META_PREFIXES):
            story.append(Paragraph(line.strip(), meta_style))
            story.append(Spacer(1, 0.05 * inch))
            continue

        if any(line.startswith(heading) for heading in SECTION_HEADINGS):
            story.append(Paragraph(line.strip(), heading_style))
            continue

        if line.strip():
            stripped = line.strip()
            if stripped.startswith("-"):
                story.append(Paragraph(stripped, bullet_style))
            else:
                story.append(Paragraph(stripped, normal_style))
        else:
            story.append(Spacer(1, 0.05 * inch))

    doc.build(story)


def build_remote_report(host, username, os_target, port=22, password=None, status_callback=None):
    client = paramiko.SSHClient()
    client.set_missing_host_key_policy(paramiko.AutoAddPolicy())

    def notify(message):
        if status_callback:
            status_callback(message)

    try:
        client.connect(
            hostname=host,
            port=port,
            username=username,
            password=password,
            timeout=10,
            auth_timeout=10,
            banner_timeout=10,
        )
    except paramiko.AuthenticationException:
        return None, (
            f"Authentication failed for {username}@{host}:{port}. Check the username and password. "
            "For Windows targets, try username or COMPUTERNAME\\username."
        )
    except paramiko.SSHException as exc:
        return None, f"SSH connection failed during negotiation: {exc}"
    except OSError as exc:
        return None, f"Network connection failed while reaching {host}:{port}: {exc}"

    notify("Authentication successful. Starting remote data collection and report generation...")

    try:
        notify("Processing remote system data. Please wait...")
        metrics = collect_remote_metrics(client, host, username, os_target, port)
        notify("Generating report from collected data...")
        report = generate_report_with_llama(metrics)
        return report, None
    except (urllib.error.URLError, TimeoutError, urllib.error.HTTPError, json.JSONDecodeError, OSError, RuntimeError) as exc:
        return None, f"Report generation failed: {format_ollama_error(exc)}"
    except Exception as exc:
        return None, f"Unexpected error while building the report: {exc}"
    finally:
        client.close()


def main():
    print(f"=== {TITLE_REMOTE} ===")
    host = input("Enter Target IP Address or Hostname: ").strip()
    username = input("Enter SSH Username: ").strip()
    print("\nWhat is the Operating System of the Target Machine?")
    print("1. Linux / macOS")
    print("2. Windows")
    os_target = input("Select 1 or 2: ").strip()

    if os_target not in {"1", "2"}:
        print("Invalid OS selection. Please run again and choose 1 or 2.")
        return

    port_input = input("Enter SSH port (press Enter for 22): ").strip()
    if port_input:
        try:
            port = int(port_input)
        except ValueError:
            print("Invalid port. Please run again and enter a numeric value.")
            return
    else:
        port = 22

    password = getpass.getpass("Enter SSH Password (hidden): ")

    print("Connecting to the target host...")
    report, error = build_remote_report(
        host,
        username,
        os_target,
        port=port,
        password=password,
        status_callback=print,
    )
    if error:
        print(f"\nError: {error}")
        return

    print(report)

    safe_host = host.replace(":", "_").replace("/", "_").replace("\\", "_")
    filename = f"Remote_SSH_Report_{safe_host}.pdf"
    save_report_as_pdf(report, filename)

    print(f"\nSaved report to: {filename}")


if __name__ == "__main__":
    main()
