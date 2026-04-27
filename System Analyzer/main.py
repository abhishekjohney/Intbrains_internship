import json
import os
import platform
import socket
import urllib.error
import urllib.request
from datetime import datetime, timedelta

import psutil
from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER
from reportlab.lib.pagesizes import letter
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import inch
from reportlab.platypus import Paragraph, SimpleDocTemplate, Spacer


TITLE_LOCAL = "LOCAL SYSTEM HEALTH REPORT"
SECTION_HEADINGS = [
    "System Overview",
    "CPU Status",
    "Summary",
    "RAM/Memory Status",
    "Storage Status",
    "Network Status",
    "AI-style Observations",
]
META_PREFIXES = ("Generated:",)
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


OLLAMA_TIMEOUT_SEC = get_env_int("OLLAMA_TIMEOUT_SEC", 180)


def format_bytes(value):
    units = ["B", "KB", "MB", "GB", "TB", "PB"]
    size = float(value)
    for unit in units:
        if size < 1024 or unit == units[-1]:
            return f"{size:.2f} {unit}"
        size /= 1024


def get_primary_disk_path():
    if os.name == "nt":
        return os.environ.get("SystemDrive", "C:") + "\\"
    return "/"


def collect_local_metrics():
    hostname = socket.gethostname()
    fqdn = socket.getfqdn()
    username = os.getlogin() if hasattr(os, "getlogin") else os.environ.get("USERNAME", "Unknown")
    system = platform.system()
    release = platform.release()
    version = platform.version()
    machine = platform.machine()
    processor = platform.processor() or "Unknown"

    cpu_physical = psutil.cpu_count(logical=False) or 0
    cpu_logical = psutil.cpu_count(logical=True) or 0
    cpu_percent = psutil.cpu_percent(interval=1)
    cpu_freq = psutil.cpu_freq()

    memory = psutil.virtual_memory()
    swap = psutil.swap_memory()

    disk_path = get_primary_disk_path()
    disk = psutil.disk_usage(disk_path)

    boot_time_dt = datetime.fromtimestamp(psutil.boot_time())
    uptime = datetime.now() - boot_time_dt
    net = psutil.net_io_counters()
    processes = len(psutil.pids())

    load_average = None
    try:
        load_average = os.getloadavg()
    except (AttributeError, OSError):
        pass

    risk_items = []
    if memory.percent >= 80:
        risk_items.append(f"memory at {memory.percent:.1f}%")
    if disk.percent >= 90:
        risk_items.append(f"storage at {disk.percent:.1f}%")
    if cpu_percent >= 85:
        risk_items.append(f"CPU at {cpu_percent:.1f}%")

    if risk_items:
        overall_status = "Attention needed"
        risk_text = ", ".join(risk_items)
    else:
        overall_status = "Healthy"
        risk_text = "no critical thresholds reached"

    if memory.percent >= 80:
        mem_obs = f"Memory usage is high at {memory.percent:.1f}%."
    else:
        mem_obs = f"Memory usage is normal at {memory.percent:.1f}%."

    if disk.percent >= 90:
        disk_obs = f"Storage usage is critical at {disk.percent:.1f}% on {disk_path}."
    else:
        disk_obs = f"Storage usage is healthy at {disk.percent:.1f}% on {disk_path}."

    if cpu_percent >= 85:
        cpu_obs = f"CPU usage is high at {cpu_percent:.1f}%."
    else:
        cpu_obs = f"CPU usage is normal at {cpu_percent:.1f}%."

    return {
        "title": TITLE_LOCAL,
        "generated": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "system_overview": {
            "hostname": hostname,
            "fqdn": fqdn,
            "user": username,
            "os": f"{system} {release}",
            "version": version,
            "architecture": machine,
            "processor": processor,
            "boot_time": boot_time_dt.strftime("%Y-%m-%d %H:%M:%S"),
            "uptime": str(timedelta(seconds=int(uptime.total_seconds()))),
            "running_processes": processes,
        },
        "cpu_status": {
            "physical_cores": cpu_physical,
            "logical_cores": cpu_logical,
            "current_usage_percent": round(cpu_percent, 1),
            "current_frequency_mhz": round(cpu_freq.current, 2) if cpu_freq else "Unknown",
            "min_frequency_mhz": round(cpu_freq.min, 2) if cpu_freq else "Unknown",
            "max_frequency_mhz": round(cpu_freq.max, 2) if cpu_freq else "Unknown",
            "load_average_1_5_15": load_average if load_average else "N/A",
        },
        "summary": {
            "overall_status": overall_status,
            "key_risks": risk_text,
            "uptime": str(timedelta(seconds=int(uptime.total_seconds()))),
            "active_processes": processes,
        },
        "memory_status": {
            "total": format_bytes(memory.total),
            "available": format_bytes(memory.available),
            "used": format_bytes(memory.used),
            "usage_percent": round(memory.percent, 1),
            "swap_total": format_bytes(swap.total),
            "swap_used": format_bytes(swap.used),
            "swap_usage_percent": round(swap.percent, 1),
        },
        "storage_status": {
            "drive": disk_path,
            "total": format_bytes(disk.total),
            "used": format_bytes(disk.used),
            "free": format_bytes(disk.free),
            "usage_percent": round(disk.percent, 1),
        },
        "network_status": {
            "bytes_sent": format_bytes(net.bytes_sent),
            "bytes_received": format_bytes(net.bytes_recv),
            "packets_sent": net.packets_sent,
            "packets_received": net.packets_recv,
        },
        "observations": [mem_obs, disk_obs, cpu_obs],
    }


def generate_report_with_llama(metrics):
    prompt = (
        "You are an expert system reliability analyst. "
        "Create a clean, standard, structured plain-text report (no markdown) with exactly these sections in this order: "
        "System Overview, CPU Status, Summary, RAM/Memory Status, Storage Status, Network Status, AI-style Observations. "
        "Use readable indentation with key: value pairs. Keep the report concise but complete. "
        "Do not invent values; if missing write Unknown. "
        "In AI-style Observations write exactly 3 bullet points for memory, storage, and CPU health.\n\n"
        f"Input metrics JSON:\n{json.dumps(metrics, ensure_ascii=True, indent=2)}"
    )

    payload = {
        "model": LLAMA_MODEL,
        "prompt": prompt,
        "stream": False,
        "options": {"temperature": 0.2, "num_predict": 900},
    }

    request = urllib.request.Request(
        OLLAMA_API_URL,
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )

    with urllib.request.urlopen(request, timeout=OLLAMA_TIMEOUT_SEC) as response:
        body = response.read().decode("utf-8", errors="replace")

    data = json.loads(body)
    text = (data.get("response") or "").strip()
    if not text:
        raise RuntimeError("LLaMA returned an empty report")

    return "\n".join([
        "=" * 60,
        TITLE_LOCAL,
        "=" * 60,
        f"Generated: {metrics['generated']}",
        "",
        text,
        "=" * 60,
    ])


def get_pdf_styles(styles):
    title_style = ParagraphStyle(
        "CustomTitle",
        parent=styles["Heading1"],
        fontSize=16,
        textColor=colors.HexColor("#1a1a1a"),
        spaceAfter=12,
        alignment=TA_CENTER,
        fontName="Helvetica-Bold",
    )

    heading_style = ParagraphStyle(
        "CustomHeading",
        parent=styles["Heading2"],
        fontSize=12,
        textColor=colors.HexColor("#2c3e50"),
        spaceAfter=8,
        spaceBefore=8,
        fontName="Helvetica-Bold",
    )

    normal_style = ParagraphStyle(
        "CustomNormal",
        parent=styles["Normal"],
        fontSize=10,
        leading=12,
        fontName="Helvetica",
    )

    return title_style, heading_style, normal_style


def save_report_as_pdf(report, filename):
    doc = SimpleDocTemplate(
        filename,
        pagesize=letter,
        topMargin=0.5 * inch,
        bottomMargin=0.5 * inch,
    )
    story = []
    styles = getSampleStyleSheet()
    title_style, heading_style, normal_style = get_pdf_styles(styles)

    for line in report.split("\n"):
        if line.startswith("="):
            continue

        if line.startswith(TITLE_LOCAL):
            story.append(Paragraph(TITLE_LOCAL, title_style))
            story.append(Spacer(1, 0.2 * inch))
            continue

        if line.startswith(META_PREFIXES):
            story.append(Paragraph(line.strip(), normal_style))
            story.append(Spacer(1, 0.1 * inch))
            continue

        if any(line.startswith(heading) for heading in SECTION_HEADINGS):
            story.append(Paragraph(line.strip(), heading_style))
            continue

        if line.strip():
            story.append(Paragraph(line.strip(), normal_style))
        else:
            story.append(Spacer(1, 0.05 * inch))

    doc.build(story)


def build_report():
    metrics = collect_local_metrics()
    return generate_report_with_llama(metrics)


def main():
    try:
        report = build_report()
    except (urllib.error.URLError, TimeoutError, json.JSONDecodeError, OSError, RuntimeError) as exc:
        print(
            f"Error: LLaMA report generation failed: {exc}. "
            f"Make sure Ollama is running and model '{LLAMA_MODEL}' is available."
        )
        return

    print(report)

    filename = f"Local_System_Report_{platform.node().replace(' ', '_')}.pdf"
    save_report_as_pdf(report, filename)

    print(f"\nSaved report to: {filename}")


if __name__ == "__main__":
    main()
