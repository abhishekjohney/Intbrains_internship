from datetime import datetime, timedelta
import os
import platform
import socket

import psutil
from reportlab.lib.pagesizes import letter
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import inch
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle
from reportlab.lib import colors
from reportlab.lib.enums import TA_LEFT, TA_CENTER


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


def get_disk_report():
    disk_path = get_primary_disk_path()
    usage = psutil.disk_usage(disk_path)
    return disk_path, usage


def build_report():
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

    disk_path, disk = get_disk_report()

    boot_time = datetime.fromtimestamp(psutil.boot_time())
    uptime = datetime.now() - boot_time
    net = psutil.net_io_counters()
    processes = len(psutil.pids())

    load_average = None
    try:
        load_average = os.getloadavg()
    except (AttributeError, OSError):
        pass

    report_lines = [
        "=" * 60,
        "LOCAL SYSTEM HEALTH REPORT",
        "=" * 60,
        f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
        "",
        "System Overview",
        f"  Hostname: {hostname}",
        f"  FQDN: {fqdn}",
        f"  User: {username}",
        f"  OS: {system} {release}",
        f"  Version: {version}",
        f"  Architecture: {machine}",
        f"  Processor: {processor}",
        f"  Boot Time: {boot_time.strftime('%Y-%m-%d %H:%M:%S')}",
        f"  Uptime: {str(timedelta(seconds=int(uptime.total_seconds())))}",
        f"  Running Processes: {processes}",
        "",
        "CPU Status",
        f"  Physical Cores: {cpu_physical}",
        f"  Logical Cores: {cpu_logical}",
        f"  Current Usage: {cpu_percent:.1f}%",
    ]

    if cpu_freq:
        report_lines.extend([
            f"  Current Frequency: {cpu_freq.current:.2f} MHz",
            f"  Min Frequency: {cpu_freq.min:.2f} MHz",
            f"  Max Frequency: {cpu_freq.max:.2f} MHz",
        ])

    if load_average:
        report_lines.append(f"  Load Average (1, 5, 15 min): {load_average}")

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

    report_lines.extend([
        "",
        "Summary",
        f"  Overall Status: {overall_status}",
        f"  Key Risks: {risk_text}",
        f"  Uptime: {str(timedelta(seconds=int(uptime.total_seconds())))}",
        f"  Active Processes: {processes}",
    ])

    report_lines.extend([
        "",
        "RAM/Memory Status",
        f"  Total: {format_bytes(memory.total)}",
        f"  Available: {format_bytes(memory.available)}",
        f"  Used: {format_bytes(memory.used)}",
        f"  Usage: {memory.percent:.1f}%",
        f"  Swap Total: {format_bytes(swap.total)}",
        f"  Swap Used: {format_bytes(swap.used)}",
        f"  Swap Usage: {swap.percent:.1f}%",
        "",
        "Storage Status",
        f"  Drive: {disk_path}",
        f"  Total: {format_bytes(disk.total)}",
        f"  Used: {format_bytes(disk.used)}",
        f"  Free: {format_bytes(disk.free)}",
        f"  Usage: {disk.percent:.1f}%",
        "",
        "Network Status",
        f"  Bytes Sent: {format_bytes(net.bytes_sent)}",
        f"  Bytes Received: {format_bytes(net.bytes_recv)}",
        f"  Packets Sent: {net.packets_sent}",
        f"  Packets Received: {net.packets_recv}",
        "",
        "AI-style Observations",
    ])

    observations = []
    if memory.percent >= 80:
        observations.append(f"Memory usage is high at {memory.percent:.1f}%.")
    else:
        observations.append(f"Memory usage is normal at {memory.percent:.1f}%.")

    if disk.percent >= 90:
        observations.append(f"Storage usage is critical at {disk.percent:.1f}% on {disk_path}.")
    else:
        observations.append(f"Storage usage is healthy at {disk.percent:.1f}% on {disk_path}.")

    if cpu_percent >= 85:
        observations.append(f"CPU usage is high at {cpu_percent:.1f}%.")
    else:
        observations.append(f"CPU usage is normal at {cpu_percent:.1f}%.")

    for item in observations:
        report_lines.append(f"  - {item}")

    report_lines.append("=" * 60)
    return "\n".join(report_lines)


def main():
    report = build_report()
    print(report)

    filename = f"Local_System_Report_{platform.node().replace(' ', '_')}.pdf"
    
    # Create PDF document
    doc = SimpleDocTemplate(filename, pagesize=letter, topMargin=0.5*inch, bottomMargin=0.5*inch)
    story = []
    styles = getSampleStyleSheet()
    
    # Custom styles
    title_style = ParagraphStyle(
        'CustomTitle',
        parent=styles['Heading1'],
        fontSize=16,
        textColor=colors.HexColor('#1a1a1a'),
        spaceAfter=12,
        alignment=TA_CENTER,
        fontName='Helvetica-Bold'
    )
    
    heading_style = ParagraphStyle(
        'CustomHeading',
        parent=styles['Heading2'],
        fontSize=12,
        textColor=colors.HexColor('#2c3e50'),
        spaceAfter=8,
        spaceBefore=8,
        fontName='Helvetica-Bold'
    )
    
    normal_style = ParagraphStyle(
        'CustomNormal',
        parent=styles['Normal'],
        fontSize=10,
        leading=12,
        fontName='Helvetica'
    )
    
    # Parse report and build PDF content
    lines = report.split('\n')
    
    for line in lines:
        if '=' * 60 in line or line.startswith('='):
            continue
        elif line.startswith('LOCAL SYSTEM HEALTH REPORT'):
            story.append(Paragraph('LOCAL SYSTEM HEALTH REPORT', title_style))
            story.append(Spacer(1, 0.2*inch))
        elif line.startswith('Generated:'):
            story.append(Paragraph(line.strip(), normal_style))
            story.append(Spacer(1, 0.1*inch))
        elif any(line.startswith(heading) for heading in ['System Overview', 'CPU Status', 'Summary', 'RAM/Memory Status', 'Storage Status', 'Network Status', 'AI-style Observations']):
            story.append(Paragraph(line.strip(), heading_style))
        elif line.strip() and not line.startswith(' '):
            story.append(Spacer(1, 0.05*inch))
        elif line.strip():
            story.append(Paragraph(line.strip(), normal_style))
        else:
            story.append(Spacer(1, 0.05*inch))
    
    # Build the PDF
    doc.build(story)
    
    print(f"\nSaved report to: {filename}")


if __name__ == "__main__":
    main()
