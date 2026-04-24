import getpass
import os
from datetime import datetime

import paramiko


def format_bytes(value):
    units = ["B", "KB", "MB", "GB", "TB"]
    size = float(value)
    for unit in units:
        if size < 1024 or unit == units[-1]:
            return f"{size:.2f} {unit}"
        size /= 1024


def run_command(client, command):
    stdin, stdout, stderr = client.exec_command(command)
    output = stdout.read().decode("utf-8", errors="replace").strip()
    error = stderr.read().decode("utf-8", errors="replace").strip()
    return output if output else error


def build_remote_report(host, username, os_target, port=22, password=None, key_filename=None, key_passphrase=None):
    client = paramiko.SSHClient()
    client.set_missing_host_key_policy(paramiko.AutoAddPolicy())

    try:
        client.connect(
            hostname=host,
            port=port,
            username=username,
            password=password,
            key_filename=key_filename,
            passphrase=key_passphrase,
            timeout=10,
            auth_timeout=10,
            banner_timeout=10,
        )
    except paramiko.AuthenticationException:
        return None, (
            "Authentication failed. Check the SSH username, password, or private key. "
            "For Windows targets, try formats like 'username', 'COMPUTERNAME\\username', or a domain account."
        )
    except paramiko.SSHException as exc:
        return None, f"SSH negotiation failed: {exc}"
    except OSError as exc:
        return None, f"Network connection failed: {exc}"

    try:
        if os_target == "1":
            os_info = run_command(client, "uname -a")
            cpu_info = run_command(client, "top -bn1 | head -n 5")
            memory_info = run_command(client, "free -h")
            disk_info = run_command(client, "df -h")
        else:
            os_info = run_command(client, "systeminfo | findstr /C:\"OS Name\" /C:\"OS Version\"")
            cpu_info = run_command(client, "wmic cpu get loadpercentage")
            memory_info = run_command(client, "wmic OS get FreePhysicalMemory,TotalVisibleMemorySize /Value")
            disk_info = run_command(client, "wmic logicaldisk get caption,freespace,size")

        report_lines = [
            "=" * 60,
            "REMOTE SSH SYSTEM REPORT",
            "=" * 60,
            f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
            f"Target Host: {host}:{port}",
            f"Username: {username}",
            "",
            "Summary",
            f"  SSH Connection: Successful",
            f"  Target OS: {'Linux / macOS' if os_target == '1' else 'Windows'}",
            "",
            "System Overview",
            f"  OS Info: {os_info or 'No output'}",
            "",
            "CPU Status",
            f"  {cpu_info or 'No output'}",
            "",
            "Memory Status",
            f"  {memory_info or 'No output'}",
            "",
            "Storage Status",
            f"  {disk_info or 'No output'}",
            "",
            "AI-style Observations",
            "  - Review memory and disk output for thresholds above 80% and 90% respectively.",
            "=" * 60,
        ]
        return "\n".join(report_lines), None
    finally:
        client.close()


def main():
    print("=== REMOTE SSH SYSTEM REPORT ===")
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
    port = int(port_input) if port_input else 22

    print("\nHow do you want to authenticate?")
    print("1. Password")
    print("2. Private key")
    auth_mode = input("Select 1 or 2: ").strip()

    password = None
    key_filename = None
    key_passphrase = None

    if auth_mode == "1":
        password = getpass.getpass("Enter SSH Password (hidden): ")
    elif auth_mode == "2":
        key_filename = input("Enter full path to private key file: ").strip().strip('"')
        key_passphrase = getpass.getpass("Enter key passphrase if any (press Enter if none): ")
        if not key_passphrase:
            key_passphrase = None
    else:
        print("Invalid authentication choice. Please run again and choose 1 or 2.")
        return

    report, error = build_remote_report(
        host,
        username,
        os_target,
        port=port,
        password=password,
        key_filename=key_filename,
        key_passphrase=key_passphrase,
    )
    if error:
        print(f"Error: {error}")
        return

    print(report)

    safe_host = host.replace(":", "_").replace("/", "_").replace("\\", "_")
    filename = f"Remote_SSH_Report_{safe_host}.txt"
    with open(filename, "w", encoding="utf-8") as file_handle:
        file_handle.write(report)

    print(f"\nSaved report to: {filename}")


if __name__ == "__main__":
    main()
