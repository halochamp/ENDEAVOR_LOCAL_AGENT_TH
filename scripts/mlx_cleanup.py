# ENDEAVOR_LOCAL_AGENT_TH — © HaloChamp
# License: MIT License + Commons Clause — personal/educational use only, no commercial use without permission
# Website: https://www.poomwat.com | GitHub: https://github.com/halochamp | Email: champoomwat@gmail.com

"""mlx_cleanup.py — ตรวจ + ฆ่า mlx_lm.server ที่ไม่ต้องการ

ปัญหา: mlx_lm.server โหลด weights ลง Unified Memory ของ Metal
ถ้ารัน 2 server พร้อมกัน + แต่ละ server โหลดหลาย model → RAM ระเบิดง่าย ๆ
(48GB เครื่องของผมก็พังมาแล้ว)

วิธีใช้:
  python scripts/mlx_cleanup.py                  # list อย่างเดียว (ปลอดภัย, default)
  python scripts/mlx_cleanup.py --keep 8080      # ฆ่าทุก server ยกเว้น :8080
  python scripts/mlx_cleanup.py --keep-config    # ฆ่าทุก server ยกเว้นตัวที่ตรง config.MLX_BASE_URL
  python scripts/mlx_cleanup.py --kill 8888      # ฆ่า server บน :8888 อย่างเดียว
  python scripts/mlx_cleanup.py --kill-all       # ฆ่าทุก server (ใช้ตอนจะ restart ใหม่)

exit code:
  0 = สำเร็จ
  1 = ไม่เจอ server / port ผิด
"""
from __future__ import annotations
import argparse
import os
import re
import subprocess
import sys


def _list_servers() -> list[dict]:
    """list mlx_lm.server processes ที่กำลังรัน

    คืน: [{"pid": int, "model": str, "port": int, "rss_mb": float, "cmdline": str}, ...]
    """
    out = subprocess.run(
        ["ps", "-axo", "pid,rss,command"],
        capture_output=True, text=True, check=True,
    ).stdout

    servers = []
    for line in out.splitlines()[1:]:
        if "mlx_lm.server" not in line:
            continue
        parts = line.strip().split(None, 2)
        if len(parts) < 3:
            continue
        pid, rss_kb, cmd = parts
        port_m = re.search(r"--port\s+(\d+)", cmd)
        model_m = re.search(r"--model\s+(\S+)", cmd)
        servers.append({
            "pid": int(pid),
            "rss_mb": int(rss_kb) / 1024,
            "port": int(port_m.group(1)) if port_m else None,
            "model": model_m.group(1) if model_m else "?",
            "cmdline": cmd,
        })
    return servers


def _get_metal_memory_mb() -> float | None:
    """ดึง Metal/GPU memory ที่ใช้ทั้งระบบ (MB) — Apple Silicon เท่านั้น"""
    try:
        out = subprocess.run(
            ["vm_stat"],
            capture_output=True, text=True, check=True,
        ).stdout
        # page size บน Apple Silicon = 16384 bytes
        m = re.search(r"page size of (\d+) bytes", out)
        page_bytes = int(m.group(1)) if m else 16384
        m = re.search(r"Pages wired down:\s+(\d+)", out)
        if not m:
            return None
        wired_mb = int(m.group(1)) * page_bytes / 1024 / 1024
        return wired_mb
    except Exception:
        return None


def _print_servers(servers: list[dict]) -> None:
    if not servers:
        print("ไม่พบ mlx_lm.server กำลังรัน")
        return
    metal_mb = _get_metal_memory_mb()
    print(f"พบ mlx_lm.server {len(servers)} ตัว:")
    print(f"  {'PID':>7}  {'PORT':>5}  {'RSS_MB':>9}  MODEL")
    for s in servers:
        port = s["port"] if s["port"] is not None else "?"
        print(f"  {s['pid']:>7}  {port:>5}  {s['rss_mb']:>9.1f}  {s['model']}")
    if metal_mb:
        print(f"\nระบบรวม wired memory (Metal weights + kernel): {metal_mb / 1024:.1f} GB")
    print("\nหมายเหตุ: RSS ใน ps แสดงแค่ Python heap — weights อยู่ใน Unified Memory ของ Metal "
          "ไม่นับ ดูที่ wired memory ด้านบนแทน")


def _kill(pid: int, label: str) -> bool:
    try:
        os.kill(pid, 15)  # SIGTERM
        print(f"  ส่ง SIGTERM → PID {pid} ({label})")
        return True
    except ProcessLookupError:
        print(f"  PID {pid} ไม่อยู่แล้ว")
        return False
    except PermissionError:
        print(f"  ไม่มีสิทธิ์ฆ่า PID {pid} — ลอง: sudo kill {pid}")
        return False


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    g = p.add_mutually_exclusive_group()
    g.add_argument("--keep", type=int, metavar="PORT", help="ฆ่าทุก server ยกเว้น port นี้")
    g.add_argument("--keep-config", action="store_true",
                   help="ฆ่าทุก server ยกเว้นตัวที่ตรง config.MLX_BASE_URL")
    g.add_argument("--kill", type=int, metavar="PORT", help="ฆ่า server บน port นี้")
    g.add_argument("--kill-all", action="store_true", help="ฆ่าทุก mlx_lm.server")
    args = p.parse_args()

    servers = _list_servers()
    _print_servers(servers)

    if not servers:
        return 1

    # list mode = ไม่ทำอะไรต่อ
    if not (args.keep or args.keep_config or args.kill or args.kill_all):
        return 0

    keep_port: int | None = None
    if args.keep:
        keep_port = args.keep
    elif args.keep_config:
        # หา port จาก config (ต้องอยู่ใน V2 root เพื่อ import)
        sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        try:
            from config import MLX_BASE_URL
            m = re.search(r":(\d+)/", MLX_BASE_URL)
            if not m:
                print(f"[!] หา port จาก MLX_BASE_URL ไม่ได้: {MLX_BASE_URL}")
                return 1
            keep_port = int(m.group(1))
            print(f"\n--keep-config → port จาก config.MLX_BASE_URL = {keep_port}")
        except Exception as e:
            print(f"[!] import config ไม่ได้: {e}")
            return 1

    print()
    if args.kill_all:
        print(f"ฆ่าทั้งหมด {len(servers)} server:")
        for s in servers:
            _kill(s["pid"], f"port={s['port']}, model={s['model']}")
    elif args.kill is not None:
        target = [s for s in servers if s["port"] == args.kill]
        if not target:
            print(f"[!] ไม่พบ server บน port {args.kill}")
            return 1
        for s in target:
            _kill(s["pid"], f"port={args.kill}, model={s['model']}")
    elif keep_port is not None:
        to_kill = [s for s in servers if s["port"] != keep_port]
        if not to_kill:
            print(f"ทุก server อยู่บน port {keep_port} อยู่แล้ว — ไม่ต้องทำอะไร")
            return 0
        print(f"เก็บ port {keep_port}, ฆ่า {len(to_kill)} server:")
        for s in to_kill:
            _kill(s["pid"], f"port={s['port']}, model={s['model']}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
