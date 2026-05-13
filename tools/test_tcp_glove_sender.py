"""Test script to send fake glove TCP data to the dashboard server.

This script connects to 127.0.0.1:9001 by default and sends clean CSV lines
in the format ``counter,A0,A2,A4,A8,A9``. It is useful for testing the
web interface TCP reader without an actual ESP32 glove.
"""

import argparse
import socket
import sys
import time


def main() -> int:
    parser = argparse.ArgumentParser(description="Send fake glove CSV data over TCP")
    parser.add_argument("--host", default="127.0.0.1", help="TCP server host")
    parser.add_argument("--port", type=int, default=9001, help="TCP server port")
    parser.add_argument("--interval", type=float, default=0.05, help="Seconds between lines")
    parser.add_argument("--count", type=int, default=200, help="Number of lines to send")
    args = parser.parse_args()

    try:
        with socket.create_connection((args.host, args.port), timeout=5) as sock:
            print(f"Connected to {args.host}:{args.port}")
            for i in range(args.count):
                values = [1000 + (i % 100), 2000 + ((i * 2) % 100), 1800 + ((i * 3) % 100), 900 + ((i * 5) % 100), 3000 + ((i * 7) % 100)]
                line = ",".join([str(i)] + [str(v) for v in values]) + "\n"
                sock.sendall(line.encode("utf-8"))
                print(f"Sent: {line.strip()}")
                time.sleep(args.interval)
    except Exception as exc:
        print(f"Failed to send TCP data: {exc}")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
