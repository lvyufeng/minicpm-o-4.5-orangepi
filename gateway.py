#!/usr/bin/env python3
"""
Stub placeholder for gateway.py - public WebSocket endpoint and worker pool management.
"""
import argparse

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8006)
    parser.add_argument("--internal-port", type=int, default=8007)
    parser.add_argument("--http", action="store_true")
    args = parser.parse_args()

    print(f"Gateway stub - would serve on {args.host}:{args.port}")
    print("TODO: Implement gateway from MiniCPM-o-Demo")

if __name__ == "__main__":
    main()
