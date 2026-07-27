#!/usr/bin/env python3
"""
Stub placeholder for worker.py - protocol forwarder between gateway and backend.
This will forward requests from the gateway to the C++ backend server.
"""
import argparse

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=22400)
    parser.add_argument("--gpu-id", type=int, default=0)
    parser.add_argument("--backend-server-url", required=True)
    args = parser.parse_args()

    print(f"Worker stub - would forward {args.host}:{args.port} -> {args.backend_server_url}")
    print("TODO: Implement protocol forwarding from MiniCPM-o-Demo")

if __name__ == "__main__":
    main()
