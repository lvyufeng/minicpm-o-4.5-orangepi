#!/usr/bin/env python3
import socket
import json
import sys

def send_request(sock, request):
    msg = json.dumps(request).encode('utf-8')
    length = len(msg)
    sock.sendall(length.to_bytes(4, byteorder='little') + msg)

def recv_response(sock):
    length_bytes = sock.recv(4)
    if not length_bytes:
        return None
    length = int.from_bytes(length_bytes, byteorder='little')
    data = b''
    while len(data) < length:
        chunk = sock.recv(length - len(data))
        if not chunk:
            break
        data += chunk
    return json.loads(data.decode('utf-8'))

sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
sock.connect(('127.0.0.1', 50051))

print("Sending init request...")
send_request(sock, {"type": "init", "model_path": "models/MiniCPM-o-4.5"})

print("Waiting for response...")
response = recv_response(sock)
print(f"Response: {response}")

sock.close()
