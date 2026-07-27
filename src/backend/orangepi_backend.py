"""Orange Pi (Ascend 310B) backend implementation for MiniCPM-O-4.5.

This backend wraps the C++ inference engine and provides the same interface
as PyTorchBackend for drop-in compatibility with MiniCPM-o-Demo.
"""

from __future__ import annotations

import asyncio
import json
import logging
import socket
import struct
import time
from typing import Any, Dict, Iterator, List, Optional

import numpy as np

logger = logging.getLogger("orangepi_backend")


class OrangePiBackend:
    """MiniCPM-O-4.5 Orange Pi (Ascend 310B) inference backend.

    Communicates with the C++ backend_server via TCP socket.
    Provides the same interface as PyTorchBackend for compatibility.
    """

    def __init__(
        self,
        model_path: str,
        gpu_id: int,
        backend_server_host: str = "127.0.0.1",
        backend_server_port: int = 50051,
        duplex_pause_timeout: float = 60.0,
        **kwargs,  # Accept and ignore PyTorch-specific args
    ):
        self.model_path = model_path
        self.gpu_id = gpu_id
        self.backend_server_host = backend_server_host
        self.backend_server_port = backend_server_port
        self.duplex_pause_timeout = duplex_pause_timeout

        self.status = "loading"
        self._socket: Optional[socket.socket] = None
        self._next_request_id = 0

        # Load tokenizer
        try:
            from transformers import AutoTokenizer
            self.tokenizer = AutoTokenizer.from_pretrained(
                model_path, trust_remote_code=True
            )
            logger.info(f"[NPU {gpu_id}] Loaded tokenizer from {model_path}")
        except Exception as e:
            logger.warning(f"[NPU {gpu_id}] Failed to load tokenizer: {e}")
            self.tokenizer = None

    def load_model(self) -> None:
        """Load model (connect to backend server)"""
        self.status = "loading"
        logger.info(
            f"[NPU {self.gpu_id}] Connecting to backend server at "
            f"{self.backend_server_host}:{self.backend_server_port}..."
        )

        try:
            self._socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            self._socket.connect((self.backend_server_host, self.backend_server_port))
            self._socket.settimeout(900.0)  # 15min timeout for model loading

            # Send init request
            response = self._send_request({
                "type": "init",
                "model_path": self.model_path,
            })

            if response.get("status") != "ok":
                raise RuntimeError(f"Backend init failed: {response.get('error')}")

            self.status = "ready"
            logger.info(f"[NPU {self.gpu_id}] Backend connected successfully")

        except Exception as e:
            self.status = "error"
            logger.error(f"[NPU {self.gpu_id}] Failed to connect to backend: {e}")
            raise

    def _send_request(self, request: Dict[str, Any]) -> Dict[str, Any]:
        """Send a request to backend server and receive response.

        Protocol: 4-byte length prefix (little-endian) + JSON payload
        """
        if self._socket is None:
            raise RuntimeError("Backend not connected")

        request["request_id"] = self._next_request_id
        self._next_request_id += 1

        payload = json.dumps(request).encode("utf-8")
        header = struct.pack("<I", len(payload))

        self._socket.sendall(header + payload)

        # Receive response
        header = self._recv_exactly(4)
        response_len = struct.unpack("<I", header)[0]
        response_data = self._recv_exactly(response_len)
        response = json.loads(response_data.decode("utf-8"))

        return response

    def _recv_exactly(self, n: int) -> bytes:
        """Receive exactly n bytes from socket."""
        data = b""
        while len(data) < n:
            chunk = self._socket.recv(n - len(data))
            if not chunk:
                raise ConnectionError("Socket connection closed")
            data += chunk
        return data

    def metrics(self) -> Dict[str, Any]:
        """Return backend metrics snapshot."""
        if self.status != "ready":
            return {"backend": "orangepi", "kv_cache_length": 0}

        try:
            response = self._send_request({"type": "metrics"})
            return response.get("metrics", {})
        except Exception as e:
            logger.warning(f"Failed to get metrics: {e}")
            return {"backend": "orangepi", "kv_cache_length": 0}

    def chat_prefill(
        self,
        session_id: str,
        msgs: list,
        omni_mode: bool = False,
        max_slice_nums: Optional[int] = None,
        use_tts_template: bool = False,
        enable_thinking: bool = False,
    ) -> str:
        """Prefill chat context."""
        if self.tokenizer is None:
            raise RuntimeError("Tokenizer not loaded")

        # Format messages into text prompt
        # Simple concatenation - in production should use apply_chat_template
        prompt_text = ""
        for msg in msgs:
            role = msg.get("role", "user")
            content = msg.get("content", "")
            prompt_text += f"{role}: {content}\n"

        # Tokenize
        input_ids = self.tokenizer.encode(prompt_text, add_special_tokens=True)
        logger.info(f"[NPU {self.gpu_id}] Tokenized {len(input_ids)} tokens")

        # Send to backend with token IDs
        response = self._send_request({
            "type": "chat_prefill",
            "session_id": session_id,
            "input_ids": input_ids,
        })

        if response.get("status") != "ok":
            raise RuntimeError(f"chat_prefill failed: {response.get('error')}")

        return response.get("prompt", "")

    def chat_init_tts(self, ref_audio: Optional[np.ndarray]) -> None:
        """Initialize TTS with reference audio."""
        request: Dict[str, Any] = {"type": "chat_init_tts"}

        if ref_audio is not None:
            # Serialize numpy array as base64
            import base64
            audio_bytes = ref_audio.astype(np.float32).tobytes()
            request["ref_audio_data"] = base64.b64encode(audio_bytes).decode("ascii")
            request["ref_audio_shape"] = list(ref_audio.shape)

        response = self._send_request(request)

        if response.get("status") != "ok":
            raise RuntimeError(f"chat_init_tts failed: {response.get('error')}")

    def chat_streaming_generate(
        self,
        session_id: str,
        generate_audio: bool = True,
        max_new_tokens: int = 256,
        length_penalty: float = 1.1,
    ) -> Iterator[Any]:
        """Stream chat generation chunks."""
        if self.tokenizer is None:
            raise RuntimeError("Tokenizer not loaded")

        response = self._send_request({
            "type": "chat_streaming_generate",
            "session_id": session_id,
            "generate_audio": generate_audio,
            "max_new_tokens": max_new_tokens,
            "length_penalty": length_penalty,
        })

        if response.get("status") != "ok":
            raise RuntimeError(f"chat_streaming_generate failed: {response.get('error')}")

        # The backend streams chunks; receive them one by one
        while True:
            chunk_response = self._send_request({"type": "get_next_chunk"})

            if chunk_response.get("done"):
                break

            if chunk_response.get("status") != "ok":
                logger.warning(f"Chunk error: {chunk_response.get('error')}")
                break

            chunk = chunk_response.get("chunk", {})
            token_id = chunk.get("token_id")
            is_eos = chunk.get("is_eos", False)

            if token_id is not None:
                # Decode token to text
                token_text = self.tokenizer.decode([int(token_id)], skip_special_tokens=False)

                # Yield chunk with decoded text
                yield {
                    "text": token_text,
                    "token_id": int(token_id),
                    "is_eos": is_eos,
                }

                if is_eos:
                    break

            yield self._deserialize_chunk(chunk_response.get("chunk"))

    def chat_non_streaming_generate(
        self,
        session_id: str,
        max_new_tokens: int = 256,
        generate_audio: bool = False,
        use_tts_template: bool = True,
        enable_thinking: bool = False,
        tts_ref_audio: Optional[np.ndarray] = None,
        length_penalty: float = 1.1,
    ) -> Any:
        """Non-streaming chat generation."""
        request: Dict[str, Any] = {
            "type": "chat_generate",
            "session_id": session_id,
            "max_new_tokens": max_new_tokens,
            "generate_audio": generate_audio,
            "use_tts_template": use_tts_template,
            "enable_thinking": enable_thinking,
            "length_penalty": length_penalty,
        }

        if tts_ref_audio is not None:
            import base64
            audio_bytes = tts_ref_audio.astype(np.float32).tobytes()
            request["tts_ref_audio_data"] = base64.b64encode(audio_bytes).decode("ascii")
            request["tts_ref_audio_shape"] = list(tts_ref_audio.shape)

        response = self._send_request(request)

        if response.get("status") != "ok":
            raise RuntimeError(f"chat_generate failed: {response.get('error')}")

        return self._deserialize_result(response.get("result"))

    def _deserialize_chunk(self, chunk_data: Dict) -> Any:
        """Deserialize a streaming chunk from JSON."""
        # TODO: implement proper chunk deserialization matching StreamingChunk schema
        return chunk_data

    def _deserialize_result(self, result_data: Dict) -> Any:
        """Deserialize a generation result from JSON."""
        # TODO: implement proper result deserialization
        return result_data

    def shutdown(self) -> None:
        """Shutdown backend connection."""
        if self._socket is not None:
            try:
                self._send_request({"type": "shutdown"})
                self._socket.close()
            except Exception as e:
                logger.warning(f"Shutdown error: {e}")
            finally:
                self._socket = None
        self.status = "shutdown"

    # Duplex mode stubs (not implemented yet)
    def set_duplex_config(self, config: Optional[Dict[str, Any]]) -> None:
        pass

    def duplex_prepare(self, *args, **kwargs) -> str:
        raise NotImplementedError("Duplex mode not yet implemented for Orange Pi backend")

    def duplex_prefill(self, *args, **kwargs) -> Dict[str, Any]:
        raise NotImplementedError("Duplex mode not yet implemented for Orange Pi backend")

    def duplex_generate(self, *args, **kwargs) -> Any:
        raise NotImplementedError("Duplex mode not yet implemented for Orange Pi backend")

    def duplex_finalize(self) -> None:
        pass

    def duplex_stop(self) -> None:
        pass

    def duplex_cleanup(self) -> None:
        pass

    # Half-duplex stubs
    def half_duplex_prefill(self, *args, **kwargs) -> str:
        raise NotImplementedError("Half-duplex not yet implemented")

    def half_duplex_init_tts(self, *args, **kwargs) -> None:
        raise NotImplementedError("Half-duplex not yet implemented")

    def half_duplex_generate(self, *args, **kwargs) -> Iterator[Any]:
        raise NotImplementedError("Half-duplex not yet implemented")

    def half_duplex_complete_turn(self, *args, **kwargs) -> Any:
        raise NotImplementedError("Half-duplex not yet implemented")

    def reset_half_duplex_session(self) -> None:
        pass
