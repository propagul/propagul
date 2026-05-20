#!/usr/bin/env python3
"""Mock inference backends for E2E testing.

Starts lightweight HTTP servers that mimic the API surfaces of
vLLM, TGI, and llama.cpp. No GPU required — pure HTTP stubs.

Usage:
    python3 scripts/mock_backends.py

Starts:
    - Mock vLLM    on :18000  (OpenAI-compatible /v1/models, /v1/chat/completions)
    - Mock TGI     on :18080  (/info, /generate)
    - Mock llama.cpp on :18081 (/health, /v1/models, /v1/chat/completions)

Each mock responds with realistic-looking JSON payloads so that
the Propagul agent's backend detection and telemetry collection
works identically to a real backend.
"""

import json
import threading
import time
from http.server import HTTPServer, BaseHTTPRequestHandler


# ─── vLLM Mock ────────────────────────────────────────────────────

class VLLMHandler(BaseHTTPRequestHandler):
    """Mimics vLLM's OpenAI-compatible API."""

    def do_GET(self):
        if self.path == "/v1/models":
            self._json_response({
                "object": "list",
                "data": [{
                    "id": "Qwen/Qwen2.5-1.5B-Instruct",
                    "object": "model",
                    "created": int(time.time()),
                    "owned_by": "vllm",
                    "root": "Qwen/Qwen2.5-1.5B-Instruct",
                    "parent": None,
                    "max_model_len": 2048,
                    "permission": [{
                        "id": "modelperm-mock",
                        "object": "model_permission",
                        "created": int(time.time()),
                        "allow_sampling": True,
                        "allow_logprobs": True,
                    }],
                }],
            })
        elif self.path == "/health":
            self._json_response({"status": "ok"})
        elif self.path == "/version":
            self._json_response({"version": "0.6.6"})
        else:
            self.send_error(404)

    def do_POST(self):
        if self.path == "/v1/chat/completions":
            content_len = int(self.headers.get("Content-Length", 0))
            body = json.loads(self.rfile.read(content_len)) if content_len else {}
            model = body.get("model", "Qwen/Qwen2.5-1.5B-Instruct")
            self._json_response({
                "id": "chatcmpl-mock-vllm",
                "object": "chat.completion",
                "created": int(time.time()),
                "model": model,
                "choices": [{
                    "index": 0,
                    "message": {
                        "role": "assistant",
                        "content": f"[Mock vLLM] Hello from {model}! This is a test response."
                    },
                    "finish_reason": "stop",
                }],
                "usage": {"prompt_tokens": 10, "completion_tokens": 20, "total_tokens": 30},
            })
        else:
            self.send_error(404)

    def _json_response(self, data, status=200):
        body = json.dumps(data).encode()
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, format, *args):
        pass  # Suppress access logs


# ─── TGI Mock ─────────────────────────────────────────────────────

class TGIHandler(BaseHTTPRequestHandler):
    """Mimics HuggingFace Text Generation Inference API."""

    def do_GET(self):
        if self.path == "/info":
            self._json_response({
                "model_id": "TinyLlama/TinyLlama-1.1B-Chat-v1.0",
                "model_sha": "mock-sha-abc123",
                "model_dtype": "float16",
                "model_device_type": "cuda",
                "model_pipeline_tag": "text-generation",
                "max_concurrent_requests": 128,
                "max_best_of": 2,
                "max_stop_sequences": 4,
                "max_input_tokens": 1024,
                "max_total_tokens": 2048,
                "waiting_served_ratio": 0.3,
                "max_batch_total_tokens": 32000,
                "max_waiting_tokens": 20,
                "max_batch_size": None,
                "validation_workers": 2,
                "max_client_batch_size": 4,
                "version": "2.4.1",
                "sha": "mock-sha-tgi",
                "docker_label": "ghcr.io/huggingface/text-generation-inference:2.4.1",
            })
        elif self.path == "/health":
            self._json_response({"status": "ok"})
        else:
            self.send_error(404)

    def do_POST(self):
        if self.path == "/generate":
            content_len = int(self.headers.get("Content-Length", 0))
            body = json.loads(self.rfile.read(content_len)) if content_len else {}
            self._json_response({
                "generated_text": "[Mock TGI] Hello! This is a test response from TinyLlama.",
                "details": {
                    "finish_reason": "length",
                    "generated_tokens": 20,
                    "seed": None,
                },
            })
        elif self.path == "/v1/chat/completions":
            content_len = int(self.headers.get("Content-Length", 0))
            body = json.loads(self.rfile.read(content_len)) if content_len else {}
            self._json_response({
                "id": "chatcmpl-mock-tgi",
                "object": "chat.completion",
                "created": int(time.time()),
                "model": "TinyLlama/TinyLlama-1.1B-Chat-v1.0",
                "choices": [{
                    "index": 0,
                    "message": {
                        "role": "assistant",
                        "content": "[Mock TGI] Hello! This is a test response."
                    },
                    "finish_reason": "stop",
                }],
                "usage": {"prompt_tokens": 10, "completion_tokens": 15, "total_tokens": 25},
            })
        else:
            self.send_error(404)

    def _json_response(self, data, status=200):
        body = json.dumps(data).encode()
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, format, *args):
        pass


# ─── llama.cpp Mock ──────────────────────────────────────────────

class LlamaCppHandler(BaseHTTPRequestHandler):
    """Mimics llama.cpp server (llama-server) API."""

    def do_GET(self):
        if self.path == "/health":
            self._json_response({"status": "ok"})
        elif self.path == "/v1/models":
            self._json_response({
                "object": "list",
                "data": [{
                    "id": "tinyllama-1.1b-chat-v1.0.Q4_K_M.gguf",
                    "object": "model",
                    "created": int(time.time()),
                    "owned_by": "llama.cpp",
                }],
            })
        elif self.path == "/props":
            self._json_response({
                "total_slots": 1,
                "chat_template": "{% for message in messages %}...",
                "default_generation_settings": {
                    "n_ctx": 2048,
                    "model": "tinyllama-1.1b-chat-v1.0.Q4_K_M.gguf",
                },
            })
        elif self.path == "/slots":
            self._json_response([{
                "id": 0,
                "state": 0,
                "model": "tinyllama-1.1b-chat-v1.0.Q4_K_M.gguf",
                "n_ctx": 2048,
                "n_predict": 256,
            }])
        else:
            self.send_error(404)

    def do_POST(self):
        if self.path == "/v1/chat/completions":
            content_len = int(self.headers.get("Content-Length", 0))
            body = json.loads(self.rfile.read(content_len)) if content_len else {}
            self._json_response({
                "id": "chatcmpl-mock-lcpp",
                "object": "chat.completion",
                "created": int(time.time()),
                "model": "tinyllama-1.1b-chat-v1.0.Q4_K_M.gguf",
                "choices": [{
                    "index": 0,
                    "message": {
                        "role": "assistant",
                        "content": "[Mock llama.cpp] Hello! This is a GGUF test response."
                    },
                    "finish_reason": "stop",
                }],
                "usage": {"prompt_tokens": 8, "completion_tokens": 12, "total_tokens": 20},
            })
        elif self.path == "/completion":
            content_len = int(self.headers.get("Content-Length", 0))
            body = json.loads(self.rfile.read(content_len)) if content_len else {}
            self._json_response({
                "content": "[Mock llama.cpp] Hello from llama-server!",
                "stop": True,
                "model": "tinyllama-1.1b-chat-v1.0.Q4_K_M.gguf",
                "tokens_predicted": 12,
                "tokens_evaluated": 8,
            })
        else:
            self.send_error(404)

    def _json_response(self, data, status=200):
        body = json.dumps(data).encode()
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, format, *args):
        pass


# ─── Main ────────────────────────────────────────────────────────

def start_server(handler_class, port, name):
    server = HTTPServer(("127.0.0.1", port), handler_class)
    print(f"  ✅ Mock {name:10s} listening on http://127.0.0.1:{port}")
    server.serve_forever()


def main():
    print(f"\n{'='*60}")
    print(f"  PROPAGUL E2E — Mock Inference Backends")
    print(f"{'='*60}\n")

    threads = [
        threading.Thread(target=start_server, args=(VLLMHandler, 18000, "vLLM"), daemon=True),
        threading.Thread(target=start_server, args=(TGIHandler, 18080, "TGI"), daemon=True),
        threading.Thread(target=start_server, args=(LlamaCppHandler, 18081, "llama.cpp"), daemon=True),
    ]

    for t in threads:
        t.start()

    print(f"\n  All mock backends running. Press Ctrl+C to stop.\n")
    print(f"  To test detection:")
    print(f"    curl http://localhost:18000/v1/models   # vLLM")
    print(f"    curl http://localhost:18080/info         # TGI")
    print(f"    curl http://localhost:18081/health       # llama.cpp")
    print(f"\n  To start agents against these mocks:")
    print(f"    python -m propagul.mesh.cli start --name mock-vllm --ollama http://localhost:18000 --api-key pg_pro_... --interval 10")
    print(f"    python -m propagul.mesh.cli start --name mock-tgi  --ollama http://localhost:18080 --api-key pg_pro_... --interval 10")
    print(f"    python -m propagul.mesh.cli start --name mock-lcpp --ollama http://localhost:18081 --api-key pg_pro_... --interval 10")
    print()

    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        print("\n  Shutting down mock backends...")


if __name__ == "__main__":
    main()
