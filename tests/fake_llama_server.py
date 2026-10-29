"""llama-server de mentira: aceita qualquer argumento, lê --port/--alias, responde /health, chat e /tts.
Flags de teste: --falhar (sai com código 1, como 'não coube'), --atraso <s> (/health responde 503 antes)."""
import json
import sys
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

args = sys.argv[1:]
porta = int(args[args.index("--port") + 1])
alias = args[args.index("--alias") + 1] if "--alias" in args else "fake"
if "--falhar" in args:
    sys.exit(1)
if "--arquitetura-desconhecida" in args:   # como o llama.cpp diante de um modelo novo demais
    print("llama_model_load: error loading model: error loading model architecture: "
          "unknown model architecture: 'paddleocr-vl'", flush=True)
    sys.exit(1)
atraso = float(args[args.index("--atraso") + 1]) if "--atraso" in args else 0.0
inicio = time.time()


class H(BaseHTTPRequestHandler):
    def log_message(self, *a):
        pass

    def _json(self, status, obj):
        corpo = json.dumps(obj).encode()
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(corpo)))
        self.end_headers()
        self.wfile.write(corpo)

    def do_GET(self):
        if self.path == "/health":
            pronto = time.time() - inicio >= atraso
            return self._json(200 if pronto else 503, {"status": "ok" if pronto else "loading"})
        self._json(404, {"error": "not found"})

    def do_POST(self):
        n = int(self.headers.get("Content-Length") or 0)
        corpo = json.loads(self.rfile.read(n) or b"{}")
        if corpo.get("stream"):
            self.send_response(200)
            self.send_header("Content-Type", "text/event-stream")
            self.end_headers()
            for parte in ("ol", "á"):
                self.wfile.write(f"data: {json.dumps({'choices': [{'delta': {'content': parte}}]})}\n\n".encode())
                self.wfile.flush()
            self.wfile.write(b"data: [DONE]\n\n")
            self.wfile.flush()
            return
        self._json(200, {"model": alias, "path": self.path, "recebido": corpo,
                         "choices": [{"message": {"content": "pronto"}}],
                         "timings": {"predicted_per_second": 42.0}})


print(f"fake llama-server {alias} na porta {porta}", flush=True)
ThreadingHTTPServer(("127.0.0.1", porta), H).serve_forever()
