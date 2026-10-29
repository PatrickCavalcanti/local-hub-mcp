"""Adaptador OpenAI <-> llama-server: anexos de áudio/vídeo, STT e TTS.

Portado de scripts anteriores do mesmo autor (openai_audio.py e transcrever.py, 2026-10-28).
"""


class ErroApi(Exception):
    def __init__(self, status: int, msg: str, tipo: str = "invalid_request_error"):
        super().__init__(msg)
        self.status, self.tipo = status, tipo
