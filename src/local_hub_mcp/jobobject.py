"""Job Object do Windows com KILL_ON_JOB_CLOSE: os llama-server morrem junto com o serviço.

Medido: no Windows, matar o pai não mata o filho; sem isto o llama-server fica órfão segurando
VRAM e a porta ainda responde. O serviço entra no próprio job ao iniciar e os filhos herdam o job
ao nascer (sem janela de corrida entre criar o processo e atribuí-lo).
"""
from __future__ import annotations

import ctypes
import os
from ctypes import wintypes as w

JOB_OBJECT_LIMIT_KILL_ON_JOB_CLOSE = 0x2000
JOB_OBJECT_EXTENDED_LIMIT_INFORMATION = 9
PROCESS_SET_QUOTA = 0x0100
PROCESS_TERMINATE = 0x0001


class _Basica(ctypes.Structure):
    _fields_ = [("PerProcessUserTimeLimit", ctypes.c_int64), ("PerJobUserTimeLimit", ctypes.c_int64),
                ("LimitFlags", w.DWORD), ("MinimumWorkingSetSize", ctypes.c_size_t),
                ("MaximumWorkingSetSize", ctypes.c_size_t), ("ActiveProcessLimit", w.DWORD),
                ("Affinity", ctypes.c_size_t), ("PriorityClass", w.DWORD), ("SchedulingClass", w.DWORD)]


class _Io(ctypes.Structure):
    _fields_ = [(n, ctypes.c_uint64) for n in ("ReadOps", "WriteOps", "OtherOps",
                                               "ReadBytes", "WriteBytes", "OtherBytes")]


class _Estendida(ctypes.Structure):
    _fields_ = [("Basica", _Basica), ("Io", _Io), ("ProcessMemoryLimit", ctypes.c_size_t),
                ("JobMemoryLimit", ctypes.c_size_t), ("PeakProcessMemoryUsed", ctypes.c_size_t),
                ("PeakJobMemoryUsed", ctypes.c_size_t)]


class JobObject:
    def __init__(self) -> None:
        if os.name != "nt":
            raise OSError("Job Object só existe no Windows")
        k = ctypes.WinDLL("kernel32", use_last_error=True)
        k.CreateJobObjectW.restype = w.HANDLE
        k.CreateJobObjectW.argtypes = [ctypes.c_void_p, w.LPCWSTR]
        k.SetInformationJobObject.argtypes = [w.HANDLE, ctypes.c_int, ctypes.c_void_p, w.DWORD]
        k.AssignProcessToJobObject.argtypes = [w.HANDLE, w.HANDLE]
        k.GetCurrentProcess.restype = w.HANDLE
        k.OpenProcess.restype = w.HANDLE
        k.OpenProcess.argtypes = [w.DWORD, w.BOOL, w.DWORD]
        k.CloseHandle.argtypes = [w.HANDLE]
        self._k = k
        self.handle = k.CreateJobObjectW(None, None)
        if not self.handle:
            raise ctypes.WinError(ctypes.get_last_error())
        info = _Estendida()
        info.Basica.LimitFlags = JOB_OBJECT_LIMIT_KILL_ON_JOB_CLOSE
        if not k.SetInformationJobObject(self.handle, JOB_OBJECT_EXTENDED_LIMIT_INFORMATION,
                                         ctypes.byref(info), ctypes.sizeof(info)):
            raise ctypes.WinError(ctypes.get_last_error())

    def atribuir_atual(self) -> bool:
        return bool(self._k.AssignProcessToJobObject(self.handle, self._k.GetCurrentProcess()))

    def atribuir(self, pid: int) -> bool:
        h = self._k.OpenProcess(PROCESS_SET_QUOTA | PROCESS_TERMINATE, False, pid)
        if not h:
            return False
        try:
            return bool(self._k.AssignProcessToJobObject(self.handle, h))
        finally:
            self._k.CloseHandle(h)


_job: JobObject | None = None
_modo = "nenhum"


def entrar() -> str:
    """Chamar uma vez no boot. Modo: 'proprio' (o serviço está no job; filhos herdam),
    'por_filho' (não deu para entrar; cada filho é atribuído depois de criado) ou 'nenhum' (fora do Windows)."""
    global _job, _modo
    if os.name != "nt":
        _modo = "nenhum"
        return _modo
    _job = JobObject()   # a referência global mantém o handle aberto até o processo morrer
    _modo = "proprio" if _job.atribuir_atual() else "por_filho"
    return _modo


def apos_criar(pid: int) -> None:
    if _modo == "por_filho" and _job is not None:
        _job.atribuir(pid)


def modo() -> str:
    return _modo
