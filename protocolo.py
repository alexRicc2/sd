"""Framing TCP: mensagens pickle com tamanho prefixado (4 bytes, big-endian)."""

from __future__ import annotations

import pickle
import socket
import struct
import time
from dataclasses import dataclass
from typing import Any

_HEADER = struct.Struct("!I")


@dataclass(frozen=True)
class MetricasEnvio:
    bytes_total: int
    tempo_serializacao_s: float
    tempo_socket_s: float


@dataclass(frozen=True)
class MetricasRecebimento:
    bytes_total: int
    tempo_socket_s: float
    tempo_deserializacao_s: float


def serializar(objeto: Any) -> tuple[bytes, float]:
    inicio = time.perf_counter()
    dados = pickle.dumps(objeto, protocol=pickle.HIGHEST_PROTOCOL)
    return dados, time.perf_counter() - inicio


def deserializar(payload: bytes) -> tuple[Any, float]:
    inicio = time.perf_counter()
    objeto = pickle.loads(payload)
    return objeto, time.perf_counter() - inicio


def enviar(sock: socket.socket, objeto: Any) -> MetricasEnvio:
    dados, tempo_serial = serializar(objeto)
    pacote = _HEADER.pack(len(dados)) + dados
    inicio_socket = time.perf_counter()
    sock.sendall(pacote)
    return MetricasEnvio(
        bytes_total=len(pacote),
        tempo_serializacao_s=tempo_serial,
        tempo_socket_s=time.perf_counter() - inicio_socket,
    )


def receber(sock: socket.socket) -> tuple[Any, MetricasRecebimento]:
    inicio_socket = time.perf_counter()
    cabecalho = _recv_exato(sock, _HEADER.size)
    (tamanho,) = _HEADER.unpack(cabecalho)
    payload = _recv_exato(sock, tamanho)
    tempo_socket = time.perf_counter() - inicio_socket
    objeto, tempo_deserial = deserializar(payload)
    return objeto, MetricasRecebimento(
        bytes_total=_HEADER.size + len(payload),
        tempo_socket_s=tempo_socket,
        tempo_deserializacao_s=tempo_deserial,
    )


def _recv_exato(sock: socket.socket, n: int) -> bytes:
    buffer = bytearray()
    while len(buffer) < n:
        pedaco = sock.recv(n - len(buffer))
        if not pedaco:
            raise ConnectionError("Conexão encerrada antes de receber todos os bytes")
        buffer.extend(pedaco)
    return bytes(buffer)
