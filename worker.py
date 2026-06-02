"""Nó worker: escuta TCP, recebe fatia de A + matriz B, calcula e devolve."""

from __future__ import annotations

import argparse
import logging
import signal
import socket
import sys
import time

from matriz_util import multiplicar_local
from protocolo import enviar, receber, serializar


def _configurar_log(nivel: str) -> None:
    logging.basicConfig(
        level=getattr(logging, nivel.upper(), logging.INFO),
        format="%(asctime)s | %(levelname)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )


def _processar_conexao(conn: socket.socket, endereco: tuple[str, int]) -> None:
    logging.info("Conexão de %s:%d", endereco[0], endereco[1])

    requisicao, metricas_receb = receber(conn)
    fatia_a = requisicao["fatia_a"]
    matriz_b = requisicao["matriz_b"]
    indice = requisicao["indice"]

    logging.info(
        "Pacote recebido | índice=%d | fatia A%s | B%s | socket: %.6f s | deserialização: %.6f s",
        indice,
        fatia_a.shape,
        matriz_b.shape,
        metricas_receb.tempo_socket_s,
        metricas_receb.tempo_deserializacao_s,
    )

    t_calc_inicio = time.perf_counter()
    resultado = multiplicar_local(fatia_a, matriz_b)
    tempo_calculo = time.perf_counter() - t_calc_inicio

    resposta = {
        "indice": indice,
        "resultado": resultado,
        "metricas": {
            "bytes_recebidos": metricas_receb.bytes_total,
            "tempo_socket_recebimento_s": metricas_receb.tempo_socket_s,
            "tempo_deserializacao_s": metricas_receb.tempo_deserializacao_s,
            "tempo_calculo_s": tempo_calculo,
            "tempo_serializacao_resposta_s": 0.0,
            "tempo_socket_resposta_s": 0.0,
        },
    }

    _, tempo_serial_resposta = serializar(resposta)
    resposta["metricas"]["tempo_serializacao_resposta_s"] = tempo_serial_resposta

    metricas_envio = enviar(conn, resposta)
    logging.info(
        "Resposta enviada | indice=%d | resultado%s | calculo: %.6f s | "
        "serializacao: %.6f s | socket: %.6f s",
        indice,
        resultado.shape,
        tempo_calculo,
        metricas_envio.tempo_serializacao_s,
        metricas_envio.tempo_socket_s,
    )


def executar(host: str, porta: int) -> None:
    servidor = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    servidor.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    servidor.bind((host, porta))
    servidor.listen()
    encerrar = False

    def _solicitar_encerramento(signum: int, _frame) -> None:
        nonlocal encerrar
        if encerrar:
            logging.warning("Forçando saída...")
            sys.exit(1)
        encerrar = True
        logging.info(
            "Sinal %s recebido — encerrando (Ctrl+C de novo para forçar)",
            signum,
        )
        servidor.close()

    signal.signal(signal.SIGINT, _solicitar_encerramento)
    if hasattr(signal, "SIGBREAK"):
        signal.signal(signal.SIGBREAK, _solicitar_encerramento)

    logging.info("Worker escutando em %s:%d", host, porta)
    logging.info("Para parar: Ctrl+C neste terminal (não Ctrl+Q)")

    try:
        while not encerrar:
            try:
                servidor.settimeout(1.0)
                conn, endereco = servidor.accept()
            except socket.timeout:
                continue
            except OSError:
                if encerrar:
                    break
                raise

            try:
                with conn:
                    _processar_conexao(conn, endereco)
            except Exception:
                logging.exception("Erro ao processar conexão de %s:%d", *endereco)
    except KeyboardInterrupt:
        logging.info("Interrompido pelo teclado")
    finally:
        try:
            servidor.close()
        except OSError:
            pass
        logging.info("Worker encerrado")


def main() -> None:
    parser = argparse.ArgumentParser(description="Worker de multiplicação de matrizes")
    parser.add_argument("porta", type=int, help="Porta TCP (ex.: 5001)")
    parser.add_argument(
        "--host",
        default="0.0.0.0",
        help="Interface de bind (padrão: 0.0.0.0 — aceita localhost e rede)",
    )
    parser.add_argument(
        "--log-level",
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
    )
    args = parser.parse_args()

    if not (1 <= args.porta <= 65535):
        logging.error("Porta inválida: %d", args.porta)
        sys.exit(2)

    _configurar_log(args.log_level)
    executar(args.host, args.porta)


if __name__ == "__main__":
    main()
