"""Orquestrador: fatia A, envia aos workers, remonta C e valida integridade."""

from __future__ import annotations

import argparse
import logging
import os
import socket
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed

import psutil

from matriz_util import (
    fatiar_matriz_a,
    gerar_matrizes,
    remontar_matriz_c,
    validar_resultado,
)
from protocolo import enviar, receber
from telemetria import (
    MetricasWorker,
    RelatorioDistribuido,
    bytes_para_mb,
    coletar_recursos,
    imprimir_relatorio,
    mb_por_s,
    medir_sequencial,
    relatorio_para_json,
)


def _configurar_log(nivel: str) -> None:
    logging.basicConfig(
        level=getattr(logging, nivel.upper(), logging.INFO),
        format="%(asctime)s | %(levelname)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )


def _parse_workers(valor: str) -> list[tuple[str, int]]:
    destinos: list[tuple[str, int]] = []
    for item in valor.split(","):
        item = item.strip()
        if not item:
            continue
        if ":" not in item:
            raise ValueError(f"Worker inválido (use host:porta): {item}")
        host, porta_str = item.rsplit(":", 1)
        destinos.append((host, int(porta_str)))
    if len(destinos) < 1:
        raise ValueError("Informe ao menos um worker (ex.: 127.0.0.1:5001,127.0.0.1:5002)")
    return destinos


def _enviar_fatia(
    host: str,
    porta: int,
    indice: int,
    fatia_a,
    matriz_b,
) -> tuple[dict, MetricasWorker]:
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    try:
        sock.connect((host, porta))
        requisicao = {
            "indice": indice,
            "fatia_a": fatia_a,
            "matriz_b": matriz_b,
        }
        metricas_envio = enviar(sock, requisicao)
        resposta, metricas_receb = receber(sock)
        wm = resposta["metricas"]

        metricas = MetricasWorker(
            indice=indice,
            host=host,
            porta=porta,
            bytes_enviados=metricas_envio.bytes_total,
            bytes_recebidos=metricas_receb.bytes_total,
            tempo_serializacao_envio_s=metricas_envio.tempo_serializacao_s,
            tempo_socket_envio_s=metricas_envio.tempo_socket_s,
            tempo_espera_resposta_s=metricas_receb.tempo_socket_s,
            tempo_deserializacao_resposta_s=metricas_receb.tempo_deserializacao_s,
            tempo_recebimento_requisicao_s=wm["tempo_socket_recebimento_s"],
            tempo_deserializacao_requisicao_s=wm["tempo_deserializacao_s"],
            tempo_calculo_s=wm["tempo_calculo_s"],
            tempo_serializacao_resposta_s=wm["tempo_serializacao_resposta_s"],
            tempo_envio_resposta_s=wm["tempo_socket_resposta_s"],
        )
        return resposta, metricas
    finally:
        sock.close()


def executar(
    tamanho: int,
    seed: int | None,
    workers: list[tuple[str, int]],
    *,
    comparar_sequencial: bool,
) -> tuple[int, RelatorioDistribuido | None]:
    n_workers = len(workers)
    if n_workers < 2:
        logging.warning("Menos de 2 workers; particionamento pode não refletir a arquitetura alvo")

    processo = psutil.Process(os.getpid())
    memoria_inicial = processo.memory_info().rss
    cpu_inicio = time.process_time()

    logging.info("Modo: distribuído (master)")
    logging.info("Tamanho da matriz: %d × %d", tamanho, tamanho)
    logging.info("Workers: %s", ", ".join(f"{h}:{p}" for h, p in workers))

    t_inicio_geracao = time.perf_counter()
    a, b = gerar_matrizes(tamanho, seed=seed)
    t0 = time.perf_counter()
    logging.info("Geração das matrizes: %.6f s", t0 - t_inicio_geracao)

    fatias = fatiar_matriz_a(a, n_workers)
    for i, fatia in enumerate(fatias):
        logging.info("Fatia %d → linhas %d (%s)", i, fatia.shape[0], fatia.shape)

    t1 = time.perf_counter()
    tempo_fatiamento = t1 - t0
    logging.info("T0→T1 | Fatiamento e preparo: %.6f s", tempo_fatiamento)

    respostas: list[dict] = []
    metricas_workers: list[MetricasWorker] = []

    with ThreadPoolExecutor(max_workers=n_workers) as pool:
        futuros = {
            pool.submit(
                _enviar_fatia,
                host,
                porta,
                indice,
                fatias[indice],
                b,
            ): (indice, host, porta)
            for indice, (host, porta) in enumerate(workers)
        }
        for futuro in as_completed(futuros):
            indice, host, porta = futuros[futuro]
            resposta, metricas = futuro.result()
            respostas.append(resposta)
            metricas_workers.append(metricas)
            logging.info(
                "Worker %s:%d | ↑ %.2f MB | rede↑ %.6f s | serialização↑ %.6f s",
                host,
                porta,
                bytes_para_mb(metricas.bytes_enviados),
                metricas.tempo_socket_envio_s,
                metricas.tempo_serializacao_envio_s,
            )

    respostas.sort(key=lambda r: r["indice"])
    metricas_workers.sort(key=lambda m: m.indice)
    partes = [r["resultado"] for r in respostas]
    c_distribuido = remontar_matriz_c(partes)
    t3 = time.perf_counter()

    tempo_total = t3 - t0
    tempo_orquestracao = t3 - t1

    tempo_serial_master = sum(m.tempo_serializacao_envio_s for m in metricas_workers)
    tempo_envio_master = sum(m.tempo_socket_envio_s for m in metricas_workers)
    tempo_receb_master = sum(m.tempo_espera_resposta_s for m in metricas_workers)
    tempo_deserial_master = sum(m.tempo_deserializacao_resposta_s for m in metricas_workers)
    bytes_enviados = sum(m.bytes_enviados for m in metricas_workers)
    bytes_recebidos = sum(m.bytes_recebidos for m in metricas_workers)

    tempo_envio_total = tempo_serial_master + tempo_envio_master
    tempo_retorno_total = sum(
        m.tempo_serializacao_resposta_s + m.tempo_envio_resposta_s
        for m in metricas_workers
    )

    integridade = validar_resultado(a, b, c_distribuido)

    sequencial = medir_sequencial(tamanho, seed) if comparar_sequencial else None

    relatorio = RelatorioDistribuido(
        tamanho=tamanho,
        n_workers=n_workers,
        workers_endereco=[f"{h}:{p}" for h, p in workers],
        tempo_fatiamento_s=tempo_fatiamento,
        tempo_total_s=tempo_total,
        tempo_orquestracao_s=tempo_orquestracao,
        tempo_serializacao_master_s=tempo_serial_master,
        tempo_socket_envio_master_s=tempo_envio_master,
        tempo_socket_recebimento_master_s=tempo_receb_master,
        tempo_deserializacao_master_s=tempo_deserial_master,
        bytes_enviados_total=bytes_enviados,
        bytes_recebidos_total=bytes_recebidos,
        vazao_envio_total_mb_s=mb_por_s(bytes_para_mb(bytes_enviados), tempo_envio_total),
        vazao_recebimento_total_mb_s=mb_por_s(bytes_para_mb(bytes_recebidos), tempo_retorno_total),
        workers=metricas_workers,
        recursos_master=coletar_recursos(processo, cpu_inicio, memoria_inicial),
        sequencial=sequencial,
        integridade_ok=integridade,
    )

    imprimir_relatorio(relatorio)

    if not integridade:
        logging.error("Falha de integridade: particionamento ou comunicação incorretos")
        return 1, relatorio

    return 0, relatorio


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Master — orquestração da multiplicação distribuída"
    )
    parser.add_argument("-n", "--tamanho", type=int, default=512)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument(
        "--workers",
        default="127.0.0.1:5001,127.0.0.1:5002",
        help="Lista host:porta separada por vírgula",
    )
    parser.add_argument(
        "--comparar-sequencial",
        action="store_true",
        help="Mede baseline sequencial (mesma seed) e calcula speedup/eficiência",
    )
    parser.add_argument(
        "--json",
        metavar="ARQUIVO",
        help="Salva relatório estruturado em JSON",
    )
    parser.add_argument(
        "--log-level",
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
    )
    args = parser.parse_args()

    _configurar_log(args.log_level)

    try:
        destinos = _parse_workers(args.workers)
        codigo, relatorio = executar(
            args.tamanho,
            args.seed,
            destinos,
            comparar_sequencial=args.comparar_sequencial,
        )
        if relatorio and args.json:
            with open(args.json, "w", encoding="utf-8") as arquivo:
                arquivo.write(relatorio_para_json(relatorio))
            logging.info("Relatório JSON salvo em %s", args.json)
    except (ValueError, ConnectionError, OSError) as exc:
        logging.error("%s", exc)
        codigo = 2

    sys.exit(codigo)


if __name__ == "__main__":
    main()
