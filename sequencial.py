"""Baseline sequencial: gera matrizes, multiplica e registra tempo e recursos."""

from __future__ import annotations

import argparse
import logging
import os
import sys
import time

import psutil

from matriz_util import gerar_matrizes, multiplicar_local, validar_resultado


def _configurar_log(nivel: str) -> None:
    logging.basicConfig(
        level=getattr(logging, nivel.upper(), logging.INFO),
        format="%(asctime)s | %(levelname)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )


def _bytes_para_mb(valor: int) -> float:
    return valor / (1024 * 1024)


def executar(tamanho: int, seed: int | None) -> int:
    processo = psutil.Process(os.getpid())
    processo.cpu_percent(interval=None)

    logging.info("Modo: sequencial (baseline)")
    logging.info("Tamanho da matriz: %d × %d", tamanho, tamanho)

    memoria_inicial = processo.memory_info().rss
    cpu_inicio_total = time.process_time()

    t_inicio_total = time.perf_counter()

    t0 = time.perf_counter()
    a, b = gerar_matrizes(tamanho, seed=seed)
    t1 = time.perf_counter()
    tempo_geracao = t1 - t0
    logging.info("T0→T1 | Geração das matrizes: %.6f s", tempo_geracao)

    t2_inicio = time.perf_counter()
    cpu_mult_inicio = time.process_time()
    c = multiplicar_local(a, b)
    cpu_multiplicacao = time.process_time() - cpu_mult_inicio
    t2 = time.perf_counter()
    tempo_multiplicacao = t2 - t2_inicio
    logging.info(
        "T1→T2 | Multiplicação local: %.6f s (CPU: %.6f s)",
        tempo_multiplicacao,
        cpu_multiplicacao,
    )

    t3 = time.perf_counter()
    valido = validar_resultado(a, b, c)
    tempo_validacao = t3 - t2
    tempo_total = t3 - t_inicio_total

    memoria_final = processo.memory_info().rss
    delta_memoria = memoria_final - memoria_inicial
    cpu_percent = processo.cpu_percent(interval=0.1)

    logging.info("T2→T3 | Validação: %.6f s | resultado=%s", tempo_validacao, valido)
    tempo_cpu_total = time.process_time() - cpu_inicio_total
    logging.info("T3−T0 | Tempo total (relógio): %.6f s", tempo_total)
    logging.info("T3−T0 | Tempo CPU do processo: %.6f s", tempo_cpu_total)
    logging.info(
        "Memória RSS: %.2f MB (Δ %.2f MB)",
        _bytes_para_mb(memoria_final),
        _bytes_para_mb(delta_memoria),
    )
    logging.info("CPU do processo (amostra): %.1f%%", cpu_percent)

    if not valido:
        logging.error("Validação falhou: C não corresponde a A @ B")
        return 1

    return 0


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Multiplicação de matrizes — baseline sequencial"
    )
    parser.add_argument(
        "-n",
        "--tamanho",
        type=int,
        default=512,
        help="Ordem N das matrizes N×N (padrão: 512)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Semente para reprodutibilidade das matrizes aleatórias",
    )
    parser.add_argument(
        "--log-level",
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        help="Nível de verbosidade do log",
    )
    args = parser.parse_args()

    _configurar_log(args.log_level)

    try:
        codigo = executar(args.tamanho, args.seed)
    except ValueError as exc:
        logging.error("%s", exc)
        codigo = 2

    sys.exit(codigo)


if __name__ == "__main__":
    main()
