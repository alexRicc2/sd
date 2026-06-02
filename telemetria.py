"""Telemetria estruturada para relatório de desempenho."""

from __future__ import annotations

import json
import os
import time
from dataclasses import asdict, dataclass, field

import psutil

from matriz_util import gerar_matrizes, multiplicar_local


def bytes_para_mb(valor: int | float) -> float:
    return valor / (1024 * 1024)


def mb_por_s(megabytes: float, segundos: float) -> float:
    if segundos <= 0:
        return 0.0
    return megabytes / segundos


@dataclass
class MetricasRecursos:
    memoria_rss_mb: float
    delta_memoria_mb: float
    tempo_cpu_s: float
    cpu_percent_amostra: float


@dataclass
class MetricasWorker:
    indice: int
    host: str
    porta: int
    bytes_enviados: int
    bytes_recebidos: int
    # Master: envio da requisicao
    tempo_serializacao_envio_s: float
    tempo_socket_envio_s: float
    # Master: espera pela resposta (inclui calculo do worker)
    tempo_espera_resposta_s: float
    tempo_deserializacao_resposta_s: float
    # Worker: recebimento e calculo (reportados pelo worker)
    tempo_recebimento_requisicao_s: float
    tempo_deserializacao_requisicao_s: float
    tempo_calculo_s: float
    tempo_serializacao_resposta_s: float
    tempo_envio_resposta_s: float
    vazao_envio_mb_s: float = 0.0
    vazao_retorno_mb_s: float = 0.0
    tempo_comunicacao_pura_s: float = 0.0

    def __post_init__(self) -> None:
        mb_env = bytes_para_mb(self.bytes_enviados)
        mb_rec = bytes_para_mb(self.bytes_recebidos)
        tempo_envio = self.tempo_serializacao_envio_s + self.tempo_socket_envio_s
        if self.tempo_envio_resposta_s <= 0:
            self.tempo_envio_resposta_s = max(
                self.tempo_espera_resposta_s
                - self.tempo_calculo_s
                - self.tempo_serializacao_resposta_s,
                1e-9,
            )
        tempo_retorno = self.tempo_serializacao_resposta_s + self.tempo_envio_resposta_s
        self.vazao_envio_mb_s = mb_por_s(mb_env, tempo_envio)
        self.vazao_retorno_mb_s = mb_por_s(mb_rec, tempo_retorno)
        self.tempo_comunicacao_pura_s = (
            self.tempo_socket_envio_s
            + self.tempo_recebimento_requisicao_s
            + self.tempo_envio_resposta_s
        )


@dataclass
class MetricasSequencial:
    tempo_total_s: float
    tempo_multiplicacao_s: float
    tempo_cpu_s: float
    memoria_rss_mb: float


@dataclass
class RelatorioDistribuido:
    tamanho: int
    n_workers: int
    workers_endereco: list[str]
    tempo_fatiamento_s: float
    tempo_total_s: float
    tempo_orquestracao_s: float
    tempo_serializacao_master_s: float
    tempo_socket_envio_master_s: float
    tempo_socket_recebimento_master_s: float
    tempo_deserializacao_master_s: float
    bytes_enviados_total: int
    bytes_recebidos_total: int
    vazao_envio_total_mb_s: float
    vazao_recebimento_total_mb_s: float
    workers: list[MetricasWorker] = field(default_factory=list)
    recursos_master: MetricasRecursos | None = None
    sequencial: MetricasSequencial | None = None
    speedup: float | None = None
    eficiencia: float | None = None
    integridade_ok: bool = False

    def __post_init__(self) -> None:
        if self.sequencial and self.sequencial.tempo_total_s > 0:
            self.speedup = self.sequencial.tempo_total_s / self.tempo_total_s
            if self.n_workers > 0:
                self.eficiencia = (self.speedup / self.n_workers) * 100


def medir_sequencial(tamanho: int, seed: int | None) -> MetricasSequencial:
    """Baseline alinhado ao master: T0 após geração, T3 após multiplicação."""
    processo = psutil.Process(os.getpid())
    memoria_inicial = processo.memory_info().rss

    a, b = gerar_matrizes(tamanho, seed=seed)
    cpu_inicio = time.process_time()
    t0 = time.perf_counter()

    t_mult_inicio = time.perf_counter()
    multiplicar_local(a, b)
    tempo_multiplicacao = time.perf_counter() - t_mult_inicio
    tempo_total = time.perf_counter() - t0

    return MetricasSequencial(
        tempo_total_s=tempo_total,
        tempo_multiplicacao_s=tempo_multiplicacao,
        tempo_cpu_s=time.process_time() - cpu_inicio,
        memoria_rss_mb=bytes_para_mb(processo.memory_info().rss - memoria_inicial),
    )


def coletar_recursos(processo: psutil.Process, cpu_inicio: float, memoria_inicial: int) -> MetricasRecursos:
    memoria_final = processo.memory_info().rss
    return MetricasRecursos(
        memoria_rss_mb=bytes_para_mb(memoria_final),
        delta_memoria_mb=bytes_para_mb(memoria_final - memoria_inicial),
        tempo_cpu_s=time.process_time() - cpu_inicio,
        cpu_percent_amostra=processo.cpu_percent(interval=0.1),
    )


def imprimir_relatorio(relatorio: RelatorioDistribuido) -> None:
    sep = "=" * 60
    print(f"\n{sep}")
    print("RELATORIO DE TELEMETRIA - MULTIPLICACAO DISTRIBUIDA")
    print(sep)
    print(f"Matriz: {relatorio.tamanho}x{relatorio.tamanho} | Workers: {relatorio.n_workers}")
    print(f"Destinos: {', '.join(relatorio.workers_endereco)}")
    print()
    print("--- Tempos (s) ---")
    print(f"  T0->T1  Fatiamento/preparo:              {relatorio.tempo_fatiamento_s:10.6f}")
    print(f"  T1->T3  Orquestracao (rede+remontagem):  {relatorio.tempo_orquestracao_s:10.6f}")
    print(f"    |- Serializacao (master):              {relatorio.tempo_serializacao_master_s:10.6f}")
    print(f"    |- Socket envio (master):              {relatorio.tempo_socket_envio_master_s:10.6f}")
    print(f"    |- Socket recebimento (master):        {relatorio.tempo_socket_recebimento_master_s:10.6f}")
    print(f"    `- Deserializacao respostas (master):  {relatorio.tempo_deserializacao_master_s:10.6f}")
    print(f"  T3-T0  Tempo total distribuido:          {relatorio.tempo_total_s:10.6f}")
    print()
    print("--- Trafego de rede ---")
    for w in relatorio.workers:
        print(
            f"  Worker {w.indice} ({w.host}:{w.porta}): "
            f"envio {bytes_para_mb(w.bytes_enviados):7.2f} MB  "
            f"retorno {bytes_para_mb(w.bytes_recebidos):7.2f} MB  "
            f"vazao_envio {w.vazao_envio_mb_s:6.1f} MB/s  "
            f"vazao_retorno {w.vazao_retorno_mb_s:6.1f} MB/s"
        )
        print(
            f"           serializacao {w.tempo_serializacao_envio_s:.6f}s | "
            f"rede pura {w.tempo_comunicacao_pura_s:.6f}s | "
            f"calculo worker {w.tempo_calculo_s:.6f}s | "
            f"latencia E2E {w.tempo_espera_resposta_s:.6f}s"
        )
    print(
        f"  TOTAL: envio {bytes_para_mb(relatorio.bytes_enviados_total):.2f} MB  "
        f"retorno {bytes_para_mb(relatorio.bytes_recebidos_total):.2f} MB  "
        f"vazao_envio {relatorio.vazao_envio_total_mb_s:.1f} MB/s  "
        f"vazao_retorno {relatorio.vazao_recebimento_total_mb_s:.1f} MB/s"
    )
    print()
    print("--- CPU isolada nos workers (s) ---")
    for w in relatorio.workers:
        print(f"  Worker {w.indice}: calculo={w.tempo_calculo_s:.6f}s")
    print()
    if relatorio.sequencial:
        print("--- Comparacao com baseline sequencial ---")
        s = relatorio.sequencial
        print(f"  Tempo total sequencial:       {s.tempo_total_s:.6f} s")
        print(f"  Tempo multiplicacao (seq.):   {s.tempo_multiplicacao_s:.6f} s")
        print(f"  Tempo CPU sequencial:         {s.tempo_cpu_s:.6f} s")
        if relatorio.speedup is not None:
            print(f"  Speedup (seq/dist):           {relatorio.speedup:.3f}x")
        if relatorio.eficiencia is not None:
            print(f"  Eficiencia (speedup/workers): {relatorio.eficiencia:.1f}%")
        overhead = relatorio.tempo_total_s - s.tempo_multiplicacao_s
        print(f"  Overhead distribuido:         {overhead:.6f} s (rede+serializacao+sync)")
    print()
    if relatorio.recursos_master:
        r = relatorio.recursos_master
        print("--- Recursos (master) ---")
        print(f"  Memoria RSS: {r.memoria_rss_mb:.2f} MB (delta {r.delta_memoria_mb:.2f} MB)")
        print(f"  Tempo CPU:   {r.tempo_cpu_s:.6f} s")
        print(f"  CPU amostra: {r.cpu_percent_amostra:.1f}%")
    print()
    print(f"Integridade: {'OK' if relatorio.integridade_ok else 'FALHA'}")
    print(sep)


def relatorio_para_json(relatorio: RelatorioDistribuido) -> str:
    return json.dumps(asdict(relatorio), indent=2, ensure_ascii=False)
