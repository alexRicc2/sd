"""Funções puras para geração, multiplicação e validação de matrizes."""

from __future__ import annotations

import numpy as np


def gerar_matrizes(n: int, seed: int | None = None) -> tuple[np.ndarray, np.ndarray]:
    """Gera duas matrizes aleatórias N×N com valores em [0, 1)."""
    if n <= 0:
        raise ValueError(f"Tamanho da matriz deve ser positivo, recebido: {n}")

    rng = np.random.default_rng(seed)
    a = rng.random((n, n), dtype=np.float64)
    b = rng.random((n, n), dtype=np.float64)
    return a, b


def multiplicar_local(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Multiplicação de matrizes C = A @ B em um único processo."""
    if a.ndim != 2 or b.ndim != 2:
        raise ValueError("Entradas devem ser matrizes bidimensionais")
    if a.shape[1] != b.shape[0]:
        raise ValueError(
            f"Dimensões incompatíveis: A{a.shape} × B{b.shape}"
        )

    return a @ b


def fatiar_matriz_a(a: np.ndarray, n_partes: int) -> list[np.ndarray]:
    """Divide A por linhas em n_partes fatias (balanceadas)."""
    if a.ndim != 2:
        raise ValueError("Matriz A deve ser bidimensional")
    if n_partes <= 0:
        raise ValueError(f"n_partes deve ser positivo, recebido: {n_partes}")

    linhas = a.shape[0]
    fatias: list[np.ndarray] = []
    inicio = 0
    for i in range(n_partes):
        fim = (linhas * (i + 1)) // n_partes
        fatias.append(a[inicio:fim])
        inicio = fim
    return fatias


def remontar_matriz_c(partes: list[np.ndarray]) -> np.ndarray:
    """Empilha as fatias parciais de C na ordem dos índices."""
    if not partes:
        raise ValueError("Lista de partes vazia")
    return np.vstack(partes)


def validar_resultado(
    a: np.ndarray,
    b: np.ndarray,
    c: np.ndarray,
    *,
    rtol: float = 1e-5,
    atol: float = 1e-8,
) -> bool:
    """Verifica se C é numericamente igual a A @ B."""
    if c.shape != (a.shape[0], b.shape[1]):
        return False

    esperado = multiplicar_local(a, b)
    return bool(np.allclose(c, esperado, rtol=rtol, atol=atol))
