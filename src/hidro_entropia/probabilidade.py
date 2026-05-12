from typing import Sequence

import numpy as np
from numpy.typing import NDArray

from .intervalos_histograma import EstimadorIntervalosHistograma

ValoresDoIntervalo = list[tuple[float, float]]


def classifica_valores_da_serie(
    serie: NDArray[np.floating], intervalos: ValoresDoIntervalo
) -> NDArray[np.integer]:
    bordas: list[float] = []
    for idx, intervalo in enumerate(intervalos):
        if idx < len(intervalos) - 1:
            bordas.append(intervalo[1])
        else:
            bordas.append(1.1 * intervalo[1])
    serie_classificada = np.digitize(serie, bordas)
    return serie_classificada


def calcula_probabilidade_de_serie_classificada(
    serie_classificada: NDArray[np.integer],
) -> dict[int, float]:
    unique_pairs, counts = np.unique(serie_classificada, return_counts=True)
    probabilidades = counts / len(serie_classificada)
    return dict(zip(unique_pairs.tolist(), probabilidades.tolist()))


def calcula_probabilidade(
    serie: NDArray[np.floating],
    estimador_intervalos: EstimadorIntervalosHistograma,
) -> dict[int, float]:
    serie_classificada = classifica_valores_da_serie(
        serie, estimador_intervalos.calcula_intervalos_histograma(np.array(serie))
    )
    return calcula_probabilidade_de_serie_classificada(serie_classificada)


def calcula_probabilidade_conjunta_series_classificadas(
    series_classificadas: Sequence[NDArray[np.integer]],
) -> dict[tuple[int, ...], float]:
    # 1. Empilhar os arrays como colunas (formato [par, par, ...])
    pairs = np.vstack(series_classificadas).T
    # Encontrar pares únicos e suas frequências
    unique_pairs, counts = np.unique(pairs, axis=0, return_counts=True)
    probabilidades = counts / len(series_classificadas[0])
    return dict(zip([tuple(i) for i in unique_pairs.tolist()], probabilidades.tolist()))


def calcula_probabilidade_conjunta(
    series: Sequence[NDArray[np.floating]],
    estimador_intervalos: EstimadorIntervalosHistograma,
) -> dict[tuple[int, ...], float]:
    tamanho_serie = len(series[0])
    for serie in series:
        if len(serie) != tamanho_serie:
            raise ValueError("Todas as séries devem ter o mesmo tamanho.")

    series_classificadas = [
        classifica_valores_da_serie(
            serie,
            estimador_intervalos.calcula_intervalos_histograma(np.array(serie)),
        )
        for serie in series
    ]

    return calcula_probabilidade_conjunta_series_classificadas(series_classificadas)
