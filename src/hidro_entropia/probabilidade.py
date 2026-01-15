from itertools import product
from typing import Sequence

import numpy as np
from numpy.typing import NDArray

from .intervalos_histograma import EstimadorIntervalosHistograma
from .normalizacao import Normalizacao

ValoresDoIntervalo = list[tuple[float, float]]


def calcula_frequencia_conjunta(
    series: Sequence[NDArray[np.floating]],
    estimador_intervalos: EstimadorIntervalosHistograma,
) -> list[int]:
    frequencias: list[int] = []

    tamanho_serie = len(series[0])
    for serie in series:
        if len(serie) != tamanho_serie:
            raise ValueError("Todas as séries devem ter o mesmo tamanho.")

    intervalos = [
        estimador_intervalos.calcula_intervalos_histograma(np.array(serie))
        for serie in series
    ]

    produto_intervalos = product(
        *[tuple(range(len(intervalo))) for intervalo in intervalos]
    )

    for indices_intervalos_series in produto_intervalos:
        frequence = np.ones(tamanho_serie)
        for idx_serie, idx_intervalo_serie in enumerate(indices_intervalos_series):
            serie = series[idx_serie]
            intervalos_serie = intervalos[idx_serie]
            intervalo_serie = intervalos_serie[idx_intervalo_serie]

            if idx_intervalo_serie == len(intervalos_serie) - 1:
                frequence *= serie >= intervalo_serie[0]
            else:
                frequence *= (serie >= intervalo_serie[0]) & (
                    serie < intervalo_serie[1]
                )
        frequencias.append(int(np.sum(frequence)))
    return frequencias



def calcula_frequencia(
    serie: Sequence[float], normalizacao: Normalizacao, num_intervalos: int = 20
) -> list[int]:
    frequencias: list[int] = []
    serie_normalizada = normalizacao.normaliza(serie)
    intervalos = normalizacao.intervalos(num_intervalos=num_intervalos)

    for idx, intervalo in enumerate(intervalos):
        if idx == len(intervalos) - 1:
            frequencias.append(int(np.sum(serie_normalizada >= intervalo[0])))
        else:
            frequencias.append(
                int(
                    np.sum(
                        (serie_normalizada >= intervalo[0])
                        & (serie_normalizada < intervalo[1])
                    )
                )
            )
    return frequencias


def propabilidade_discretizada(frequencias: Sequence[float]) -> list[float]:
    total_dados = sum(frequencias)
    return [cont / total_dados for cont in frequencias]


def probabilidade_conjunta_discretizada(
    probs_x: Sequence[float], probs_y: Sequence[float]
) -> NDArray[np.floating]:
    freq_conjunta = np.outer(probs_x, probs_y)
    return freq_conjunta
