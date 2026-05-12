from math import gamma, log
from typing import Sequence, cast

import numpy as np
from numpy.typing import NDArray
from scipy.special import digamma
from sklearn.neighbors import NearestNeighbors

from hidro_entropia.intervalos_histograma import EstimadorIntervalosHistograma

from .probabilidade import (
    calcula_probabilidade_conjunta_series_classificadas,
    calcula_probabilidade_de_serie_classificada,
    classifica_valores_da_serie,
)


def entropia(probabilidades: Sequence[float], base_log: float = 2) -> float:
    return -1 * sum(
        [px * log(px, base_log) if px != 0.0 else 0.0 for px in probabilidades]
    )


def _entropia_marginal_de_serie_classificada(
    serie_classificada: NDArray[np.integer],
    base_log: float = 2,
) -> float:
    probabilidades = calcula_probabilidade_de_serie_classificada(serie_classificada)
    return entropia(list(probabilidades.values()), base_log)


def entropia_marginal(
    serie: NDArray[np.floating],
    estimador_intervalos: EstimadorIntervalosHistograma,
    base_log: float = 2,
) -> float:
    serie_classificada = classifica_valores_da_serie(
        serie, estimador_intervalos.calcula_intervalos_histograma(np.array(serie))
    )
    return _entropia_marginal_de_serie_classificada(serie_classificada, base_log)


def _entropia_conjunta_de_series_classificadas(
    series_classificadas: list[NDArray[np.integer]], base_log: float = 2
):
    dict_probs_conjuntas = calcula_probabilidade_conjunta_series_classificadas(
        series_classificadas
    )
    return entropia(list(dict_probs_conjuntas.values()), base_log)


def entropia_conjunta(
    series: Sequence[NDArray[np.floating]],
    estimador_intervalos: EstimadorIntervalosHistograma,
    base_log: float = 2,
) -> float:
    series_classificadas = [
        classifica_valores_da_serie(
            serie,
            estimador_intervalos.calcula_intervalos_histograma(np.array(serie)),
        )
        for serie in series
    ]

    return _entropia_conjunta_de_series_classificadas(series_classificadas, base_log)


def informacao_mutua(
    series: Sequence[NDArray[np.floating]],
    estimador_intervalos: EstimadorIntervalosHistograma,
    base_log: float = 2,
):
    series_classificadas = [
        classifica_valores_da_serie(
            serie,
            estimador_intervalos.calcula_intervalos_histograma(np.array(serie)),
        )
        for serie in series
    ]
    dict_probs_conjuntas = calcula_probabilidade_conjunta_series_classificadas(
        series_classificadas
    )
    dict_probs_marginais = [
        calcula_probabilidade_de_serie_classificada(serie)
        for serie in series_classificadas
    ]
    valor_informacao_mutua = 0.0
    for chaves, prob_conjunta in dict_probs_conjuntas.items():
        probs_marginais: list[float] = [
            dict_probs_marginais[idx][chave] for idx, chave in enumerate(chaves)
        ]
        produto_marginais = np.prod(probs_marginais)
        valor_informacao_mutua += (
            prob_conjunta * log(prob_conjunta / produto_marginais, base_log)
            if produto_marginais != 0
            else 0.0
        )
    return valor_informacao_mutua


def informacao_mutua_v2(
    series: tuple[NDArray[np.floating], NDArray[np.floating]],
    estimador_intervalos: EstimadorIntervalosHistograma,
    base_log: float = 2,
):
    series_classificadas = [
        classifica_valores_da_serie(
            serie,
            estimador_intervalos.calcula_intervalos_histograma(np.array(serie)),
        )
        for serie in series
    ]

    Hxy = _entropia_conjunta_de_series_classificadas(series_classificadas, base_log)
    Hx = _entropia_marginal_de_serie_classificada(series_classificadas[0], base_log)
    Hy = _entropia_marginal_de_serie_classificada(series_classificadas[1], base_log)

    return Hx + Hy - Hxy


def entropy_knn(data: NDArray[np.floating], k: int = 5) -> float:
    """
    Estima H(X,Y,Z) usando kNN (Kozachenko–Leonenko)
    data: array (N, 3)
    """
    N, d = cast(tuple[int, int], data.shape)

    # Busca k+1 porque o vizinho mais próximo é o próprio ponto
    nbrs = NearestNeighbors(n_neighbors=k + 1).fit(data)
    distances, _ = nbrs.kneighbors(data)

    # Distância ao k-ésimo vizinho
    eps = distances[:, k]

    volume_unit_ball = np.pi ** (d / 2) / gamma(d / 2 + 1)

    H = digamma(N) - digamma(k) + np.log(volume_unit_ball) + d * np.mean(np.log(eps))

    return cast(float, H)
