from math import log
from typing import Sequence

import numpy as np
from numpy.typing import NDArray

from hidro_entropia.intervalos_histograma import EstimadorIntervalosHistograma

from .probabilidade import (
    calcula_probabilidade,
    calcula_probabilidade_conjunta_series_classificadas,
    calcula_probabilidade_serie_classificada,
    classifica_valores_da_serie,
)


def entropia(probabilidades: Sequence[float], base_log: int = 2) -> float:
    return -1 * sum(
        [px * log(px, base_log) if px != 0.0 else 0.0 for px in probabilidades]
    )


def entropia_marginal(
    serie: NDArray[np.floating],
    estimador_intervalos: EstimadorIntervalosHistograma,
    base_log: int = 2,
) -> float:
    probabilidades = calcula_probabilidade(serie, estimador_intervalos)
    return entropia(list(probabilidades.values()), base_log)


def informacao_mutua(
    series: Sequence[NDArray[np.floating]],
    estimador_intervalos: EstimadorIntervalosHistograma,
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
        calcula_probabilidade_serie_classificada(serie)
        for serie in series_classificadas
    ]
    valor_informacao_mutua = 0.0
    for chaves, prob_conjunta in dict_probs_conjuntas.items():
        probs_marginais: list[float] = [
            dict_probs_marginais[idx][chave] for idx, chave in enumerate(chaves)
        ]
        produto_marginais = np.prod(probs_marginais)
        valor_informacao_mutua += (
            prob_conjunta * log(prob_conjunta / produto_marginais, 2)
            if produto_marginais > 0
            else 0.0
        )

    return valor_informacao_mutua
