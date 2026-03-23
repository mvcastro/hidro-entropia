from math import log
from typing import Sequence

import numpy as np
from numpy.typing import NDArray

from hidro_entropia.intervalos_histograma import EstimadorIntervalosHistograma

from .probabilidade import (
    calcula_frequencia_conjunta_v2,
    calcula_frequencia_v2,
    propabilidade_discretizada,
)


def entropia(freqs: Sequence[float], base_log: int = 2) -> float:
    probabilidades = propabilidade_discretizada(freqs)
    return -1 * sum(
        [px * log(px, base_log) if px != 0.0 else 0.0 for px in probabilidades]
    )


def entropia_marginal(
    serie: NDArray[np.floating],
    estimador_intervalos: EstimadorIntervalosHistograma,
    base_log: int = 2,
) -> float:
    frequencias = calcula_frequencia_v2(serie, estimador_intervalos)
    return entropia(list(frequencias.values()), base_log)


def informacao_mutua(
    series: Sequence[NDArray[np.floating]],
    estimador_intervalos: EstimadorIntervalosHistograma,
) -> float:
    dict_probs_conjuntas = calcula_frequencia_conjunta_v2(series, estimador_intervalos)
    dict_probs_marginais = [
        calcula_frequencia_v2(serie, estimador_intervalos) for serie in series
    ]
    valor_informacao_mutua = 0.0
    for chaves, prob_conjunta in dict_probs_conjuntas.items():
        probs_marginais: list[float] = [dict_probs_marginais[idx][chave] for idx, chave in enumerate(chaves)]
        produto_marginais = np.prod(probs_marginais)
        valor_informacao_mutua += (
            prob_conjunta * log(prob_conjunta / produto_marginais, 2)
            if produto_marginais > 0
            else 0.0
        )

    return valor_informacao_mutua


if __name__ == "__main__":
    ex = entropia([80, 20])
    ey = entropia([25, 75])
    e_xy = entropia([10, 70, 15, 5])
    print(ex + ey - e_xy)
