from typing import Protocol, Sequence

import numpy as np
import pandas as pd
from numpy.typing import NDArray

type Intervalos = list[tuple[float, float]]


def gera_intervalos(
    valor_min: float,
    valor_max: float,
    num_intervalos: int = 20,
) -> Intervalos:
    fracao = (valor_max - valor_min) / num_intervalos

    intervalos = [
        (-1 + i * fracao, -1 + (i + 1) * fracao) for i in range(num_intervalos)
    ]
    return intervalos


class Normalizacao(Protocol):
    def normaliza(
        self, serie: Sequence[float] | pd.Series | NDArray[np.floating]
    ) -> NDArray[np.floating]: ...

    def intervalos(self, num_intervalos: int = 20) -> Intervalos: ...


class NormalizacaoMinMax_1_1:
    def normaliza(self, serie: Sequence[float] | pd.Series) -> NDArray[np.floating]:
        """
        Normaliza uma série numérica para o intervalo [-1, 1].

        Parâmetros:
            serie (NDArray[np.floating]): Array NumPy contendo valores numéricos.

        Retorna:
            NDArray[np.floating]: Array normalizado no intervalo [-1, 1].
        """
        valor_min = np.min(serie)
        valor_max = np.max(serie)
        return (2 * (serie - valor_min) / (valor_max - valor_min)) - 1

    def intervalos(self, num_intervalos: int = 20) -> Intervalos:
        return gera_intervalos(
            valor_min=-1.0, valor_max=1.0, num_intervalos=num_intervalos
        )


class NormalizacaoMinMax_0_1:
    def normaliza(self, serie: Sequence[float] | pd.Series) -> NDArray[np.floating]:
        """
        Normaliza uma série numérica para o intervalo [0, 1].

        Parâmetros:
            serie (NDArray[np.floating]): Array NumPy contendo valores numéricos.

        Retorna:
            NDArray[np.floating]: Array normalizado no intervalo [0, 1].
        """
        valor_min = np.min(serie)
        valor_max = np.max(serie)
        return (serie - valor_min) / (valor_max - valor_min)

    def intervalos(self, num_intervalos: int = 20) -> Intervalos:
        return gera_intervalos(
            valor_min=0.0, valor_max=1.0, num_intervalos=num_intervalos
        )
