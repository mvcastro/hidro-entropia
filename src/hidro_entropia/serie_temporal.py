import numpy as np
from numpy.typing import NDArray

from .normalizacao import Normalizacao


class SerieTemporalNormalizada:
    def __init__(self, serie: NDArray[np.floating], normalizacao: Normalizacao) -> None:
        self.serie = serie
        self.serie_normalizada = normalizacao.normaliza(serie)
        self.normalizacao = normalizacao

    def intervalos_histograma(
        self, num_intervalos: int = 20
    ) -> list[tuple[float, float]]:
        return self.normalizacao.intervalos(num_intervalos=num_intervalos)
