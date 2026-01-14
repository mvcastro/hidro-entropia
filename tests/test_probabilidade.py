from typing import Sequence

import numpy as np
import pandas as pd
from numpy.typing import NDArray

from hidro_entropia.normalizacao import Intervalos
from hidro_entropia.probabilidade import calcula_frequencia


class DummyNormalizacao:
    def normaliza(
        self, serie: Sequence[float] | pd.Series | NDArray[np.floating]
    ) -> NDArray[np.floating]:
        # Simula normalização para [0, 1]
        return np.array(serie) / max(serie)

    def intervalos(self, num_intervalos: int = 20) -> Intervalos:
        # Simula 2 intervalos: [0, 0.5), [0.5, 1]
        return [(0, 0.5), (0.5, 1)]


def test_calcula_frequencia():
    serie: list[float] = [0, 0.25, 0.5, 0.75, 1]
    normalizacao = DummyNormalizacao()
    frequencias = calcula_frequencia(serie, normalizacao, num_intervalos=2)
    # Espera-se que 2 valores estejam no primeiro intervalo e 3 no segundo
    assert frequencias == [2, 3]
