import numpy as np
import pytest
from numpy.typing import NDArray

from hidro_entropia.probabilidade import calcula_frequencia_conjunta


class DummyEstimadorIntervalosHistograma:
    def calcula_intervalos_histograma(
        self, serie: NDArray[np.floating]
    ) -> list[tuple[float, float]]:
        # Retorna 2 intervalos fixos: [0, 0.5), [0.5, 1]
        return [(0, 0.5), (0.5, 1)]


# ...existing code...


def test_calcula_frequencia_conjunta_uma_serie():
    # Uma série normalizada para [0, 1]
    serie = np.array([0.1, 0.4, 0.6, 0.8])
    series = [serie]
    estimador = DummyEstimadorIntervalosHistograma()
    frequencias = calcula_frequencia_conjunta(series, estimador)
    # Espera-se 2 intervalos: [0,0.5) e [0.5,1)
    # [0,0.5): 2 elementos (0.1, 0.4)
    # [0.5,1): 2 elementos (0.6, 0.8)
    assert frequencias == [2, 2]


def test_calcula_frequencia_conjunta_duas_series():
    # Duas séries normalizadas para [0, 1]
    serie1 = np.array([0.1, 0.4, 0.6, 0.8])
    serie2 = np.array([0.2, 0.3, 0.7, 0.9])
    series = [serie1, serie2]
    estimador = DummyEstimadorIntervalosHistograma()
    frequencias = calcula_frequencia_conjunta(series, estimador)
    # Espera-se 4 combinações de intervalos (2x2)
    # [0,0.5)x[0,0.5): 2 elementos (0.1,0.2) e (0.4,0.3)
    # [0,0.5)x[0.5,1): 0 elementos
    # [0.5,1)x[0,0.5): 0 elementos
    # [0.5,1)x[0.5,1): 2 elementos (0.6,0.7) e (0.8,0.9)
    assert frequencias == [2, 0, 0, 2]


def test_calcula_frequencia_conjunta_raises_value_error():
    serie1 = np.array([0.1, 0.4, 0.6])
    serie2 = np.array([0.2, 0.3])  # tamanho diferente
    series = [serie1, serie2]
    estimador = DummyEstimadorIntervalosHistograma()
    with pytest.raises(ValueError):
        calcula_frequencia_conjunta(series, estimador)


def test_calcula_frequencia_conjunta_tres_series():
    # Três séries normalizadas para [0, 1]
    serie1 = np.array([0.1, 0.4, 0.6, 0.8])
    serie2 = np.array([0.2, 0.3, 0.7, 0.9])
    serie3 = np.array([0.05, 0.45, 0.65, 0.85])
    series = [serie1, serie2, serie3]
    estimador = DummyEstimadorIntervalosHistograma()
    frequencias = calcula_frequencia_conjunta(series, estimador)
    # Espera-se 8 combinações de intervalos (2x2x2)
    # Para cada posição, todos os valores estão no mesmo intervalo:
    # [0,0.5)x[0,0.5)x[0,0.5): 2 elementos (índices 0 e 1)
    # [0.5,1)x[0.5,1)x[0.5,1): 2 elementos (índices 2 e 3)
    # Os demais casos não têm elementos
    assert frequencias == [2, 0, 0, 0, 0, 0, 0, 2]
