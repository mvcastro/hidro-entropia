import numpy as np
import pytest

from hidro_entropia.probabilidade import calcula_frequencia_conjunta

from .conftest import DummyEstimador2Intervalos, DummyEstimador4Intervalos


def test_calcula_frequencia_conjunta_uma_serie(
    estimador_2intervalos: DummyEstimador2Intervalos,
):
    # Uma série normalizada para [0, 1]
    serie = np.array([0.1, 0.4, 0.6, 0.8])
    series = [serie]
    estimador = estimador_2intervalos
    frequencias = calcula_frequencia_conjunta(series, estimador)
    # Espera-se 2 intervalos: [0,0.5) e [0.5,1)
    # [0,0.5): 2 elementos (0.1, 0.4)
    # [0.5,1): 2 elementos (0.6, 0.8)
    assert frequencias == [2, 2]


def test_calcula_frequencia_conjunta_duas_series(
    estimador_2intervalos: DummyEstimador2Intervalos,
):
    # Duas séries normalizadas para [0, 1]
    serie1 = np.array([0.1, 0.4, 0.6, 0.8])
    serie2 = np.array([0.2, 0.3, 0.7, 0.9])
    series = [serie1, serie2]
    estimador = estimador_2intervalos
    frequencias = calcula_frequencia_conjunta(series, estimador)
    # Espera-se 4 combinações de intervalos (2x2)
    # [0,0.5)x[0,0.5): 2 elementos (0.1,0.2) e (0.4,0.3)
    # [0,0.5)x[0.5,1): 0 elementos
    # [0.5,1)x[0,0.5): 0 elementos
    # [0.5,1)x[0.5,1): 2 elementos (0.6,0.7) e (0.8,0.9)
    assert frequencias == [2, 0, 0, 2]


def test_calcula_frequencia_conjunta_raises_value_error(
    estimador_2intervalos: DummyEstimador2Intervalos,
):
    serie1 = np.array([0.1, 0.4, 0.6])
    serie2 = np.array([0.2, 0.3])  # tamanho diferente
    series = [serie1, serie2]
    estimador = estimador_2intervalos
    with pytest.raises(ValueError):
        calcula_frequencia_conjunta(series, estimador)


def test_calcula_frequencia_conjunta_tres_series(
    estimador_2intervalos: DummyEstimador2Intervalos,
):
    # Três séries normalizadas para [0, 1]
    serie1 = np.array([0.1, 0.4, 0.6, 0.8])
    serie2 = np.array([0.2, 0.3, 0.7, 0.9])
    serie3 = np.array([0.05, 0.45, 0.65, 0.85])
    series = [serie1, serie2, serie3]
    estimador = estimador_2intervalos
    frequencias = calcula_frequencia_conjunta(series, estimador)
    # Espera-se 8 combinações de intervalos (2x2x2)
    # Para cada posição, todos os valores estão no mesmo intervalo:
    # [0,0.5)x[0,0.5)x[0,0.5): 2 elementos (índices 0 e 1)
    # [0.5,1)x[0.5,1)x[0.5,1): 2 elementos (índices 2 e 3)
    # Os demais casos não têm elementos
    assert frequencias == [2, 0, 0, 0, 0, 0, 0, 2]


def test_calcula_frequencia_conjunta_tres_series_maiores(
    estimador_4intervalos: DummyEstimador4Intervalos,
    tres_series_longas: list[np.ndarray],
):
    # Três séries normalizadas para [0, 1] com tamanho maior (12 valores)
    # e distribuição nos diversos intervalos
    series = tres_series_longas
    estimador = estimador_4intervalos
    frequencias = calcula_frequencia_conjunta(series, estimador)
    # Espera-se 64 combinações de intervalos (4x4x4)
    # Com 4 intervalos: [0,0.25), [0.25,0.5), [0.5,0.75), [0.75,1]
    # Analisando cada posição (série1, série2, série3):
    # Posição 0: (0.1, 0.15, 0.05) -> [0,0.25)x[0,0.25)x[0,0.25)
    # Posição 1: (0.2, 0.25, 0.22) -> [0,0.25)x[0.25,0.5)x[0,0.25)
    # Posição 2: (0.3, 0.35, 0.38) -> [0.25,0.5)x[0.25,0.5)x[0.25,0.5)
    # Posição 3: (0.4, 0.45, 0.48) -> [0.25,0.5)x[0.25,0.5)x[0.25,0.5)
    # Posição 4: (0.55, 0.5, 0.52) -> [0.5,0.75)x[0.5,0.75)x[0.5,0.75)
    # Posição 5: (0.6, 0.65, 0.62) -> [0.5,0.75)x[0.5,0.75)x[0.5,0.75)
    # Posição 6: (0.7, 0.72, 0.68) -> [0.5,0.75)x[0.5,0.75)x[0.5,0.75)
    # Posição 7: (0.75, 0.78, 0.76) -> [0.75,1)x[0.75,1)x[0.75,1)
    # Posição 8: (0.8, 0.82, 0.81) -> [0.75,1)x[0.75,1)x[0.75,1)
    # Posição 9: (0.85, 0.88, 0.87) -> [0.75,1)x[0.75,1)x[0.75,1)
    # Posição 10: (0.9, 0.92, 0.91) -> [0.75,1)x[0.75,1)x[0.75,1)
    # Posição 11: (0.95, 0.98, 0.96) -> [0.75,1)x[0.75,1)x[0.75,1)
    # Resultado esperado (array com 64 elementos, maioria zeros):
    # Índice 0: 1 ocorrência (posição 0)
    # Índice 4: 1 ocorrência (posição 1)
    # Índice 21: 2 ocorrências (posições 2 e 3)
    # Índice 42: 3 ocorrências (posições 4, 5, 6)
    # Índice 63: 5 ocorrências (posições 7, 8, 9, 10, 11)
    print(frequencias)
    esperado = [0] * 64
    esperado[0] = 1
    esperado[4] = 1
    esperado[21] = 2
    esperado[42] = 3
    esperado[63] = 5
    assert frequencias == esperado
