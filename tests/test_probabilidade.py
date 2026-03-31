import numpy as np
import pytest

from hidro_entropia.probabilidade import (
    calcula_probabilidade,
    calcula_probabilidade_conjunta,
    calcula_probabilidade_conjunta_series_classificadas,
    calcula_probabilidade_serie_classificada,
    classifica_valores_da_serie,
)

from .conftest import DummyEstimador2Intervalos, DummyEstimador4Intervalos


def test_classifica_valores_da_serie_com_intervalos_binarios():
    serie = np.array([0.1, 0.4, 0.5, 0.6, 1.0])
    intervalos: list[tuple[float, float]] = [(0, 0.5), (0.5, 1)]
    resultado = classifica_valores_da_serie(serie, intervalos)

    assert resultado.tolist() == [0, 0, 1, 1, 2]

def test_classifica_valores_da_serie_com_intervalos_binarios2():
    serie = np.array([0.0, 0.1, 0.4, 0.5, 0.6, 1.0])
    intervalos: list[tuple[float, float]] = [(0, 0.5), (0.5, 1)]
    resultado = classifica_valores_da_serie(serie, intervalos)

    assert resultado.tolist() == [0, 0, 0, 1, 1, 2]


def test_calcula_probabilidade_serie_classificada_retorna_probabilidades_corretas():
    serie_classificada = np.array([0, 0, 1, 2, 2, 2])
    resultado = calcula_probabilidade_serie_classificada(serie_classificada)
    esperado = {0: 2 / 6, 1: 1 / 6, 2: 3 / 6}
    assert resultado == esperado


def test_calcula_probabilidade_com_estimador_2intervalos(
    estimador_2intervalos: DummyEstimador2Intervalos,
):
    serie = np.array([0.1, 0.4, 0.6, 0.8])
    resultado = calcula_probabilidade(serie, estimador_2intervalos)
    esperado = {0: 0.5, 1: 0.5}
    assert resultado == esperado


def test_calcula_probabilidade_conjunta_series_classificadas_retornam_contagens():
    series_classificadas = [
        np.array([0, 0, 1, 1]),
        np.array([0, 1, 0, 1]),
    ]
    resultado = calcula_probabilidade_conjunta_series_classificadas(
        series_classificadas
    )
    esperado = {
        (0, 0): 1,
        (0, 1): 1,
        (1, 0): 1,
        (1, 1): 1,
    }
    assert resultado == esperado


def test_calcula_probabilidade_conjunta_com_duas_series(
    estimador_2intervalos: DummyEstimador2Intervalos,
):
    serie1 = np.array([0.1, 0.4, 0.6, 0.8])
    serie2 = np.array([0.2, 0.3, 0.7, 0.9])

    resultado = calcula_probabilidade_conjunta([serie1, serie2], estimador_2intervalos)

    esperado = {(0, 0): 2, (1, 1): 2}
    assert resultado == esperado


def test_calcula_probabilidade_conjunta_raises_value_error_por_tamanho_diferente(
    estimador_2intervalos: DummyEstimador2Intervalos,
):
    serie1 = np.array([0.1, 0.4, 0.6])
    serie2 = np.array([0.2, 0.3])

    with pytest.raises(ValueError):
        calcula_probabilidade_conjunta([serie1, serie2], estimador_2intervalos)


def test_calcula_probabilidade_conjunta_tres_series_4_intervalos(
    estimador_4intervalos: DummyEstimador4Intervalos,
    tres_series_longas: list[np.ndarray],
):
    resultado = calcula_probabilidade_conjunta(
        tres_series_longas, estimador_4intervalos
    )

    assert resultado[(0, 0, 0)] == 1
    assert resultado[(0, 1, 0)] == 1
    assert resultado[(1, 1, 1)] == 2
    assert resultado[(2, 2, 2)] == 3
    assert resultado[(3, 3, 3)] == 5
    assert sum(resultado.values()) == 12
