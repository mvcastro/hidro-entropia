import numpy as np

from hidro_entropia.entropia import entropia_conjunta, entropia_marginal, informacao_mutua, informacao_mutua_v2, informacao_mutua_v2

from .conftest import DummyEstimador2Intervalos


def test_entropia_marginal_binaria_igualdade_entropia_maxima(
    estimador_2intervalos: DummyEstimador2Intervalos,
):
    serie = np.array([0.1, 0.2, 0.8, 0.9])
    resultado = entropia_marginal(serie, estimador_2intervalos)
    assert np.allclose(resultado, 1.0, rtol=1e-8)


def test_entropia_marginal_binaria_igualdade_entropia_maxima2(
    estimador_2intervalos: DummyEstimador2Intervalos,
):
    serie = np.array([0.15, 0.75, 0.25, 0.85])
    resultado = entropia_marginal(serie, estimador_2intervalos)
    assert np.allclose(resultado, 1.0, rtol=1e-8)


def test_entropia_marginal_toda_em_um_intervalo_retornado_zero(
    estimador_2intervalos: DummyEstimador2Intervalos,
):
    serie = np.array([0.1, 0.2, 0.3, 0.4])
    resultado = entropia_marginal(serie, estimador_2intervalos)
    assert np.allclose(resultado, 0.0, rtol=1e-8)
    
def test_entropia_conjunta(
    estimador_2intervalos: DummyEstimador2Intervalos,
):
    serie_x = np.array([0.1, 0.2, 0.8, 0.9])
    serie_y = np.array([0.15, 0.25, 0.7, 0.85])
    # estimador_2intervalos: retorna 2 intervalos fixos: [0, 0.5), [0.5, 1]
    # serie_x classificada: np.array([0, 0, 1, 1]) -> P(0) = P(1) = 0.5
    # serie_y classificada: np.array([0, 0, 1, 1]) -> P(0) = P(1) = 0.5

    # Probabilidade Conjunta = {(0, 0): 0.5, (1, 1): 0.5}
    # Entropia Conjunta =  - [0.5 x log(0.5) + 0.5 x log(0.5)] = 1
    resultado = entropia_conjunta([serie_x, serie_y], estimador_2intervalos)
    assert np.allclose(resultado, 1.0, rtol=1e-10)

def test_informacao_mutua_entre_series_independentes_e_zero(
    estimador_2intervalos: DummyEstimador2Intervalos,
):
    serie_x = np.array([0.1, 0.2, 0.8, 0.9])
    serie_y = np.array([0.15, 0.75, 0.25, 0.85])
    # estimador_2intervalos: retorna 2 intervalos fixos: [0, 0.5), [0.5, 1]
    # serie_x classificada: np.array([0, 0, 1, 1]) -> P(0) = P(1) = 0.5
    # serie_y classificada: np.array([0, 1, 0, 1]) -> P(0) = P(1) = 0.5
    # probabilidade_conjunta = {(0, 0): 0.25, (0, 1): 0.25, (1, 0): 0.25, (1, 1): 0.25}
    # Informação Mútua = 0.25 x log(0.25 / (0.5 x 0.5))
    resultado = informacao_mutua([serie_x, serie_y], estimador_2intervalos)
    resultado2 = informacao_mutua_v2((serie_x, serie_y), estimador_2intervalos)

    assert np.allclose(resultado, 0.0, rtol=1e-10)
    assert np.allclose(resultado2, 0.0, rtol=1e-10)


def test_informacao_mutua_entre_series_correlacionadas_e_um(
    estimador_2intervalos: DummyEstimador2Intervalos,
):
    serie_x = np.array([0.1, 0.2, 0.8, 0.9])
    serie_y = np.array([0.15, 0.25, 0.7, 0.85])
    # estimador_2intervalos: retorna 2 intervalos fixos: [0, 0.5), [0.5, 1]
    # serie_x classificada: np.array([0, 0, 1, 1]) -> P(0) = P(1) = 0.5
    # serie_y classificada: np.array([0, 0, 1, 1]) -> P(0) = P(1) = 0.5

    # probabilidade_conjunta = {(0, 0): 0.5, (1, 1): 0.5}
    # Informação Mútua = 0.5 x log(0.5 / (0.5 x 0.5)) + 0.5 x log(0.5 / (0.5 x 0.5)) = 1
    resultado = informacao_mutua([serie_x, serie_y], estimador_2intervalos)
    resultado2 = informacao_mutua_v2((serie_x, serie_y), estimador_2intervalos)

    assert np.allclose(resultado, 1.0, rtol=1e-8)
    assert np.allclose(resultado2, 1.0, rtol=1e-10)


def test_informacao_mutua_tres_series_correlacionadas_e_dois(
    estimador_2intervalos: DummyEstimador2Intervalos,
):
    serie_x = np.array([0.1, 0.2, 0.8, 0.9])
    serie_y = np.array([0.1, 0.2, 0.8, 0.9])
    serie_z = np.array([0.1, 0.2, 0.8, 0.9])
    # estimador_2intervalos: retorna 2 intervalos fixos: [0, 0.5), [0.5, 1]
    # serie_x classificada: np.array([0, 0, 1, 1]) -> P(0) = P(1) = 0.5
    # serie_y classificada: np.array([0, 0, 1, 1]) -> P(0) = P(1) = 0.5
    # serie_y classificada: np.array([0, 0, 1, 1]) -> P(0) = P(1) = 0.5

    # probabilidade_conjunta = {(0, 0, 0): 0.5, (1, 1, 1): 0.5}
    # Informação Mútua = 0.5 x log(0.5 / (0.5 x 0.5)) + 0.5 x log(0.5 / (0.5 x 0.5)) = 1

    resultado = informacao_mutua([serie_x, serie_y, serie_z], estimador_2intervalos)

    assert np.allclose(resultado, 2.0, rtol=1e-8)
