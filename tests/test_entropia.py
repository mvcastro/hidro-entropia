import numpy as np
import pytest

from hidro_entropia.entropia import entropia_marginal, informacao_mutua

from .conftest import DummyEstimador2Intervalos


def test_entropia_marginal_binaria_igualdade_entropia_maxima(
    estimador_2intervalos: DummyEstimador2Intervalos,
):
    serie = np.array([0.1, 0.2, 0.8, 0.9])
    resultado = entropia_marginal(serie, estimador_2intervalos)
    assert pytest.approx(resultado, rel=1e-8) == 1.0


def test_entropia_marginal_toda_em_um_intervalo_retornado_zero(
    estimador_2intervalos: DummyEstimador2Intervalos,
):
    serie = np.array([0.1, 0.2, 0.3, 0.4])
    resultado = entropia_marginal(serie, estimador_2intervalos)
    assert pytest.approx(resultado, rel=1e-8) == 0.0


def test_informacao_mutua_entre_series_independentes_e_zero(
    estimador_2intervalos: DummyEstimador2Intervalos,
):
    serie_x = np.array([0.1, 0.2, 0.8, 0.9])
    serie_y = np.array([0.15, 0.75, 0.25, 0.85])

    resultado = informacao_mutua([serie_x, serie_y], estimador_2intervalos)

    assert pytest.approx(resultado, abs=1e-10) == 0.0


def test_informacao_mutua_entre_series_correlacionadas_e_um(
    estimador_2intervalos: DummyEstimador2Intervalos,
):
    serie_x = np.array([0.1, 0.2, 0.8, 0.9])
    serie_y = np.array([0.15, 0.25, 0.7, 0.85])

    resultado = informacao_mutua([serie_x, serie_y], estimador_2intervalos)

    assert pytest.approx(resultado, rel=1e-8) == 1.0


def test_informacao_mutua_tres_series_correlacionadas_e_dois(
    estimador_2intervalos: DummyEstimador2Intervalos,
):
    serie_x = np.array([0.1, 0.2, 0.8, 0.9])
    serie_y = np.array([0.1, 0.2, 0.8, 0.9])
    serie_z = np.array([0.1, 0.2, 0.8, 0.9])

    resultado = informacao_mutua([serie_x, serie_y, serie_z], estimador_2intervalos)

    assert pytest.approx(resultado, rel=1e-8) == 2.0
