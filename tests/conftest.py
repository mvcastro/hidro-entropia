from typing import Sequence

import numpy as np
import pandas as pd
import pytest
from numpy.typing import NDArray

from hidro_entropia.normalizacao import Intervalos


class DummyEstimador2Intervalos:
    def calcula_intervalos_histograma(
        self, serie: NDArray[np.floating]
    ) -> list[tuple[float, float]]:
        # Retorna 2 intervalos fixos: [0, 0.5), [0.5, 1]
        return [(0, 0.5), (0.5, 1)]


class DummyEstimador4Intervalos:
    def calcula_intervalos_histograma(
        self, serie: NDArray[np.floating]
    ) -> list[tuple[float, float]]:
        # Retorna 4 intervalos fixos: [0, 0.25), [0.25, 0.5), [0.5, 0.75), [0.75, 1]
        return [(0, 0.25), (0.25, 0.5), (0.5, 0.75), (0.75, 1)]


class DummyNormalizacaoMinMax_0_1:
    def normaliza(
        self, serie: Sequence[float] | pd.Series | NDArray[np.floating]
    ) -> NDArray[np.floating]:
        # Simula normalização para [0, 1]
        return np.array(serie) / max(serie)

    def intervalos(self, num_intervalos: int = 20) -> Intervalos:
        # Simula 2 intervalos: [0, 0.5), [0.5, 1]
        return [(0, 0.5), (0.5, 1)]


@pytest.fixture
def estimador_2intervalos() -> DummyEstimador2Intervalos:
    return DummyEstimador2Intervalos()


@pytest.fixture
def estimador_4intervalos() -> DummyEstimador4Intervalos:
    return DummyEstimador4Intervalos()


@pytest.fixture
def normalizacao_min_max_0_1() -> DummyNormalizacaoMinMax_0_1:
    return DummyNormalizacaoMinMax_0_1()


@pytest.fixture
def tres_series_longas() -> list[NDArray[np.floating]]:
    # Três séries normalizadas para [0, 1] com tamanho maior (12 valores)
    # e distribuição nos diversos intervalos
    serie1 = np.array([0.1, 0.2, 0.3, 0.4, 0.55, 0.6, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95])
    serie2 = np.array(
        [0.15, 0.25, 0.35, 0.45, 0.5, 0.65, 0.72, 0.78, 0.82, 0.88, 0.92, 0.98]
    )
    serie3 = np.array(
        [0.05, 0.22, 0.38, 0.48, 0.52, 0.62, 0.68, 0.76, 0.81, 0.87, 0.91, 0.96]
    )
    return [serie1, serie2, serie3]