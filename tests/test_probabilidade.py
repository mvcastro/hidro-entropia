import numpy as np
import pytest

from hidro_entropia.entropia import entropia
from hidro_entropia.probabilidade import calcula_frequencia_conjunta

from .conftest import DummyEstimador4Intervalos


def test_calcula_ia_conjunta_tres_series_maiores(
    estimador_4intervalos: DummyEstimador4Intervalos,
    tres_series_longas: list[np.ndarray],
):
    # Três séries normalizadas para [0, 1] com tamanho maior (12 valores)
    # e distribuição nos diversos intervalos
    series = tres_series_longas
    estimador = estimador_4intervalos
    frequencias = calcula_frequencia_conjunta(series, estimador)
    entropia_conjunta = entropia(frequencias)
    valor_esperado = pytest.approx(2.0546, rel=1e-4) # pyright: ignore[reportUnknownMemberType]
    assert entropia_conjunta == valor_esperado
