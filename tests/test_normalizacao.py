import numpy as np

from hidro_entropia.normalizacao import NormalizacaoMinMax_0_1, NormalizacaoMinMax_1_1


def test_normalizacao_intervalo_menos_um_um():
    serie = [1, 2, 3, 4, 5]
    resultado = NormalizacaoMinMax_1_1().normaliza(serie)
    assert np.allclose(resultado, [-1, -0.5, 0, 0.5, 1])


def test_normalizacao_0_1():
    serie = [1, 2, 3, 4, 5]
    resultado = NormalizacaoMinMax_0_1().normaliza(serie)
    assert np.allclose(resultado, [0, 0.25, 0.5, 0.75, 1])
