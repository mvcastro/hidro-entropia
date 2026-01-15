import numpy as np

from hidro_entropia.normalizacao import Normalizacao, NormalizacaoMinMax_0_1, NormalizacaoMinMax_1_1
from hidro_entropia.probabilidade import calcula_frequencia


def test_normalizacao_intervalo_menos_um_um():
    serie = [1, 2, 3, 4, 5]
    resultado = NormalizacaoMinMax_1_1().normaliza(serie)
    assert np.allclose(resultado, [-1, -0.5, 0, 0.5, 1])


def test_normalizacao_0_1():
    serie = [1, 2, 3, 4, 5]
    resultado = NormalizacaoMinMax_0_1().normaliza(serie)
    assert np.allclose(resultado, [0, 0.25, 0.5, 0.75, 1])


# Testes de frequência com NormalizacaoMinMax_0_1

def test_calcula_frequencia(normalizacao_min_max_0_1: Normalizacao):
    serie: list[float] = [0, 0.25, 0.5, 0.75, 1]
    normalizacao = normalizacao_min_max_0_1
    frequencias = calcula_frequencia(serie, normalizacao, num_intervalos=2)
    # Espera-se que 2 valores estejam no primeiro intervalo e 3 no segundo
    assert frequencias == [2, 3]