from math import log

from .probabilidade import propabilidade_discretizada


def entropia(freqs: list[float], base_log: int = 2) -> float:
    probabilidades = propabilidade_discretizada(freqs)
    return -1 * sum([px * log(px, base_log) for px in probabilidades])


ex = entropia([80,20])
ey = entropia([25,75])
e_xy = entropia([10,70,15,5])
print(ex + ey - e_xy)