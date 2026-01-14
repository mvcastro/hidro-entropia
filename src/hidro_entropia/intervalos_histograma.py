from typing import Protocol

import numpy as np
from numpy.typing import NDArray


class EstimadorIntervalosHistograma(Protocol):
    def calcula_intervalos_histograma(
        self, serie: NDArray[np.floating]
    ) -> list[tuple[float, float]]: ...


class EstimadorFreedmanDiaconis:
    def calcula_intervalos_histograma(
        self, serie: NDArray[np.floating]
    ) -> list[tuple[float, float]]:
        q75, q25 = np.percentile(serie, [75, 25])
        iqr = q75 - q25
        n = len(serie)
        h = 2 * iqr / (n ** (1 / 3))
        self.num_bins = int(np.ceil((np.max(serie) - np.min(serie)) / h))
        intervalos = np.linspace(np.min(serie), np.max(serie), self.num_bins + 1)
        return [
            (float(intervalos[i]), float(intervalos[i + 1]))
            for i in range(len(intervalos) - 1)
        ]


class EstimadorSturges:
    def calcula_intervalos_histograma(
        self, serie: NDArray[np.floating]
    ) -> list[tuple[float, float]]:
        n = len(serie)
        self.num_bins = int(np.ceil(np.log2(n) + 1))
        intervalos = np.linspace(np.min(serie), np.max(serie), self.num_bins + 1)
        return [
            (float(intervalos[i]), float(intervalos[i + 1]))
            for i in range(len(intervalos) - 1)
        ]


class EstimadorSqrt:
    def calcula_intervalos_histograma(
        self, serie: NDArray[np.floating]
    ) -> list[tuple[float, float]]:
        n = len(serie)
        self.num_bins = int(np.ceil(np.sqrt(n)))
        intervalos = np.linspace(np.min(serie), np.max(serie), self.num_bins + 1)
        return [
            (float(intervalos[i]), float(intervalos[i + 1]))
            for i in range(len(intervalos) - 1)
        ]


class EstimadorRice:
    def calcula_intervalos_histograma(
        self, serie: NDArray[np.floating]
    ) ->list[tuple[float, float]]:
        n = len(serie)
        self.num_bins = int(np.ceil(2 * (n ** (1 / 3))))
        intervalos = np.linspace(np.min(serie), np.max(serie), self.num_bins + 1)
        return [
            (float(intervalos[i]), float(intervalos[i + 1]))
            for i in range(len(intervalos) - 1)
        ]


class EstimadorDoane:
    def calcula_intervalos_histograma(
        self, serie: NDArray[np.floating]
    ) -> list[tuple[float, float]]:
        n = len(serie)
        skewness = (np.mean((serie - np.mean(serie)) ** 3)) / (np.std(serie) ** 3)
        sigma_g1 = np.sqrt((6 * (n - 2)) / ((n + 1) * (n + 3)))
        self.num_bins = int(
            np.ceil(1 + np.log2(n) + np.log2(1 + abs(skewness) / sigma_g1))
        )
        intervalos = np.linspace(np.min(serie), np.max(serie), self.num_bins + 1)
        return [
            (float(intervalos[i]), float(intervalos[i + 1]))
            for i in range(len(intervalos) - 1)
        ]


class EstimadorScott:
    def calcula_intervalos_histograma(
        self, serie: NDArray[np.floating]
    ) -> list[tuple[float, float]]:
        n = len(serie)
        sigma = np.std(serie)
        h = 3.5 * sigma / (n ** (1 / 3))
        self.num_bins = int(np.ceil((np.max(serie) - np.min(serie)) / h))
        intervalos = np.linspace(np.min(serie), np.max(serie), self.num_bins + 1)
        return [
            (float(intervalos[i]), float(intervalos[i + 1]))
            for i in range(len(intervalos) - 1)
        ]


if __name__ == "__main__":
    serie = np.array([0, 0, 0, 1, 2, 3, 3, 4, 5])
    print("Estimador FreedmanDiaconis:")
    estimador_fd = EstimadorFreedmanDiaconis()
    print(estimador_fd.calcula_intervalos_histograma(serie))
    print("Num bins:", estimador_fd.num_bins)
    print("Estimador Sturges:")
    estimador_sturges = EstimadorSturges()
    print(estimador_sturges.calcula_intervalos_histograma(serie))
    print("Num bins:", estimador_sturges.num_bins)
