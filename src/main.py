import timeit
from math import e
from typing import TypedDict, cast

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pprint
from numpy.typing import NDArray
from scipy.optimize import curve_fit

from hidro_entropia.entropia import (
    entropia_conjunta,
    entropia_marginal,
    entropy_knn,
    informacao_mutua,
    informacao_mutua_v2,
)
from hidro_entropia.intervalos_histograma import (
    EstimadorDoane,
    EstimadorFreedmanDiaconis,
    EstimadorIntervalosHistograma,
    EstimadorRice,
    EstimadorScott,
    EstimadorSqrt,
    EstimadorSturges,
)


class DummyEstimador2Intervalos:
    def calcula_intervalos_histograma(
        self, serie: NDArray[np.floating]
    ) -> list[tuple[float, float]]:
        # Retorna 2 intervalos fixos: [0, 0.5), [0.5, 1]
        return [(0, 0.5), (0.5, 1)]


def benchmarking():
    estimador_2intervalos = DummyEstimador2Intervalos()

    serie_x = np.array([0.1, 0.2, 0.8, 0.9])
    serie_y = np.array([0.15, 0.25, 0.7, 0.85])
    # estimador_2intervalos: retorna 2 intervalos fixos: [0, 0.5), [0.5, 1]
    # serie_x classificada: np.array([0, 0, 1, 1]) -> P(0) = P(1) = 0.5
    # serie_y classificada: np.array([0, 0, 1, 1]) -> P(0) = P(1) = 0.5

    # probabilidade_conjunta = {(0, 0): 0.5, (1, 1): 0.5}
    # Informação Mútua = 0.5 x log(0.5 / (0.5 x 0.5)) + 0.5 x log(0.5 / (0.5 x 0.5)) = 1

    def resultado1():
        resultado = informacao_mutua([serie_x, serie_y], estimador_2intervalos)
        return resultado

    def resultado2():
        resultado = informacao_mutua_v2((serie_x, serie_y), estimador_2intervalos)
        return resultado

    execution_time1 = timeit.timeit(stmt=resultado1, number=10000)
    print(execution_time1)

    execution_time2 = timeit.timeit(stmt=resultado2, number=10000)
    print(execution_time2)


def teste_entropia():

    X = np.array([0.0, 5.2, 0.0, 12.3, 3.1, 0.0, 8.4, 1.2, 0.0, 6.7])
    Y = np.array([0.1, 4.8, 0.0, 10.9, 2.9, 0.0, 7.9, 0.9, 0.0, 6.1])

    print("Intervalos de X:", EstimadorSturges().calcula_intervalos_histograma(X))
    print("Intervalos de Y:", EstimadorSturges().calcula_intervalos_histograma(Y))

    print("ENTROPIA CONJUNTA:")
    resultado = entropia_conjunta((X, Y), EstimadorSturges(), base_log=2)
    print("Resultado calculado por discretização: H(X,Y) ≈", resultado, "bits")

    print("ENTROPIA MARGINAL")
    entropia_marg_x = entropia_marginal(
        serie=X, estimador_intervalos=EstimadorSturges()
    )
    print("Entropia marginal de X:", entropia_marg_x)
    entropia_marg_y = entropia_marginal(
        serie=Y, estimador_intervalos=EstimadorSturges()
    )
    print("Entropia marginal de Y:", entropia_marg_y)


def teste_entropia2():

    X = np.array([0.0, 6.0, 10.0, 4.0, 9.0, 10.0, 0.0, 0.0, 0.0, 0.0])
    Y = np.array([0.1, 4.8, 0.0, 10.9, 2.9, 0.0, 7.9, 0.9, 0.0, 6.1])

    print("Intervalos de X:", EstimadorSturges().calcula_intervalos_histograma(X))
    print("Intervalos de Y:", EstimadorSturges().calcula_intervalos_histograma(Y))

    print("ENTROPIA CONJUNTA:")
    resultado = entropia_conjunta((X, Y), EstimadorSturges(), base_log=2)
    print("Resultado calculado por discretização: H(X,Y) ≈", resultado, "bits")

    print("ENTROPIA MARGINAL")
    entropia_marg_x = entropia_marginal(
        serie=X, estimador_intervalos=EstimadorSturges()
    )
    print("Entropia marginal de X:", entropia_marg_x)
    entropia_marg_y = entropia_marginal(
        serie=Y, estimador_intervalos=EstimadorSturges()
    )
    print("Entropia marginal de Y:", entropia_marg_y)


def test_entropia_knn():
    serie_x = np.array([0.1, 0.2, 0.8, 0.9])
    serie_y = np.array([0.15, 0.25, 0.7, 0.85])

    resultado = entropia_conjunta(
        (serie_x, serie_y), EstimadorFreedmanDiaconis(), base_log=e
    )
    print(resultado)

    # Dados de exemplo
    data = np.column_stack([serie_x, serie_y])
    # data = np.array([serie_x, serie_y])
    H_xy_knn = entropy_knn(data, k=3)
    print(f"H(X,Y,Z) ≈ {H_xy_knn:.3f} nats")


def test_entropia_knn2():
    print("ENTROPIA:")
    X = np.array([0.0, 5.2, 0.0, 12.3, 3.1, 0.0, 8.4, 1.2, 0.0, 6.7])
    Y = np.array([0.1, 4.8, 0.0, 10.9, 2.9, 0.0, 7.9, 0.9, 0.0, 6.1])

    resultado = entropia_conjunta((X, Y), EstimadorSturges(), base_log=e)
    print("Resultado calculado por discretização: H(X,Y) ≈", resultado, "nats")

    # Dados de exemplo
    data = np.column_stack([X, Y])
    data = (data - data.mean(axis=0)) / data.std(axis=0)
    H_XY = entropy_knn(data, k=3)
    print(f"Resultado calculado por KNN: H(X,Y) ≈ {H_XY:.3f} nats")


def test_informacao_mutua():
    print("INFORMAÇÃO MÚTUA:")
    X = np.array([0.0, 5.2, 0.0, 12.3, 3.1, 0.0, 8.4, 1.2, 0.0, 6.7])
    Y = np.array([0.1, 4.8, 0.0, 10.9, 2.9, 0.0, 7.9, 0.9, 0.0, 6.1])

    resultado = informacao_mutua((X, Y), EstimadorSturges(), base_log=e)
    print("Resultado calculada por discretização:", resultado, "nats")

    # Dados de exemplo
    rng = np.random.default_rng(42)

    Xn = (X - X.mean()) / X.std()
    Yn = (Y - Y.mean()) / Y.std()

    # jitter muito pequeno
    Xn += 1e-10 * rng.standard_normal(len(Xn))
    Yn += 1e-10 * rng.standard_normal(len(Yn))

    H_X = entropy_knn(Xn.reshape(-1, 1), k=3)
    H_Y = entropy_knn(Yn.reshape(-1, 1), k=3)

    XY = np.column_stack([Xn, Yn])
    H_XY = entropy_knn(XY, k=3)

    I_XY = H_X + H_Y - H_XY
    print("Resultado calculada por KNN:", I_XY, "nats")


def modelo_potencial(
    x: NDArray[np.floating] | pd.Series, y: NDArray[np.floating] | pd.Series
) -> tuple[NDArray[np.floating], NDArray[np.floating]]:

    def potential_model(
        x: NDArray[np.floating], a: float, b: float
    ) -> NDArray[np.floating]:
        return a * np.power(x, b)

    # Ajustar o modelo (curve_fit)
    params, _ = curve_fit(potential_model, x, y)
    print(f"{params=}")

    a_fit, b_fit = cast(tuple[float, float], params)

    print(f"Modelo ajustado: y = {a_fit:.2f} * x^({b_fit:.2f})")

    x_fit = np.linspace(start=x.min(), stop=x.max(), num=100)
    y_fit = potential_model(x_fit, a_fit, b_fit)

    return x_fit, y_fit


def teste_estacoes_ana():

    class ResultadoEstimador(TypedDict):
        estimador: str
        num_bins: int
        entropia_marginal: float

    df_serie = pd.read_csv(
        "C:/Users/marco.goncalves/Downloads/dados_estacao_02347055_Itaici.csv",
        delimiter=";",
        decimal=",",
    )
    df_serie["Chuva"] = df_serie["Chuva"].interpolate(method="linear")

    print(df_serie[df_serie.Chuva > 999])

    serie = df_serie["Chuva"].to_numpy()

    estimadores_intervalos: list[EstimadorIntervalosHistograma] = [
        EstimadorFreedmanDiaconis(),
        EstimadorSturges(),
        EstimadorSqrt(),
        EstimadorRice(),
        EstimadorDoane(),
        EstimadorScott(),
    ]

    resultados: list[ResultadoEstimador] = []

    for estimador in estimadores_intervalos:
        try:
            intervalos = estimador.calcula_intervalos_histograma(serie)
            entropia_marg = entropia_marginal(
                serie=serie, estimador_intervalos=estimador
            )

            resultados.append(
                {
                    "estimador": estimador.__class__.__name__,
                    "num_bins": len(intervalos),
                    "entropia_marginal": entropia_marg,
                }
            )
        except:
            pass

    print(resultados)
    df_resultados = pd.DataFrame(resultados)
    df_resultados.plot.scatter(x="num_bins", y="entropia_marginal")
    x_fit, y_fit = modelo_potencial(
        x=df_resultados.num_bins, y=df_resultados.entropia_marginal
    )
    print(x_fit)
    plt.plot(x_fit, y_fit, color="red", label="Ajuste Exponencial")
    plt.legend()
    plt.show()


def teste_estacoes_ana_multiplos_intervalos():

    class ResultadoEstimador(TypedDict):
        num_bins: int
        entropia_marginal: float

    class EstimadorIntervalosADefinir:
        def __init__(self, num_bins: int) -> None:
            self.num_bins = num_bins

        def calcula_intervalos_histograma(
            self, serie: NDArray[np.floating]
        ) -> list[tuple[float, float]]:
            intervalos = np.linspace(np.min(serie), np.max(serie), self.num_bins + 1)
            return [
                (float(intervalos[i]), float(intervalos[i + 1]))
                for i in range(len(intervalos) - 1)
            ]

    df_serie = pd.read_csv(
        "C:/Users/marco.goncalves/Downloads/dados_estacao_02347055_Itaici.csv",
        delimiter=";",
        decimal=",",
    )
    df_serie["Chuva"] = df_serie["Chuva"].interpolate(method="linear")
    print("Número de registros da série:", len(df_serie))

    serie = df_serie["Chuva"].to_numpy()

    lista_num_bins = [10 * i for i in range(1, 200)]

    resultados: list[ResultadoEstimador] = []

    for num_bins in lista_num_bins:
        entropia_marg = entropia_marginal(
            serie=serie, estimador_intervalos=EstimadorIntervalosADefinir(num_bins)
        )
        resultados.append({"num_bins": num_bins, "entropia_marginal": entropia_marg})

    pprint.pprint(resultados)
    df_resultados = pd.DataFrame(resultados)
    df_resultados.plot.scatter(x="num_bins", y="entropia_marginal")
    # x_fit, y_fit = modelo_potencial(
    #     x=df_resultados.num_bins, y=df_resultados.entropia_marginal
    # )
    # print(x_fit)
    # plt.plot(x_fit, y_fit, color="red", label="Ajuste Exponencial")
    # plt.legend()
    plt.show()


if __name__ == "__main__":
    # test_entropia_knn2()
    # test_informacao_mutua()
    # teste_entropia2()
    # teste_estacoes_ana()
    teste_estacoes_ana_multiplos_intervalos()
