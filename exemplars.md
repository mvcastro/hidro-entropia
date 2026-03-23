# Code Exemplars for hidro-entropia

This document identifies high-quality, representative code examples from the hidro-entropia codebase. These exemplars demonstrate our coding standards and patterns for maintaining consistency in this Python library focused on hydrological entropy calculations.

## Table of Contents
- [Protocol Implementations](#protocol-implementations)
- [Class Implementations](#class-implementations)
- [Function Implementations](#function-implementations)
- [Test Fixtures](#test-fixtures)

## Protocol Implementations

### EstimadorIntervalosHistograma Protocol
**File**: [src/hidro_entropia/intervalos_histograma.py](src/hidro_entropia/intervalos_histograma.py)  
**Description**: Defines a protocol for histogram interval estimators, enabling pluggable implementations.  
**Key Details**: Uses typing.Protocol for interface definition, demonstrates clean abstraction for different bin estimation methods.  
**Code Snippet**:
```python
class EstimadorIntervalosHistograma(Protocol):
    def calcula_intervalos_histograma(
        self, serie: NDArray[np.floating]
    ) -> list[tuple[float, float]]: ...
```

### Normalizacao Protocol
**File**: [src/hidro_entropia/normalizacao.py](src/hidro_entropia/normalizacao.py)  
**Description**: Protocol for data normalization strategies, allowing flexible normalization approaches.  
**Key Details**: Includes both normalization and interval generation methods, shows proper protocol design with multiple methods.  
**Code Snippet**:
```python
class Normalizacao(Protocol):
    def normaliza(
        self, serie: Sequence[float] | pd.Series | NDArray[np.floating]
    ) -> NDArray[np.floating]: ...

    def intervalos(self, num_intervalos: int = 20) -> Intervalos: ...
```

## Class Implementations

### SerieTemporalNormalizada Class
**File**: [src/hidro_entropia/serie_temporal.py](src/hidro_entropia/serie_temporal.py)  
**Description**: Wrapper class for normalized time series data with histogram interval access.  
**Key Details**: Demonstrates composition with normalization protocol, clean initialization, and method delegation.  
**Code Snippet**:
```python
class SerieTemporalNormalizada:
    def __init__(self, serie: NDArray[np.floating], normalizacao: Normalizacao) -> None:
        self.serie = serie
        self.serie_normalizada = normalizacao.normaliza(serie)
        self.normalizacao = normalizacao

    def intervalos_histograma(
        self, num_intervalos: int = 20
    ) -> list[tuple[float, float]]:
        return self.normalizacao.intervalos(num_intervalos=num_intervalos)
```

### NormalizacaoMinMax_1_1 Class
**File**: [src/hidro_entropia/normalizacao.py](src/hidro_entropia/normalizacao.py)  
**Description**: Implementation of MinMax normalization to [-1, 1] range.  
**Key Details**: Well-documented with docstrings, proper type hints, and clear mathematical implementation.  
**Code Snippet**:
```python
class NormalizacaoMinMax_1_1:
    def normaliza(self, serie: Sequence[float] | pd.Series) -> NDArray[np.floating]:
        """
        Normaliza uma série numérica para o intervalo [-1, 1].
        ...
        """
        valor_min = np.min(serie)
        valor_max = np.max(serie)
        return (2 * (serie - valor_min) / (valor_max - valor_min)) - 1
```

## Function Implementations

### calcula_frequencia_conjunta Function
**File**: [src/hidro_entropia/probabilidade.py](src/hidro_entropia/probabilidade.py)  
**Description**: Calculates joint frequencies for multiple time series using histogram intervals.  
**Key Details**: Complex algorithm with input validation, uses itertools.product for cartesian combinations, demonstrates numpy vectorized operations.  
**Code Snippet**:
```python
def calcula_frequencia_conjunta(
    series: Sequence[NDArray[np.floating]],
    estimador_intervalos: EstimadorIntervalosHistograma,
) -> list[int]:
    frequencias: list[int] = []

    tamanho_serie = len(series[0])
    for serie in series:
        if len(serie) != tamanho_serie:
            raise ValueError("Todas as séries devem ter o mesmo tamanho.")
    # ... rest of implementation
```

### entropia Function
**File**: [src/hidro_entropia/entropia.py](src/hidro_entropia/entropia.py)  
**Description**: Computes Shannon entropy from frequency data.  
**Key Details**: Clean mathematical implementation with zero-probability handling, uses list comprehension for readability.  
**Code Snippet**:
```python
def entropia(freqs: Sequence[float], base_log: int = 2) -> float:
    probabilidades = propabilidade_discretizada(freqs)
    return -1 * sum(
        [px * log(px, base_log) if px != 0.0 else 0.0 for px in probabilidades]
    )
```

## Test Fixtures

### Dummy Estimator Classes
**File**: [tests/conftest.py](tests/conftest.py)  
**Description**: Provides mock implementations for testing histogram estimators.  
**Key Details**: Clean fixture classes with consistent naming, implements protocol correctly for testing.  
**Code Snippet**:
```python
class DummyEstimador4Intervalos:
    def calcula_intervalos_histograma(
        self, serie: NDArray[np.floating]
    ) -> list[tuple[float, float]]:
        # Retorna 4 intervalos fixos: [0, 0.25), [0.25, 0.5), [0.5, 0.75), [0.75, 1]
        return [(0, 0.25), (0.25, 0.5), (0.5, 0.75), (0.75, 1)]
```

### Pytest Fixtures
**File**: [tests/conftest.py](tests/conftest.py)  
**Description**: Defines reusable test fixtures for estimators and normalizers.  
**Key Details**: Proper pytest fixture syntax, type hints, and clear naming conventions.  
**Code Snippet**:
```python
@pytest.fixture
def estimador_4intervalos() -> DummyEstimador4Intervalos:
    return DummyEstimador4Intervalos()
```

## Conclusion

These exemplars represent the high-quality patterns in hidro-entropia. When implementing new features, reference these examples for:
- Protocol-based design for extensibility
- Clean class structures with proper composition
- Well-documented functions with input validation
- Comprehensive test fixtures for reliable testing

Maintain consistency by following the demonstrated patterns of type hints, Portuguese naming, and numpy-based computations.</content>
<parameter name="filePath">c:\Users\marco\Documents\VisualStudioCode\hidro-entropia\exemplars.md