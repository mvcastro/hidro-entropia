# Copilot Instructions for hidro-entropia

## Overview
This is a Python library for entropy analysis of hydrological time series data. It calculates Shannon entropy using histogram-based discretization of multiple series.

## Architecture
- **Core Modules**: `entropia.py` (entropy calculation), `probabilidade.py` (joint frequency/probability), `intervalos_histograma.py` (bin estimation), `normalizacao.py` (data normalization), `serie_temporal.py` (normalized series wrapper).
- **Data Flow**: Series → Normalization → Interval Estimation → Joint Frequencies → Probabilities → Entropy.
- **Design**: Uses protocols for pluggable estimators (e.g., Freedman-Diaconis, Sturges) and normalizers (e.g., MinMax to [-1,1]).

## Key Patterns
- **Type Hints**: Use `NDArray[np.floating]` for numpy arrays, `Sequence[float]` for inputs.
- **Normalization**: Always normalize series before processing; see `NormalizacaoMinMax_1_1` in `normalizacao.py`.
- **Interval Estimation**: Choose estimator based on data; Freedman-Diaconis for robust IQR-based bins.
- **Joint Calculations**: For multiple series, use `calcula_frequencia_conjunta` with cartesian product of intervals.
- **Entropy Formula**: Standard Shannon entropy with base 2 (bits); handle zero probabilities explicitly.

## Workflows
- **Build/Install**: Use `uv` for dependencies and building; `uv pip install -e .` for editable install.
- **Tests**: Run `pytest -v -s` with `pythonpath = ["src"]`; fixtures in `conftest.py` provide dummy estimators/normalizers.
- **Debug**: Print intermediate frequencies/probabilities; use numpy's vectorized ops for performance.

## Conventions
- **Naming**: Portuguese for variables/comments (e.g., `probabilidades`, `frequencias`); English for classes/functions.
- **Imports**: Relative imports within package; absolute for external (numpy, pandas).
- **Error Handling**: Raise `ValueError` for mismatched series lengths in joint calculations.
- **Dependencies**: Core on numpy/pandas; dev on pytest.

Reference: `src/hidro_entropia/` for implementations, `tests/` for usage examples.</content>
<parameter name="filePath">c:\Users\marco\Documents\VisualStudioCode\hidro-entropia\.github\copilot-instructions.md