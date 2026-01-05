# Quantum Machine Learning: VQC vs SVM (Iris)

Proyecto de **Quantum Machine Learning (QML)** comparando un baseline clásico (**SVM**) contra un clasificador variacional cuántico (**VQC**) sobre el dataset **Iris** (2 features), usando **Qiskit Machine Learning**.

## Objetivos
- Implementar un baseline clásico (SVM en Scikit-Learn).
- Implementar un modelo cuántico variacional (VQC).
- Comparar desempeño y costo computacional (accuracy, tiempo, tamaño del circuito/parámetros).

## Contenido del repositorio
- `ProyectoQuantumML.ipynb`: notebook principal con instalación, entrenamiento y gráficas.

## Requisitos
- Python recomendado: 3.10+ (probado en 3.13 con venv).
- Dependencias: ver `requirements.txt`.

## Instalación (venv)
```bash
python -m venv .venv
source .venv/bin/activate
pip install -U pip
pip install -r requirements.txt
```

## Ejecución
- Abrir `ProyectoQuantumML.ipynb` en VS Code/Jupyter.
- Ejecutar las celdas en orden.

## Reproducibilidad
- Fija `random_state` y el split estratificado para el dataset.
- El entrenamiento variacional puede variar por inicialización/ruido del optimizador; si comparas métricas, repite varias corridas y reporta media±desv.

## Resultados esperados
- SVM suele obtener alta accuracy en Iris con 2 features.
- VQC en simulación puede ser competitivo, pero suele ser más lento; el resultado depende de `feature_map`, `ansatz`, `reps`, `shots` y el optimizador.

## Notas técnicas
- Se usa un `Sampler` **API V2** y un `pass_manager` para transpilar/decomponer circuitos antes de ejecutarlos en Aer.
