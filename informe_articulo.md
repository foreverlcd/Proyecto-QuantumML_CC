# Proyecto Quantum ML

**Alumnos:** 
* Lucian Neptalí Fernandez Baca Castro
* Oriol Fernando Palacios Durand    
* Jhon Jesus Quispe Machaca  
* Ronaldo Ticona Jancco  
* Dorian Roger Zavala Ttito  

**Afiliación:**  
Universidad Nacional de San Antonio Abad del Cusco  
Departamento de Informática - Ingeniería Informática y de Sistemas

**Palabras clave:** Quantum Machine Learning, VQC, QSVM, QNN, Iris, Qiskit, Computación Cuántica

---

## 1. Introducción y justificación

### 1.1 Motivación
El Quantum Machine Learning (QML) representa la intersección entre computación cuántica y aprendizaje automático, prometiendo ventajas computacionales en problemas específicos. La clasificación de datos es fundamental en ML, y explorar enfoques cuánticos permite entender sus capacidades y limitaciones actuales.

### 1.2 Problema abordado
Este proyecto compara el desempeño de un clasificador clásico (SVM) contra un clasificador variacional cuántico (VQC) utilizando el dataset Iris, enfocándose en métricas de precisión y costo computacional.

### 1.3 Objetivos
- Implementar y evaluar un baseline clásico (SVM)
- Desarrollar e implementar un modelo VQC
- Comparar desempeño, tiempos de entrenamiento y recursos computacionales
- Documentar lecciones prácticas sobre simulación cuántica

### 1.4 Contribuciones
- Análisis comparativo cuantitativo SVM vs VQC
- Implementación reproducible con Qiskit Machine Learning
- Evaluación de limitaciones prácticas en simuladores

---

## 2. Marco teórico y antecedentes

### 2.1 Computación cuántica básica
**Qubits y estados:** Un qubit puede representarse como |ψ⟩ = α|0⟩ + β|1⟩, donde |α|² + |β|² = 1.

**Puertas cuánticas:** Operaciones unitarias que manipulan estados cuánticos (Hadamard, CNOT, Rotaciones).

**Medición:** Colapsa el estado cuántico a una base, obteniendo resultados probabilísticos.

**Transformada Cuántica de Fourier (QFT):** Herramienta fundamental en algoritmos cuánticos, aunque no aplicada directamente en VQC/QSVM.

### 2.2 Machine Learning clásico
**Aprendizaje supervisado:** Entrenamiento con datos etiquetados para tareas de clasificación y regresión.

**Métricas de evaluación:**
- Accuracy: Proporción de predicciones correctas
- Precision, Recall, F1-score: Métricas detalladas por clase
- Matriz de confusión: Visualización de predicciones vs valores reales

### 2.3 Quantum Machine Learning (QML)

**Feature Maps (Embeddings cuánticos):** Transforman datos clásicos en estados cuánticos mediante circuitos parametrizados. Ejemplo: `ZZFeatureMap` codifica características en rotaciones y entrelazamiento.

**Modelos variacionales:** Circuitos cuánticos parametrizados optimizados mediante algoritmos clásicos (híbrido cuántico-clásico).

**Variational Quantum Classifier (VQC):**
- Feature map: Codifica datos de entrada
- Ansatz: Circuito parametrizado variable
- Medición: Extrae predicciones
- Optimización clásica: Ajusta parámetros minimizando una función de pérdida

**Quantum Support Vector Machine (QSVM):** Utiliza kernels cuánticos para calcular productos internos en espacio de Hilbert de alta dimensión.

**Quantum Neural Networks (QNN):** Redes parametrizadas cuánticas entrenables, análogas a redes neuronales clásicas.

### 2.4 Frameworks y herramientas

**Qiskit Machine Learning:** Framework principal utilizado, proporciona VQC, QSVM, y herramientas de integración con scikit-learn.

**Alternativas:**
- PennyLane: Diferenciación automática y soporte multi-backend
- TensorFlow Quantum: Integración con TensorFlow para modelos híbridos
- Amazon Braket: Acceso a hardware cuántico diverso

---

## 3. Metodología

### 3.1 Investigación y diseño

**Revisión bibliográfica:** Análisis de papers fundacionales en QML, documentación oficial de Qiskit, y casos de estudio comparativos.

**Selección del modelo:** VQC elegido por su flexibilidad, interpretabilidad y soporte robusto en Qiskit. Comparación con QSVM y QNN realizada para justificar la elección.

**Dataset:** Iris con 2 features (petal length, petal width) seleccionado por:
- Problema de clasificación multi-clase bien establecido
- Reducción dimensional facilita simulación cuántica eficiente
- Permite visualización y análisis interpretable

### 3.2 Implementación

**Baseline clásico (SVM):**
- Kernel RBF
- Pipeline: StandardScaler + SVC
- Parámetros por defecto de scikit-learn

**Modelo cuántico (VQC):**
- **Feature map:** `ZZFeatureMap` (2 qubits, reps=2) - entrelazamiento ZZ para codificación no lineal
- **Ansatz:** `RealAmplitudes` (reps=3) - circuito parametrizado con rotaciones Ry y CNOT
- **Optimizador:** COBYLA (maxiter=100) - libre de gradientes, robusto a ruido
- **Sampler:** API V2 (`AerSampler` o `StatevectorSampler`) con shots configurables
- **Pass manager:** Transpilación optimizada (nivel 1) para eficiencia en simulador

**Preprocesamiento:**
- Split estratificado 70/30 (train/test)
- StandardScaler para normalización
- Random state fijo (42) para reproducibilidad

### 3.3 Evaluación

**Métricas de desempeño:**
- Accuracy global
- Precision, recall, F1-score por clase
- Matriz de confusión

**Métricas de costo computacional:**
- Tiempo de entrenamiento (segundos)
- Número de qubits
- Profundidad del circuito
- Número de parámetros variacionales
- Evaluaciones de función objetivo (nfev)

**Protocolo experimental:**
- Múltiples corridas para estimar variabilidad
- Reporte de media ± desviación estándar
- Análisis de convergencia del optimizador

### 3.4 Documentación y reproducibilidad
 
**Estructura del repositorio:**
- `ProyectoQuantumML.ipynb`: Notebook principal
- `requirements.txt`: Dependencias con versiones
- `README.md`: Instrucciones de instalación y ejecución
- `informe_articulo.md`: Documentación técnica

**Reproducibilidad:** Entorno virtual (venv), random seeds fijos, documentación de versiones de librerías.

---

## 4. Resultados y análisis

### 4.1 Comparación de métricas

| Modelo | Accuracy | Tiempo (s) | Qubits | Parámetros | Observaciones |
|--------|----------|------------|--------|------------|---------------|
| SVM    | ~91%     | 0.002      | -      | -          | Rápido, preciso |
| VQC    | ~87-91%  | 30-60      | 2      | 8          | Variable, costoso |

### 4.2 Visualizaciones clave

**Convergencia del optimizador:** Gráfica de función de pérdida vs iteraciones muestra reducción gradual, con posibles plateaus.

**Matrices de confusión:** Comparación visual de errores de clasificación entre SVM y VQC por clase.

### 4.3 Discusión

**Sensibilidad a hiperparámetros:** VQC sensible a número de repeticiones (reps), profundidad del ansatz, y límite de iteraciones del optimizador.

**Estabilidad del optimizador:** COBYLA puede converger a mínimos locales; experimentación con SPSA o L-BFGS-B recomendada.

**Limitaciones en simulación:** Overhead computacional alto en simuladores clásicos; resultados no reflejan ventajas de hardware cuántico real.

---

## 5. Conclusiones y trabajos futuros

### 5.1 Hallazgos principales
- VQC competitivo en accuracy pero significativamente más lento en simulación
- SVM clásico superior en relación costo-beneficio para este problema
- Implementación exitosa de pipeline cuántico robusto con Qiskit

### 5.2 Limitaciones
- **Simulador:** No captura beneficios de paralelismo cuántico real
- **Escalabilidad:** 2 qubits insuficientes para problemas complejos
- **Optimización variacional:** Convergencia no garantizada, sensible a inicialización
- **Ruido:** Simulador ideal; hardware real requiere mitigación de errores

### 5.3 Trabajo futuro
- **QSVM:** Explorar kernels cuánticos explícitos y comparar con VQC
- **QNN alternativa:** Implementar arquitecturas más profundas o híbridas
- **Hardware real:** Ejecutar en IBM Quantum con análisis de ruido
- **Feature maps/ansatz:** Experimentar con `PauliFeatureMap`, `EfficientSU2`, etc.
- **Datasets más complejos:** MNIST reducido, Wine, Breast Cancer
- **Error mitigation:** Implementar técnicas de corrección de errores

---

## 6. Referencias

1. **Repositorio base:** [ml-to-qml](https://github.com/LukePower01/ml-to-qml)
2. **Qiskit Documentation:** [qiskit.org/documentation](https://qiskit.org/documentation/)
3. **Qiskit Machine Learning:** [qiskit.org/ecosystem/machine-learning](https://qiskit.org/ecosystem/machine-learning/)
4. Havlíček, V., et al. (2019). "Supervised learning with quantum-enhanced feature spaces." *Nature*, 567, 209-212.
5. Cerezo, M., et al. (2021). "Variational quantum algorithms." *Nature Reviews Physics*, 3, 625-644.
6. Schuld, M., & Petruccione, F. (2021). *Machine Learning with Quantum Computers*. Springer.
