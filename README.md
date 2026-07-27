# SurgiVision AI  
## Clasificación multilabel de instrumentos quirúrgicos

**Clasificación multilabel con MobileNetV2 | AUC 0.9434 | Inferencia multivista | Validación externa con revisión humana | Streamlit + Gradio + Google Colab**

Esta documentación presenta un proyecto académico de inteligencia artificial desarrollado para reconocer la presencia de uno o varios instrumentos quirúrgicos en una misma imagen.

El sistema utiliza una red neuronal convolucional basada en **MobileNetV2**, transferencia de aprendizaje, cuatro salidas con activación sigmoide y umbrales independientes por clase. El entrenamiento y las pruebas se realizaron en **Google Colab**; la aplicación principal fue desarrollada con **Streamlit** y desplegada en **Streamlit Community Cloud**; adicionalmente, se construyó una prueba de concepto en **Gradio** para demostrar la predicción, la revisión humana y la validación externa con imágenes nuevas.

> **Proyecto académico independiente.** No contiene información clínica identificable ni pretende reemplazar procedimientos institucionales de control de instrumental.

---

## 📌 Información general

| Parámetro | Descripción |
|---|---|
| Nombre del sistema | SurgiVision AI |
| Tarea | Clasificación multilabel de imágenes |
| Área | Inteligencia artificial aplicada al instrumental quirúrgico |
| Modelo base | MobileNetV2 preentrenada con ImageNet |
| Técnica principal | Transferencia de aprendizaje y *fine-tuning* |
| Entrada | Imagen RGB de `224 × 224 × 3` |
| Salida | Cuatro probabilidades sigmoidales independientes |
| Número de clases | 4 |
| Formato del modelo | `.h5` |
| Aplicación principal | Streamlit |
| POC interactiva | Gradio |
| Entorno de desarrollo | Google Colab |
| Despliegue | Streamlit Community Cloud |
| Estado | Prototipo académico |
| Año | 2026 |

### Equipo académico

- **Madeleine Arévalo**
- **Miriam Tatiana Aucapiña Calle**
- **Esteban Tuquiñagui**

---

## 👩‍💻 Perfil técnico demostrado

Este proyecto integra conocimientos de:

- Inteligencia artificial aplicada al área biomédica.
- Aprendizaje profundo con redes neuronales convolucionales.
- Clasificación multilabel.
- Transferencia de aprendizaje con MobileNetV2.
- Preparación y análisis de conjuntos de imágenes.
- Aumento de datos.
- Entrenamiento inicial y ajuste fino.
- Evaluación mediante métricas multilabel.
- Optimización de umbrales por clase.
- Inferencia con aumento en tiempo de prueba.
- Desarrollo de interfaces con Streamlit y Gradio.
- Despliegue de aplicaciones en la nube.
- Validación externa con imágenes nuevas.
- Revisión humana de predicciones.
- Registro trazable en CSV y JSONL.
- Documentación reproducible en GitHub.

---

## 🎥 Demostración en video

### Video único: demostración funcional y técnica — 2 minutos

https://github.com/user-attachments/assets/fc3929fb-02dc-4870-a1e3-9e9dbeee843e

### Estructura del video

#### Demostración funcional: «El sistema en acción»

- Apertura de SurgiVision AI.
- Confirmación visual de que el sistema se encuentra disponible.
- Registro opcional de un código de bandeja o procedimiento.
- Selección del momento del control.
- Carga de una fotografía desde el dispositivo.
- Captura alternativa mediante cámara.
- Ejecución del análisis.
- Reconocimiento de una o varias clases simultáneamente.
- Presentación del resultado preliminar.
- Visualización del estado de la imagen.
- Verificación pendiente.
- Confirmación o corrección por parte del usuario.
- Registro opcional de observaciones.
- Almacenamiento de una nueva validación externa.

#### Métricas de rendimiento

- AUC global: **0.9434**.
- Exactitud binaria: **0.8591**.
- Precisión micro: **0.8053**.
- Recall micro: **0.7839**.
- F1 macro: **0.7937**.
- Exactitud multilabel exacta: **0.6514**.
- F1 específico para cada una de las cuatro clases.
- Umbrales independientes optimizados con el conjunto de validación.

#### Arquitectura técnica

- Entrada RGB de `224 × 224 × 3`.
- MobileNetV2 preentrenada con ImageNet.
- Transferencia de aprendizaje.
- Entrenamiento inicial.
- *Fine-tuning*.
- Cuatro salidas sigmoidales independientes.
- Pérdida `BinaryCrossentropy`.
- Inferencia con tres vistas.
- Aplicación de umbrales por clase.
- Interfaz principal en Streamlit.
- POC interactiva en Gradio.
- Desarrollo y pruebas en Google Colab.
- Modelo final exportado en formato `.h5`.

#### Propuesta de valor

- Reconocimiento simultáneo de varias categorías.
- Interfaz accesible desde el navegador.
- Predicción interpretable por clase.
- Ausencia de clasificación forzada cuando no se supera ningún umbral.
- Revisión humana antes de aceptar el resultado.
- Registro de confirmaciones y correcciones.
- Posibilidad de evaluar generalización con imágenes externas.
- Aplicación académica para capacitación y experimentación.

#### Detalles técnicos adicionales

- Organización del dataset.
- Conversión de etiquetas a vectores multilabel.
- División en entrenamiento, validación y prueba.
- Aumento de datos.
- Construcción de MobileNetV2.
- Carga y verificación del modelo.
- Generación de tres vistas de cada imagen.
- Cálculo de probabilidades.
- Comparación con umbrales.
- Cálculo del margen por clase.
- Cálculo del acuerdo entre vistas.
- Confirmación o corrección de etiquetas.
- Almacenamiento en CSV, JSONL y carpeta de imágenes.
- Gestión de la versión del modelo.
- Registro de fecha, hash y observaciones.

---

Copia y pega este bloque directamente en tu archivo `README_SurgiVision_PORTAFOLIO_COMPLETO.md`:

```markdown
## ⏱️ Estructura del video de 2 minutos

### 1. Presentación del proyecto — 0:00 a 0:12

- Presentación de la autora.
- Nombre del sistema: **SurgiVision AI**.
- Objetivo general del proyecto.
- Clasificación multilabel de instrumentos quirúrgicos.

### 2. Funcionamiento de la aplicación — 0:12 a 0:38

- Registro opcional del código de control.
- Selección del momento del procedimiento.
- Carga de una imagen desde el dispositivo.
- Captura de una fotografía mediante la cámara.
- Presentación de las cuatro clases reconocidas.
- Ejecución del análisis desde la interfaz.

### 3. Resultado y revisión humana — 0:38 a 0:58

- Visualización de los instrumentos reconocidos.
- Presentación de las probabilidades por clase.
- Comparación con los umbrales de decisión.
- Estado de verificación pendiente.
- Confirmación del resultado.
- Corrección de las clases reconocidas.
- Registro opcional de observaciones.

### 4. Código y proceso de predicción — 0:58 a 1:22

- Entrada de imágenes de `224 × 224` píxeles.
- Conversión de la imagen a formato RGB.
- Generación de tres vistas:
  - Imagen original.
  - Reflejo horizontal.
  - Recorte central del 85 %.
- Obtención de probabilidades para cada clase.
- Selección del valor máximo entre las tres vistas.
- Comparación con el umbral específico de cada instrumento.
- Generación del vector multilabel final.

### 5. Rendimiento del modelo — 1:22 a 1:40

- AUC: `0.9434`.
- Exactitud binaria: `0.8591`.
- Precisión micro: `0.8053`.
- Recall micro: `0.7839`.
- F1 macro: `0.7937`.
- Exactitud multilabel exacta: `0.6514`.
- Presentación resumida del desempeño por clase.

### 6. Validación externa — 1:40 a 1:53

- Uso de imágenes nuevas no empleadas durante el entrenamiento.
- Evaluación del modelo en condiciones diferentes.
- Confirmación o corrección de la predicción.
- Registro de las etiquetas predichas.
- Registro de las etiquetas validadas.
- Almacenamiento de probabilidades y observaciones.
- Exportación de resultados en archivos CSV y JSONL.
- Evaluación posterior de la capacidad de generalización.

### 7. Cierre y alcance — 1:53 a 2:00

- Integración de aprendizaje profundo, aplicación web y revisión humana.
- Aplicación académica para reconocimiento de instrumental.
- Posibilidad de ampliar el sistema a nuevas clases.
- Aclaración de que el prototipo no reemplaza el conteo manual.
- Aclaración de que no sustituye los protocolos del personal de salud.
```


---

## 🎯 Problema y solución

### El desafío

El reconocimiento visual de instrumental puede dificultarse cuando:

- Existen varios instrumentos en una misma imagen.
- Las clases presentan formas o componentes similares.
- Los objetos aparecen en orientaciones diferentes.
- Cambian el fondo, la iluminación o la distancia.
- Se producen oclusiones parciales.
- Una fotografía contiene varias categorías al mismo tiempo.
- Se requiere evaluar el modelo fuera del conjunto utilizado para desarrollarlo.

### La solución: SurgiVision AI

SurgiVision AI es un clasificador multilabel que:

- Procesa la imagen completa.
- Genera una probabilidad independiente para cada clase.
- Permite reconocer varias clases simultáneamente.
- Utiliza umbrales específicos para cada instrumento.
- Evita asignar una etiqueta cuando ninguna clase supera su umbral.
- Presenta resultados interpretables.
- Incorpora revisión humana.
- Registra confirmaciones y correcciones.
- Permite realizar validación externa con fotografías nuevas.
- Puede ampliarse en versiones posteriores.

---

## 🌐 Aplicación web

### Acceso directo a la demostración

[**Abrir SurgiVision AI en Gradio**](https://979e429e1407f01db3.gradio.live/)

> La aplicación se ejecuta mediante un enlace público temporal de Gradio. Si el enlace deja de estar disponible, será necesario volver a ejecutar el cuaderno en Google Colab y actualizar esta dirección.

### Funciones disponibles

#### Registro contextual

- Código opcional de bandeja o procedimiento.
- Selección del momento del control.
- Separación entre predicción automática y verificación humana.
- Aviso permanente sobre el alcance académico del sistema.

#### Entrada de imágenes

- Carga desde el dispositivo.
- Captura mediante cámara.
- Conversión automática a RGB.
- Corrección de orientación EXIF.
- Verificación básica de dimensiones.
- Compatibilidad con imágenes comunes.

#### Predicción

- Redimensionamiento a `224 × 224`.
- Generación de tres vistas.
- Obtención de cuatro probabilidades.
- Aplicación de umbrales.
- Presentación de clases reconocidas.
- Estado `DETECCION`.
- Estado `SIN_DETECCIONES`.
- Manejo de imagen no apta o entrada inválida.

#### Revisión del resultado

- Número de tipos reconocidos.
- Estado de la imagen.
- Estado de la verificación.
- Confirmación y guardado.
- Corrección de etiquetas.
- Registro de observaciones.
- Reinicio del análisis.

#### Gestión de validaciones

- Conteo de registros validados.
- Almacenamiento de imágenes.
- Registro CSV.
- Registro JSONL.
- Asociación con la versión del modelo.
- Trazabilidad mediante fecha, UUID y hash SHA-256.

---

## 🧾 Clases reconocidas

| Índice | Clase | Abreviatura |
|---:|---|---|
| 0 | Bisturí n.º 4 | B4 |
| 1 | Pinza de disección recta | PR |
| 2 | Tijera Mayo recta | MR |
| 3 | Tijera Mayo curva | MC |

Al ser una clasificación multilabel, una imagen puede recibir:

- Ninguna etiqueta.
- Una sola etiqueta.
- Dos o más etiquetas simultáneamente.

### Ejemplo

```text
[1, 0, 1, 0]
```

Interpretación:

- Bisturí n.º 4: presente.
- Pinza de disección recta: ausente.
- Tijera Mayo recta: presente.
- Tijera Mayo curva: ausente.

---

## 🧠 Arquitectura del modelo

### Configuración principal

| Parámetro | Valor |
|---|---|
| Arquitectura base | MobileNetV2 |
| Pesos iniciales | ImageNet |
| Tipo de aprendizaje | Transferencia de aprendizaje |
| Ajuste posterior | Fine-tuning |
| Forma de entrada | `(224, 224, 3)` |
| Número de salidas | 4 |
| Activación final | Sigmoide |
| Función de pérdida | BinaryCrossentropy |
| Tipo de problema | Multilabel |
| Formato exportado | HDF5 `.h5` |
| Nombre esperado | `modelo_instrumentos_multilabel.h5` |

### Razón para utilizar sigmoide

Cada neurona de salida produce una probabilidad independiente:

```text
p(clase i presente | imagen)
```

Esto es coherente con el problema porque la presencia de un instrumento no excluye la presencia de los demás.

No se utiliza *softmax*, ya que esa activación supone clases mutuamente excluyentes.

### Transferencia de aprendizaje

El flujo general comprende:

1. Carga de MobileNetV2 con pesos de ImageNet.
2. Sustitución de la cabeza de clasificación.
3. Congelamiento inicial de la base convolucional.
4. Entrenamiento de las nuevas capas.
5. Descongelamiento parcial.
6. Ajuste fino con una tasa de aprendizaje reducida.
7. Evaluación.
8. Optimización de umbrales.
9. Exportación del modelo final.

---

## 🔬 Flujo de inferencia multivista

### Etapas

| Etapa | Operación | Resultado |
|---:|---|---|
| 1 | Carga del archivo `.h5` | Modelo disponible |
| 2 | Verificación de entrada y salida | Compatibilidad confirmada |
| 3 | Conversión a PIL y RGB | Entrada uniforme |
| 4 | Corrección EXIF | Orientación normalizada |
| 5 | Generación de tres vistas | Original, reflejo y recorte |
| 6 | Redimensionamiento | Lote de `224 × 224` |
| 7 | Predicción | Matriz `3 × 4` |
| 8 | Agregación | Máximo por clase |
| 9 | Aplicación de umbrales | Vector multilabel |
| 10 | Presentación | Resultado y tabla técnica |

### Vistas utilizadas

1. **Original:** imagen sin transformación.
2. **Flip horizontal:** reflejo de la imagen.
3. **Recorte central:** región central correspondiente al 85 % del ancho y alto.

```text
Imagen original
     ├── Vista original
     ├── Reflejo horizontal
     └── Recorte central 85 %
              ↓
        Modelo MobileNetV2
              ↓
      Probabilidad por vista
              ↓
      Máximo por cada clase
```

### Parámetros del procesamiento

| Parámetro | Valor |
|---|---:|
| Tamaño objetivo | `224 × 224` |
| Canales | 3 |
| Número de vistas | 3 |
| Proporción de recorte | 0.85 |
| Interpolación | Bilinear |
| Forma del lote | `(3, 224, 224, 3)` |
| Forma esperada de salida | `(3, 4)` |
| Agregación | Máximo por clase |
| Tamaño mínimo admitido | `20 × 20` píxeles |

### Normalización

El modelo contiene internamente una capa `Rescaling` que transforma los píxeles desde:

```text
[0, 255] → [-1, 1]
```

Por esta razón, la función de predicción no divide otra vez la imagen para 255. Una segunda normalización alteraría la escala esperada por el modelo.

---

## 🎚️ Umbrales operativos

La POC actual utiliza los siguientes umbrales:

| Índice | Instrumento | Umbral |
|---:|---|---:|
| 0 | Bisturí n.º 4 | 0.57 |
| 1 | Pinza de disección recta | 0.70 |
| 2 | Tijera Mayo recta | 0.59 |
| 3 | Tijera Mayo curva | 0.48 |

### Regla de decisión

```python
detectado = probabilidad_final >= umbral
```

### Ejemplo

```text
Probabilidad final: 0.76
Umbral: 0.70
Resultado: detectado
Margen: +0.06
```

Los umbrales se determinan con datos de validación. No deben ajustarse utilizando el conjunto de prueba ni las imágenes externas destinadas a evaluar generalización.

---

## 📊 Información devuelta por la predicción

| Campo | Interpretación |
|---|---|
| `Instrumento` | Nombre de la clase |
| `Original` | Probabilidad en la imagen original |
| `Flip horizontal` | Probabilidad en el reflejo |
| `Recorte central` | Probabilidad en el recorte |
| `Probabilidad final` | Máximo de las tres vistas |
| `Umbral` | Valor de decisión de la clase |
| `Margen` | Probabilidad final menos umbral |
| `Detectado` | Decisión multilabel |
| `Acuerdo de vistas` | Porcentaje de vistas que superaron el umbral |
| `estado` | Estado general |
| `mensaje` | Explicación preparada para la interfaz |

### Estados principales

#### `DETECCION`

Al menos una clase supera su umbral.

#### `SIN_DETECCIONES`

Ninguna clase supera su umbral. El sistema no fuerza una etiqueta.

#### `IMAGEN_NO_APTA`

La imagen no cumple una condición mínima definida por la interfaz o el proceso de validación.

> `SIN_DETECCIONES` no significa que la imagen esté libre de instrumental. Significa únicamente que ninguna de las cuatro clases superó los umbrales configurados.

---

## 📈 Resultados del modelo

### Métricas globales

| Métrica | Valor | Porcentaje aproximado |
|---|---:|---:|
| AUC | 0.9434 | 94.34 % |
| Exactitud binaria | 0.8591 | 85.91 % |
| Precisión micro | 0.8053 | 80.53 % |
| Recall micro | 0.7839 | 78.39 % |
| F1 macro | 0.7937 | 79.37 % |
| Exactitud multilabel exacta | 0.6514 | 65.14 % |

### Interpretación de las métricas

#### AUC

Evalúa la capacidad de separar ejemplos positivos y negativos a diferentes umbrales.

#### Exactitud binaria

Calcula la proporción de decisiones correctas considerando cada etiqueta por separado.

#### Precisión micro

Indica qué proporción de todas las etiquetas predichas como positivas fue realmente correcta.

#### Recall micro

Indica qué proporción de todas las etiquetas verdaderamente presentes fue recuperada.

#### F1 macro

Calcula el F1 de cada clase y promedia los resultados asignando igual peso a todas las categorías.

#### Exactitud multilabel exacta

Exige que el vector completo de etiquetas de una imagen coincida exactamente con la referencia. Es una métrica más estricta que la exactitud binaria.

### F1 por clase

| Instrumento | F1 | Porcentaje aproximado |
|---|---:|---:|
| Bisturí n.º 4 | 0.8065 | 80.65 % |
| Pinza de disección recta | 0.7629 | 76.29 % |
| Tijera Mayo recta | 0.7635 | 76.35 % |
| Tijera Mayo curva | 0.8419 | 84.19 % |

### Lectura general

- La tijera Mayo curva obtuvo el F1 más alto.
- El bisturí n.º 4 presentó el segundo mejor resultado.
- La pinza recta y la tijera Mayo recta mostraron resultados similares.
- La exactitud multilabel exacta es menor porque requiere acertar simultáneamente todas las etiquetas de la imagen.

---

## 🧪 Cobertura funcional

La evaluación funcional debe incluir, como mínimo:

| Caso | Resultado esperado |
|---|---|
| Imagen con un instrumento conocido | Reconocimiento de una clase |
| Imagen con varias clases | Varias etiquetas positivas |
| Imagen sin clases reconocibles | Estado `SIN_DETECCIONES` |
| Imagen demasiado pequeña | Mensaje de entrada no válida |
| Archivo inexistente | Manejo controlado del error |
| Modelo incorrecto | Error de forma de entrada o salida |
| Orientación EXIF | Conversión correcta |
| Fondo distinto | Evaluación externa |
| Iluminación diferente | Evaluación de generalización |
| Instrumento ocluido | Registro de posible error |
| Confirmación correcta | Guardado como confirmación |
| Corrección de clase | Guardado como corrección |
| Observación vacía | Registro permitido |
| Varias etiquetas corregidas | Vector multilabel actualizado |
| Reinicio | Limpieza del estado de la interfaz |

### Verificaciones automáticas del modelo

La carga del modelo comprueba:

```text
Entrada esperada: (None, 224, 224, 3)
Salida esperada:  (None, 4)
```

La inferencia verifica:

```text
Matriz esperada: (3, 4)
```

Estas comprobaciones reducen el riesgo de utilizar por error otro modelo o un archivo incompatible.

---

## 🌍 Validación externa con imágenes nuevas

### Definición

La validación externa utiliza fotografías independientes que no participaron en:

- Entrenamiento.
- Validación interna.
- Optimización de umbrales.
- Prueba original.

Su objetivo es comprobar el comportamiento del sistema en condiciones distintas a las utilizadas durante el desarrollo.

### Condiciones que puede evaluar

- Dispositivo de captura diferente.
- Fondo nuevo.
- Cambios de iluminación.
- Distancia diferente.
- Rotaciones.
- Escalas.
- Oclusiones.
- Superposición.
- Combinaciones de instrumentos.
- Imágenes sin instrumentos reconocibles.

### Flujo

```text
Imagen nueva
   ↓
Predicción del modelo
   ↓
Etiquetas preliminares
   ↓
Revisión humana
   ├── Confirmar
   └── Corregir
   ↓
Registro independiente
   ↓
Cálculo posterior de métricas externas
```

### Regla metodológica

Las imágenes externas deben permanecer separadas mientras se calculan los resultados externos.

Después de cerrar y documentar la validación, podrán incorporarse a una versión posterior del conjunto de entrenamiento. Si se utilizan para reentrenar antes de medir el desempeño externo, dejarían de ser datos independientes.

### Métricas recomendadas para la validación externa

| Métrica | Estado |
|---|---|
| Número total de imágenes externas | `[REGISTRAR]` |
| Distribución por clase | `[REGISTRAR]` |
| Exactitud binaria externa | `[CALCULAR]` |
| Precisión micro externa | `[CALCULAR]` |
| Recall micro externo | `[CALCULAR]` |
| F1 macro externo | `[CALCULAR]` |
| Exactitud multilabel exacta externa | `[CALCULAR]` |
| F1 externo por clase | `[CALCULAR]` |
| Casos confirmados | `[REGISTRAR]` |
| Casos corregidos | `[REGISTRAR]` |
| Tasa de concordancia humana | `[CALCULAR]` |

No se asignan valores a estas métricas hasta completar una muestra externa suficiente y revisar las etiquetas.

---

## 👤 Revisión humana

La aplicación mantiene un enfoque de **human-in-the-loop**:

1. El modelo genera una predicción preliminar.
2. El usuario revisa la fotografía.
3. Puede confirmar el resultado.
4. Puede abrir el panel de corrección.
5. Selecciona todas las clases realmente observadas.
6. Añade observaciones.
7. Guarda la validación.

### Tipos de retroalimentación

| Tipo | Descripción |
|---|---|
| Confirmación | Predicción y etiqueta validada coinciden |
| Corrección | El usuario modifica una o varias etiquetas |
| Sin detecciones confirmado | Ninguna clase del sistema está presente |
| Omisión corregida | El modelo no reconoció una clase observada |
| Falso positivo corregido | El modelo marcó una clase ausente |

---

## 💾 Registro de validaciones

### Directorio de almacenamiento

La prioridad de ubicación es:

1. Variable de entorno `SURGIVISION_FEEDBACK_DIR`.
2. Google Drive montado en Colab.
3. Carpeta local del entorno.

Ruta predeterminada en Google Drive:

```text
/content/drive/MyDrive/SurgiVision_AI/feedback_entrenamiento
```

Ruta local alternativa:

```text
surgivision_feedback/
```

### Estructura

```text
surgivision_feedback/
├── imagenes/
├── feedback_validado.csv
└── feedback_validado.jsonl
```

### Campos registrados

| Campo | Propósito |
|---|---|
| `id_registro` | UUID único |
| `fecha_validacion_utc` | Fecha y hora en UTC |
| `archivo_imagen` | Nombre del archivo almacenado |
| `sha256_imagen` | Huella de integridad de la imagen |
| `codigo_control` | Código opcional de bandeja o procedimiento |
| `momento_control` | Etapa seleccionada |
| `modelo_version` | Archivo o versión del modelo |
| `estado_modelo` | Estado devuelto por la inferencia |
| `etiquetas_predichas` | Clases seleccionadas por el modelo |
| `etiquetas_validadas` | Clases confirmadas por el usuario |
| `vector_etiquetas` | Vector binario multilabel |
| `tipo_feedback` | Confirmación o corrección |
| `probabilidades` | Valores por clase |
| `observaciones` | Comentario del revisor |

### Formatos utilizados

#### CSV

Adecuado para:

- Excel.
- Pandas.
- Análisis tabular.
- Cálculo de métricas.
- Auditoría manual.

#### JSONL

Adecuado para:

- Lectura registro por registro.
- Procesamiento programático.
- Conservación de listas y estructuras.
- Integraciones futuras.

#### Imágenes

Se guardan en un directorio separado y se relacionan mediante el nombre de archivo y el hash SHA-256.

---

## 🏗️ Arquitectura técnica

### Vista general

```text
┌─────────────────────────────────────────────────────────────┐
│                       USUARIO                               │
│       Carga una imagen o utiliza la cámara                 │
└───────────────────────────┬─────────────────────────────────┘
                            │
                            ▼
┌─────────────────────────────────────────────────────────────┐
│                STREAMLIT / GRADIO                           │
│ Código de control · Momento · Imagen · Observaciones       │
└───────────────────────────┬─────────────────────────────────┘
                            │
                            ▼
┌─────────────────────────────────────────────────────────────┐
│             PREPARACIÓN DE LA IMAGEN                        │
│ EXIF → RGB → 3 vistas → 224 × 224                          │
└───────────────────────────┬─────────────────────────────────┘
                            │
                            ▼
┌─────────────────────────────────────────────────────────────┐
│                MODELO MOBILENETV2                           │
│ Entrada: (3, 224, 224, 3) · Salida: (3, 4)                │
└───────────────────────────┬─────────────────────────────────┘
                            │
                            ▼
┌─────────────────────────────────────────────────────────────┐
│            AGREGACIÓN Y DECISIÓN                            │
│ Máximo por clase → Umbral → Margen → Acuerdo               │
└───────────────────────────┬─────────────────────────────────┘
                            │
                            ▼
┌─────────────────────────────────────────────────────────────┐
│                 RESULTADO PRELIMINAR                        │
│ Detección · Sin detecciones · Imagen no apta               │
└───────────────────────────┬─────────────────────────────────┘
                            │
                            ▼
┌─────────────────────────────────────────────────────────────┐
│                  REVISIÓN HUMANA                            │
│ Confirmación · Corrección · Observaciones                  │
└───────────────────────────┬─────────────────────────────────┘
                            │
                            ▼
┌─────────────────────────────────────────────────────────────┐
│               VALIDACIÓN EXTERNA                            │
│ CSV · JSONL · Imágenes · SHA-256 · Versión                 │
└─────────────────────────────────────────────────────────────┘
```

### Arquitectura por capas

| Capa | Responsabilidad |
|---|---|
| 1. Datos | Imágenes y etiquetas multilabel |
| 2. Preparación | Limpieza, vectores, divisiones |
| 3. Entrenamiento | MobileNetV2 y fine-tuning |
| 4. Modelo | Archivo `.h5` |
| 5. Inferencia | Preprocesamiento, vistas y probabilidades |
| 6. Decisión | Umbrales, margen y acuerdo |
| 7. Interfaz | Streamlit y Gradio |
| 8. Validación externa | Confirmación, corrección y persistencia |

---

## 🔌 Modelo de integración

### Google Colab

Se utiliza para:

- Ejecutar los cuadernos.
- Acceder a recursos de cómputo.
- Preparar el dataset.
- Entrenar el modelo.
- Calcular métricas.
- Probar imágenes nuevas.
- Ejecutar la POC en Gradio.
- Guardar validaciones en Google Drive.

### Streamlit

Se utiliza como aplicación principal:

```text
app.py
   ↓
Carga modelo .h5
   ↓
Recibe imagen
   ↓
Genera predicción
   ↓
Muestra probabilidades
   ↓
Permite revisión
```

### Gradio

Se utiliza como POC de experimentación:

- Interfaz temporal desde Colab.
- Prueba directa del modelo.
- Validación externa.
- Corrección de etiquetas.
- Registro CSV y JSONL.
- Presentación técnica de probabilidades y umbrales.

### GitHub

Se utiliza para:

- Almacenar el código.
- Mantener el historial de cambios.
- Documentar el proyecto.
- Conectar el repositorio con Streamlit Community Cloud.
- Compartir los cuadernos y requisitos.

---

## 📁 Estructura del proyecto

```text
Reconocimiento-instrumentos-quirurgicos/
├── C1_Carga_Preparacion_Datos.ipynb
├── C2_Entrenamiento_Multilabel.ipynb
├── C3_Prediccion_Conclusiones_POC.ipynb
├── modelo_instrumentos_multilabel.h5
├── app.py
├── requirements.txt
├── README.md
└── validacion_externa/
    ├── imagenes/
    ├── feedback_validado.csv
    └── feedback_validado.jsonl
```

> La carpeta de validación externa puede encontrarse fuera del repositorio para evitar publicar fotografías o información sensible.

---

## 📚 Referencia de archivos

### `C1_Carga_Preparacion_Datos.ipynb`

Responsabilidades:

- Descarga del dataset.
- Organización de carpetas.
- Lectura de imágenes.
- Lectura de etiquetas.
- Conversión a vectores multilabel.
- Verificación de archivos.
- Análisis exploratorio.
- Distribución de clases.
- División en entrenamiento, validación y prueba.

### `C2_Entrenamiento_Multilabel.ipynb`

Responsabilidades:

- Creación de datasets TensorFlow.
- Aumento de datos.
- Carga de MobileNetV2.
- Transferencia de aprendizaje.
- Construcción de la cabeza de clasificación.
- Configuración de salida sigmoide.
- Entrenamiento inicial.
- Fine-tuning.
- Cálculo de métricas.
- Matrices de confusión.
- Análisis por clase.
- Optimización de umbrales.
- Exportación en `.h5`.

### `C3_Prediccion_Conclusiones_POC.ipynb`

Responsabilidades:

- Instalación de Gradio y Pillow.
- Configuración de clases.
- Configuración de umbrales.
- Localización automática del modelo.
- Verificación de entrada y salida.
- Conversión de imagen.
- Corrección EXIF.
- Inferencia multivista.
- Cálculo de probabilidades.
- Cálculo de margen.
- Cálculo de acuerdo.
- Manejo de estados.
- Pruebas con imágenes nuevas.
- Interfaz Gradio.
- Registro de validaciones.
- Conclusiones y referencias.

### `modelo_instrumentos_multilabel.h5`

Contiene:

- Arquitectura entrenada.
- Pesos aprendidos.
- Capa de reescalado.
- Cuatro salidas.
- Parámetros necesarios para inferencia.

### `app.py`

Punto de entrada de Streamlit:

- Configuración visual.
- Carga del modelo.
- Entrada de imagen.
- Registro contextual.
- Inferencia.
- Presentación de resultados.
- Confirmación o corrección.
- Avisos de alcance.

### `requirements.txt`

Declara las dependencias necesarias para:

- Streamlit.
- TensorFlow.
- NumPy.
- Pandas.
- Pillow.
- Otras bibliotecas utilizadas.

### `README.md`

Incluye:

- Descripción general.
- Demostración.
- Arquitectura.
- Parámetros.
- Resultados.
- Instalación.
- Limitaciones.
- Validación externa.
- Hoja de ruta.

---

## 🛠️ Pila tecnológica

### Inteligencia artificial

| Componente | Tecnología | Propósito |
|---|---|---|
| Framework | TensorFlow | Entrenamiento e inferencia |
| API de alto nivel | Keras | Construcción del modelo |
| Red base | MobileNetV2 | Extracción de características |
| Pesos | ImageNet | Transferencia de aprendizaje |
| Activación | Sigmoide | Probabilidad independiente |
| Pérdida | BinaryCrossentropy | Optimización multilabel |
| Estrategia | Fine-tuning | Adaptación al dataset |

### Procesamiento

| Componente | Tecnología | Propósito |
|---|---|---|
| Arreglos | NumPy | Manejo numérico |
| Tablas | Pandas | Métricas y registros |
| Imágenes | Pillow | Conversión y transformaciones |
| Gráficos | Matplotlib | Visualización |
| Métricas | Scikit-learn | Evaluación |

### Aplicaciones

| Componente | Tecnología | Propósito |
|---|---|---|
| Aplicación principal | Streamlit | Interfaz publicada |
| POC | Gradio `>=5,<6` | Prueba interactiva |
| Entorno | Google Colab | Desarrollo y ejecución |
| Despliegue | Streamlit Community Cloud | Acceso web |
| Repositorio | GitHub | Versionamiento |

### Persistencia de validación

| Componente | Formato | Propósito |
|---|---|---|
| Tabla principal | CSV UTF-8 | Análisis tabular |
| Registro estructurado | JSONL | Procesamiento programático |
| Evidencia visual | Imágenes | Revisión posterior |
| Integridad | SHA-256 | Identificación del archivo |
| ID | UUID | Registro único |
| Tiempo | UTC | Trazabilidad temporal |

---

## 📝 Ejemplos de uso

### Ejemplo 1: una clase

Entrada:

```text
Fotografía con una pinza de disección recta
```

Salida conceptual:

```json
{
  "estado": "DETECCION",
  "instrumentos_detectados": [
    "Pinza de disección recta"
  ]
}
```

### Ejemplo 2: varias clases

Entrada:

```text
Fotografía con bisturí y tijera Mayo recta
```

Salida:

```text
[1, 0, 1, 0]
```

### Ejemplo 3: ninguna clase supera el umbral

```json
{
  "estado": "SIN_DETECCIONES",
  "instrumentos_detectados": []
}
```

### Ejemplo 4: corrección humana

Predicción:

```text
Pinza de disección recta
```

Validación:

```text
Tijera Mayo recta
```

Registro:

```json
{
  "tipo_feedback": "correccion",
  "etiquetas_predichas": ["Pinza de disección recta"],
  "etiquetas_validadas": ["Tijera Mayo recta"]
}
```

---

## 🚀 Ejecución local

### Clonar el repositorio

```bash
git clone https://github.com/MadeleineArevalo/Reconocimiento-instrumentos-quirurgicos.git
cd Reconocimiento-instrumentos-quirurgicos
```

### Crear entorno virtual

```bash
python -m venv .venv
```

Windows:

```powershell
.venv\Scripts\activate
```

Linux o macOS:

```bash
source .venv/bin/activate
```

### Instalar dependencias

```bash
pip install -r requirements.txt
```

### Verificar el modelo

```text
modelo_instrumentos_multilabel.h5
```

Debe encontrarse en la ruta utilizada por `app.py`.

### Ejecutar Streamlit

```bash
streamlit run app.py
```

Dirección local habitual:

```text
http://localhost:8501
```

---

## ☁️ Despliegue en Streamlit Community Cloud

1. Subir el repositorio a GitHub.
2. Confirmar que `app.py` esté en la raíz.
3. Confirmar que `requirements.txt` incluya todas las dependencias.
4. Abrir Streamlit Community Cloud.
5. Conectar la cuenta de GitHub.
6. Seleccionar el repositorio.
7. Seleccionar la rama.
8. Indicar `app.py` como archivo principal.
9. Ejecutar el despliegue.
10. Probar la carga, la cámara y la predicción.

### Recomendaciones

- No subir datos sensibles.
- Evitar modelos excesivamente grandes.
- Mantener versiones compatibles.
- No guardar validaciones privadas directamente en el repositorio público.
- Documentar cualquier cambio en umbrales o pesos.

---

## 🔐 Privacidad y seguridad de datos

### Principios

- No solicitar datos personales del paciente.
- No incluir nombres ni historias clínicas.
- No utilizar fotografías clínicas identificables.
- Registrar solo un código técnico opcional.
- Evitar publicar imágenes externas sin autorización.
- Mantener los archivos de validación fuera del repositorio público cuando corresponda.
- Utilizar hashes para identificar archivos sin depender únicamente del nombre.
- Controlar la versión del modelo utilizada en cada registro.

### Datos que no deben publicarse

- Información identificable.
- Credenciales.
- Tokens.
- Claves de servicios.
- Rutas privadas.
- Imágenes sin autorización.
- Archivos de validación con contenido sensible.

---

## ⚠️ Limitaciones

- Clasifica la imagen completa.
- No localiza instrumentos con cajas delimitadoras.
- No determina coordenadas.
- No cuenta varias unidades de la misma clase.
- No diferencia dos objetos iguales.
- Solo reconoce cuatro clases.
- Puede fallar con clases no vistas.
- Puede verse afectado por fondos diferentes.
- Puede verse afectado por iluminación.
- Puede verse afectado por orientación.
- Puede verse afectado por oclusión.
- El máximo entre vistas puede favorecer sensibilidad, pero también requiere evaluación de falsos positivos.
- Una probabilidad alta no garantiza corrección.
- `SIN_DETECCIONES` no equivale a ausencia confirmada.
- La validación externa todavía debe completarse con una muestra suficiente.
- No constituye un dispositivo médico validado.

---

## ✅ Propuesta de valor

SurgiVision AI demuestra una cadena completa:

```text
Dataset
→ preparación
→ entrenamiento
→ evaluación
→ optimización de umbrales
→ inferencia multivista
→ aplicación web
→ revisión humana
→ validación externa
```

### Valor académico

- Evidencia competencias en aprendizaje profundo.
- Demuestra comprensión de problemas multilabel.
- Integra teoría, código y aplicación.
- Presenta métricas reales.
- Incorpora interpretabilidad básica.
- Añade trazabilidad.
- Facilita pruebas con usuarios.
- Permite estudiar generalización.

### Posibles usos académicos

- Capacitación visual.
- Demostración de clasificación multilabel.
- Prácticas de transferencia de aprendizaje.
- Estudio de umbrales.
- Comparación de modelos.
- Evaluación de interfaces.
- Investigación sobre validación externa.

---

## 🎨 Patrones y conceptos de diseño

| Concepto | Implementación | Propósito |
|---|---|---|
| Separación de responsabilidades | C1, C2, C3 y app | Organización |
| Transferencia de aprendizaje | MobileNetV2 | Reducir entrenamiento desde cero |
| Multilabel | Sigmoide | Varias clases simultáneas |
| Umbrales por clase | Vector de 4 valores | Decisión adaptada |
| TTA | Tres vistas | Evaluar estabilidad |
| Agregación | Máximo por clase | Favorecer sensibilidad |
| Human-in-the-loop | Confirmar/corregir | Supervisión |
| Registro de auditoría | CSV y JSONL | Trazabilidad |
| Integridad | SHA-256 | Identificación de imagen |
| Versionado | `modelo_version` | Reproducibilidad |
| Fallo seguro | `SIN_DETECCIONES` | Evitar etiqueta forzada |

---

## 🔮 Hoja de ruta

### Corto plazo

- [ ] Completar una validación externa con mayor número de imágenes.
- [ ] Calcular métricas externas.
- [ ] Comparar resultados internos y externos.
- [ ] Verificar consistencia de umbrales en todos los archivos.
- [ ] Añadir pruebas automatizadas.
- [ ] Documentar errores frecuentes.
- [ ] Publicar el video de dos minutos.

### Mediano plazo

- [ ] Incorporar más imágenes por clase.
- [ ] Equilibrar el dataset.
- [ ] Añadir nuevas clases.
- [ ] Mejorar ejemplos con oclusión.
- [ ] Añadir un panel de métricas.
- [ ] Exportar informes.
- [ ] Implementar control de versiones de modelos.
- [ ] Comparar MobileNetV2 con EfficientNet.

### Largo plazo

- [ ] Evaluar detección de objetos.
- [ ] Localizar instrumentos con cajas.
- [ ] Contar varias unidades.
- [ ] Evaluar video en tiempo real.
- [ ] Desarrollar una versión móvil.
- [ ] Realizar validación con especialistas.
- [ ] Analizar requisitos regulatorios antes de cualquier uso real.

---

## 📚 Documentación adicional recomendada

```text
docs/
├── ARCHITECTURE.md
├── SETUP_GUIDE.md
├── EXTERNAL_VALIDATION.md
├── MODEL_CARD.md
├── TEST_REPORT.md
└── VIDEO_SCRIPT.md
```

### `ARCHITECTURE.md`

- Diagrama del sistema.
- Flujo de datos.
- Diseño del modelo.
- Inferencia multivista.

### `SETUP_GUIDE.md`

- Preparación local.
- Google Colab.
- Streamlit.
- Gradio.
- Solución de errores.

### `EXTERNAL_VALIDATION.md`

- Protocolo.
- Criterios de inclusión.
- Variables registradas.
- Métricas.
- Separación de datos.

### `MODEL_CARD.md`

- Uso previsto.
- Datos.
- Arquitectura.
- Métricas.
- Limitaciones.
- Consideraciones éticas.

### `TEST_REPORT.md`

- Casos de prueba.
- Evidencias.
- Resultados.
- Errores.
- Correcciones.

### `VIDEO_SCRIPT.md`

- Guion.
- Secuencia visual.
- Narración.
- Tiempos.

---

## 📧 Contacto y colaboración

Para consultas relacionadas con:

- Proyecto académico.
- Clasificación multilabel.
- Aplicaciones biomédicas.
- Revisión del código.
- Validación externa.
- Documentación.
- Colaboración estudiantil.

**Equipo del proyecto**

- Madeleine Arévalo
- Miriam Tatiana Aucapiña Calle
- Esteban Tuquiñagui

Correo de contacto:

```text
[COLOCAR CORREO PROFESIONAL]
```

Repositorio:

[MadeleineArevalo/Reconocimiento-instrumentos-quirurgicos](https://github.com/MadeleineArevalo/Reconocimiento-instrumentos-quirurgicos)

---

## 📄 Disponibilidad del código y licencia

### Contenido público

El repositorio puede incluir:

- Código de preparación.
- Código de entrenamiento.
- Código de predicción.
- Aplicación Streamlit.
- POC Gradio.
- Métricas.
- Documentación.
- Ejemplos autorizados.

### Contenido que debe permanecer privado

- Credenciales.
- Tokens.
- Imágenes no autorizadas.
- Información identificable.
- Datos externos sensibles.
- Archivos de retroalimentación privados.

### Licencia

```text
[DEFINIR LICENCIA DEL REPOSITORIO]
```

Opciones habituales:

- MIT.
- Apache-2.0.
- Licencia académica personalizada.
- Creative Commons para documentación.

> Antes de declarar una licencia, el equipo debe acordar qué usos y redistribuciones desea permitir.

---

## 📌 Estado del proyecto

| Elemento | Estado |
|---|---|
| Preparación de datos | Completada |
| Clasificación multilabel | Completada |
| MobileNetV2 | Integrada |
| Entrenamiento inicial | Completado |
| Fine-tuning | Completado |
| Evaluación interna | Completada |
| Umbrales por clase | Configurados |
| Modelo `.h5` | Exportado |
| Aplicación Streamlit | Implementada |
| Despliegue en la nube | Implementado |
| POC Gradio | Implementada |
| Inferencia multivista | Implementada |
| Validación humana | Implementada |
| Registro CSV/JSONL | Implementado |
| Validación externa amplia | Pendiente |
| Uso clínico | No autorizado |

---

## 🧾 Registro de cambios

### Versión 1.0.0 — 2026

- ✅ Organización del dataset.
- ✅ Vectores multilabel.
- ✅ MobileNetV2 con transferencia.
- ✅ Entrenamiento inicial.
- ✅ Fine-tuning.
- ✅ Evaluación con métricas multilabel.
- ✅ F1 por clase.
- ✅ Umbrales independientes.
- ✅ Exportación `.h5`.
- ✅ Aplicación Streamlit.
- ✅ Despliegue en Streamlit Community Cloud.
- ✅ Función propia de predicción.
- ✅ Tres vistas en inferencia.
- ✅ Estados de salida explícitos.
- ✅ Interfaz Gradio.
- ✅ Revisión humana.
- ✅ Registro CSV.
- ✅ Registro JSONL.
- ✅ Integridad SHA-256.
- ✅ Preparación para validación externa.
- ⬜ Métricas finales de validación externa.
- ⬜ Ampliación de clases.
- ⬜ Detección y conteo de objetos.

---

## 📖 Referencias

Abadi, M., Barham, P., Chen, J., Chen, Z., Davis, A., Dean, J., Devin, M., Ghemawat, S., Irving, G., Isard, M., Kudlur, M., Levenberg, J., Monga, R., Moore, S., Murray, D. G., Steiner, B., Tucker, P., Vasudevan, V., Warden, P., ... Zheng, X. (2016). TensorFlow: A system for large-scale machine learning. En *12th USENIX Symposium on Operating Systems Design and Implementation (OSDI 16)* (pp. 265–283). USENIX Association.

Gradio Team. (s. f.). *Gradio documentation*.

Keras Team. (s. f.). *Keras API documentation*.

Lavado, D., da Silva, J., & Caramelo, F. (2018). *Labeled surgical tools and images* [Conjunto de datos]. Kaggle.

Saito, T., & Rehmsmeier, M. (2015). The precision-recall plot is more informative than the ROC plot when evaluating binary classifiers on imbalanced datasets. *PLOS ONE, 10*(3), e0118432.

Sandler, M., Howard, A., Zhu, M., Zhmoginov, A., & Chen, L.-C. (2018). MobileNetV2: Inverted residuals and linear bottlenecks. En *2018 IEEE/CVF Conference on Computer Vision and Pattern Recognition* (pp. 4510–4520).

---

## ⚕️ Advertencia

SurgiVision AI es un prototipo académico y una prueba de concepto.

No está diseñado para:

- Uso clínico.
- Diagnóstico.
- Conteo quirúrgico oficial.
- Control hospitalario.
- Control de esterilización.
- Toma de decisiones médicas.
- Sustitución del personal.
- Sustitución de protocolos institucionales.

Toda predicción debe ser revisada por una persona responsable.

---

**Versión:** 1.0.0  
**Año:** 2026  
**Estado:** prototipo académico  
**Aplicación:** Streamlit Community Cloud  
**Modelo:** MobileNetV2 multilabel  
**Clases:** 4  

**Construido con:**

TensorFlow · Keras · MobileNetV2 · NumPy · Pandas · Pillow · Matplotlib · Scikit-learn · Streamlit · Gradio · Google Colab · GitHub


© 2026 SurgiVision AI — Clasificación multilabel de instrumentos quirúrgicos
