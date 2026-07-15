# Clasificación multilabel de instrumentos quirúrgicos

Proyecto académico de inteligencia artificial para reconocer la presencia de varios instrumentos quirúrgicos en una misma imagen.

El sistema utiliza una red neuronal convolucional basada en **MobileNetV2**, transferencia de aprendizaje y una salida sigmoide para clasificación multilabel. El modelo fue integrado en una aplicación web desarrollada con **Streamlit** y desplegada en **Streamlit Community Cloud**.

## Aplicación web

La aplicación permite:

- Subir una fotografía desde el dispositivo.
- Capturar una imagen mediante la cámara.
- Analizar la imagen con el modelo entrenado.
- Mostrar la probabilidad de cada instrumento.
- Comparar las probabilidades con umbrales específicos.
- Reconocer varias clases simultáneamente.

### Acceso directo

[**Abrir la aplicación web**]([https://reconocimiento-instrumentos-quirurgicos-ewuruhnvnvwnkdijjskntl.streamlit.app](https://reconocimiento-instrumentos-quirurgicos-ewuruhnvvnwnkdijjskntl.streamlit.app)

> La aplicación puede tardar unos segundos en iniciar si estuvo inactiva.

## Clases reconocidas

El modelo reconoce cuatro clases:

1. Bisturí n.º 4.
2. Pinza de disección recta.
3. Tijera Mayo recta.
4. Tijera Mayo curva.

Al ser un problema multilabel, una imagen puede contener una o varias clases al mismo tiempo.

## Funcionamiento general

1. El usuario carga o captura una imagen.
2. La imagen se redimensiona a `224 × 224` píxeles.
3. El modelo genera una probabilidad para cada clase.
4. Cada probabilidad se compara con su umbral.
5. La aplicación presenta los instrumentos reconocidos y su nivel de confianza.

Ejemplo de salida:

```text
[1, 0, 1, 0]
```

El vector anterior indica la presencia de un bisturí n.º 4 y una tijera Mayo recta.

## Arquitectura del modelo

- Modelo base: **MobileNetV2** preentrenada con ImageNet.
- Técnica: transferencia de aprendizaje.
- Entrada: imágenes de `224 × 224 × 3`.
- Salida: cuatro neuronas con activación sigmoide.
- Función de pérdida: `BinaryCrossentropy`.
- Estrategia: entrenamiento inicial y fine-tuning.
- Formato del modelo final: `.h5`.

## Resultados

| Métrica | Resultado |
|---|---:|
| AUC | 0.9434 |
| Exactitud binaria | 0.8591 |
| Precisión micro | 0.8053 |
| Recall micro | 0.7839 |
| F1 macro | 0.7937 |
| Exactitud multilabel exacta | 0.6514 |

### F1 por clase

| Instrumento | F1 |
|---|---:|
| Bisturí n.º 4 | 0.8065 |
| Pinza de disección recta | 0.7629 |
| Tijera Mayo recta | 0.7635 |
| Tijera Mayo curva | 0.8419 |

### Umbrales de decisión

| Instrumento | Umbral |
|---|---:|
| Bisturí n.º 4 | 0.36 |
| Pinza de disección recta | 0.35 |
| Tijera Mayo recta | 0.47 |
| Tijera Mayo curva | 0.50 |

## Contenido del repositorio

```text
Reconocimiento-instrumentos-quirurgicos/
├── C1_Carga_Preparacion_Datos.ipynb
├── C2_Entrenamiento_Multilabel.ipynb
├── C3_Prediccion_Conclusiones_POC.ipynb
├── modelo_instrumentos_multilabel.h5
├── app.py
├── requirements.txt
└── README.md
```

## Descripción de los cuadernos

### `C1_Carga_Preparacion_Datos.ipynb`

Incluye:

- Descarga y organización del dataset.
- Lectura de imágenes y etiquetas.
- Conversión de etiquetas a vectores multilabel.
- Análisis exploratorio.
- División de entrenamiento, validación y prueba.

### `C2_Entrenamiento_Multilabel.ipynb`

Incluye:

- Creación de datasets con TensorFlow.
- Aumento de datos.
- Construcción de MobileNetV2.
- Entrenamiento inicial.
- Fine-tuning.
- Evaluación del modelo.
- Matrices de confusión.
- Optimización de umbrales.
- Exportación del modelo en formato `.h5`.

### `C3_Prediccion_Conclusiones_POC.ipynb`

Incluye:

- Carga del modelo entrenado.
- Función propia de predicción.
- Preprocesamiento de imágenes externas.
- Aplicación de umbrales por clase.
- Visualización de probabilidades.
- Conclusiones del proyecto.

## Ejecución local

Clonar el repositorio:

```bash
git clone https://github.com/MadeleineArevalo/Reconocimiento-instrumentos-quirurgicos.git
cd Reconocimiento-instrumentos-quirurgicos
```

Instalar las dependencias:

```bash
pip install -r requirements.txt
```

Ejecutar la aplicación:

```bash
streamlit run app.py
```

La aplicación se abrirá normalmente en:

```text
http://localhost:8501
```

## Tecnologías utilizadas

- Python.
- TensorFlow y Keras.
- MobileNetV2.
- NumPy.
- Pandas.
- Pillow.
- Streamlit.
- Google Colab.
- GitHub.
- Streamlit Community Cloud.

## Limitaciones

Este sistema realiza clasificación multilabel sobre la imagen completa.

Por tanto:

- Reconoce qué clases están presentes.
- Puede reconocer varias clases simultáneamente.
- No localiza los instrumentos mediante cajas delimitadoras.
- No determina la posición exacta de cada objeto.
- No cuenta varias unidades de una misma clase.
- Puede presentar errores con fondos, iluminación u orientaciones diferentes a las del dataset.

## Advertencia

Este proyecto es un prototipo académico y una prueba de concepto. No está diseñado para uso clínico, diagnóstico, control hospitalario ni toma de decisiones médicas. Los resultados deben ser verificados por una persona responsable.
