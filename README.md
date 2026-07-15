# Reconocimiento multilabel de instrumentos quirúrgicos

Proyecto académico de aprendizaje profundo para reconocer uno o varios
instrumentos quirúrgicos presentes en una imagen.

## Objetivo

Desarrollar un modelo de clasificación multilabel e implementarlo en una
aplicación web como prueba de concepto.

## Instrumentos reconocidos

- Bisturí n.º 4
- Pinza de disección recta
- Tijera Mayo recta
- Tijera Mayo curva

## Cuadernos del proyecto

- `C1_Carga_Preparacion_Datos.ipynb`: carga, preparación y exploración de datos.
- `C2_Entrenamiento_Multilabel.ipynb`: entrenamiento y evaluación del modelo.
- `C3_Prediccion_Conclusiones_POC.ipynb`: función de predicción, conclusiones y aplicación web Gradio.

## Modelo entrenado

El archivo `modelo_instrumentos_multilabel.h5` contiene el modelo final
entrenado mediante transferencia de aprendizaje con MobileNetV2.

## Tecnologías utilizadas

- Python
- TensorFlow y Keras
- MobileNetV2
- NumPy y Pandas
- Matplotlib
- Pillow
- Gradio
- Google Colab

## Prueba de concepto

La POC consiste en una aplicación web desarrollada con Gradio. El usuario
carga una imagen y el sistema muestra:

- Instrumentos reconocidos.
- Probabilidad por clase.
- Umbral de decisión.
- Tabla y gráfica de resultados.

## Ejecución

1. Abrir `C3_Prediccion_Conclusiones_POC.ipynb` en Google Colab.
2. Ejecutar las celdas en orden.
3. Cargar `modelo_instrumentos_multilabel.h5`.
4. Ejecutar la interfaz Gradio.
5. Cargar una imagen y presionar **Analizar imagen**.

## Limitaciones

- Solo reconoce las cuatro clases utilizadas en el entrenamiento.
- Clasifica la imagen completa y no dibuja cajas delimitadoras.
- Puede verse afectado por iluminación, fondo, orientación y oclusiones.
- Es un prototipo académico y no constituye un dispositivo médico.

## Video demostrativo

El video demostrativo del proyecto tiene una duración inferior a dos minutos.

[Ver video demostrativo](PEGAR_AQUI_EL_ENLACE_DEL_VIDEO)

## Autora

**Madeleine Arévalo**  
Carrera de Biomedicina  
Universidad Politécnica Salesiana, sede Cuenca  
2026
