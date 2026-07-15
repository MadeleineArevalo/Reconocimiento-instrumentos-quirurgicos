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
- Clasifica la imagen completa.
- Puede verse afectado por iluminación, fondo, orientación y oclusiones.
- Es un prototipo académico y no constituye un dispositivo médico.

## Video demostrativo

El video demostrativo del proyecto tiene una duración inferior a dos minutos.

[Ver video demostrativo](PEGAR_AQUI_EL_ENLACE_DEL_VIDEO)

## Autores

**Madeleine Arévalo, Tatiana Aucapiña, Esteban Tuquiñagui**  
Carrera de Biomedicina  
Universidad Politécnica Salesiana, sede Cuenca  
2026


## Referencias

- Abadi, M., et al. (2016). TensorFlow: A system for large-scale machine
  learning. En *12th USENIX Symposium on Operating Systems Design and
  Implementation (OSDI 16)*, 265–283.
  https://www.usenix.org/conference/osdi16/technical-sessions/presentation/abadi

- Gradio Team. (s. f.). *Gradio documentation*.
  https://www.gradio.app/docs

- Keras Team. (s. f.). *Keras API documentation*.
  https://keras.io/api/

- Lavado, D., da Silva, J., y Caramelo, F. (2018).
  *Labeled surgical tools and images* [Conjunto de datos]. Kaggle.
  https://www.kaggle.com/datasets/dilavado/labeled-surgical-tools

- Sandler, M., Howard, A., Zhu, M., Zhmoginov, A., y Chen, L.-C. (2018).
  MobileNetV2: Inverted residuals and linear bottlenecks.
  En *2018 IEEE/CVF Conference on Computer Vision and Pattern Recognition*,
  4510–4520.
  https://doi.org/10.1109/CVPR.2018.00474
