# Clasificación multilabel de instrumentos quirúrgicos

Aplicación web académica para reconocer la presencia de cuatro instrumentos quirúrgicos en una imagen:

- Bisturí n.º 4
- Pinza de disección recta
- Tijera Mayo recta
- Tijera Mayo curva

El modelo utiliza **MobileNetV2**, transferencia de aprendizaje y una salida sigmoide multilabel. La interfaz fue desarrollada con **Streamlit**.

## Ejecutar localmente

```bash
pip install -r requirements.txt
streamlit run app.py
```

## Archivos necesarios

- `app.py`
- `modelo_instrumentos_multilabel.h5`
- `requirements.txt`
- `.streamlit/config.toml`

## Advertencia

Este proyecto es un prototipo académico. No localiza objetos mediante cajas y no sustituye la verificación profesional.
