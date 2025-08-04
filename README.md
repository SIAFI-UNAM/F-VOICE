````markdown
# VITS2: Mejora de la Calidad y Eficiencia en Síntesis de Voz en una Sola Etapa mediante Aprendizaje Adversarial y Diseño Arquitectónico

## Jungil Kong, Jihoon Park, Beomjeong Kim, Jeongmin Kim, Dohee Kong, Sangjin Kim

Implementación no oficial del artículo [VITS2 paper](https://arxiv.org/abs/2307.16430), secuela del artículo original [VITS paper](https://arxiv.org/abs/2106.06103).  
Este repositorio está basado en [VITS2 p0p4k](https://github.com/p0p4k/vits2_pytorch).

![Alt text](resources/image.png)

## Descripción

VITS2 es un modelo de síntesis de voz en una sola etapa que mejora significativamente la naturalidad, eficiencia y accesibilidad en comparación con su predecesor. Entre sus ventajas destacan:

- Mayor naturalidad en la voz generada.
- Mejor adaptación a múltiples hablantes.
- Mayor eficiencia en entrenamiento e inferencia.
- Menor dependencia de conversión fonémica (enfoque end-to-end).

## Créditos

- [VITS repo](https://github.com/jaywalnut310/vits)
- [VITS2 p0p4k](https://github.com/p0p4k/vits2_pytorch)

---

## **Requisitos Previos**

1. **Python** >= 3.10  
2. **PyTorch** >= 2.7.0  
3. Crear y activar un entorno virtual  
4. Clonar este repositorio  
5. Instalar las dependencias:

```bash
pip install -r requirements.txt
````

6. Instalar **espeak-ng**

   * En Linux:

     ```bash
     sudo apt-get install espeak-ng
     ```
   * En Windows: seguir este [tutorial](https://www.youtube.com/watch?v=BBlivx6o0WM)

---

## **Inferencia (uso del modelo)**

Ya no es necesario ejecutar notebooks ni compilar módulos adicionales.
Para usar el modelo entrenado, simplemente ejecuta:

```bash
python app.py
```

* Descarga los modelos desde este [Google Drive](https://drive.google.com/drive/folders/1GDOh8VqPcJNO-0dKtMc_B5dAoU-0p9Ht?usp=sharing)
* Colócalo en una carpeta llamada `models/`

---

## **Entrenamiento**

### Preparación de datos

1. Asegúrate de que tus audios estén en **mono** y a **22050 Hz**:

```bash
ffmpeg -i input.wav -ac 1 -ar 22050 output.wav
```

2. Usa la app de segmentación (`Audio-segmentation`) para dividir los audios largos y generar el `metadata.csv`.

3. Ejecuta el script para generar los archivos `train.txt` y `val.txt` desde el `metadata.csv` generado:

```bash
python create_train_val.py
```

### Transfer Learning

Si deseas realizar fine-tuning sobre un modelo ya entrenado:

1. Crea una carpeta llamada `logs/NOMBRE_MODELO`
2. Coloca el checkpoint del modelo base dentro de esa carpeta
3. Renómbralo como:

```text
G_0.pth
```

Esto reiniciará el contador de pasos a cero, pero conservará los pesos del modelo anterior.

---

### Entrenamiento del modelo

1. Copia un archivo `.json` de configuración desde `configs/`, renómbralo (ej. `MYMODEL.json`) y modifícalo según tu dataset.
2. Si es en inglés, usa `"english_cleaners"`. Para español, deja `"spanish_cleaners"`.
3. Ejecuta:

```bash
python -m Train.trainer_engine --config configs/MYMODEL.json -m MYMODEL
```

---

## **Conversión a ONNX (opcional)**

Una vez entrenado el modelo, puedes convertirlo a ONNX.
Desde `src/`, usa el módulo `import onnx`.

---

## **Resumen**

| Objetivo                    | Acción                                                                  |
| --------------------------- | ----------------------------------------------------------------------- |
| Usar modelo entrenado       | `python app.py`                                                         |
| Entrenar modelo nuevo       | `python -m Train.trainer_engine --config configs/NOMBRE.json -m NOMBRE` |
| Preparar audios             | Convertir con ffmpeg a 22050Hz mono + segmentación automática           |
| Crear `train.txt`/`val.txt` | `python create_train_val.py` usando `metadata.csv` generado             |
| Transfer learning           | Crear `logs/MODELO/`, poner `G_0.pth`                                   |
| Convertir a ONNX            | Usar `import onnx` desde `src/` después del entrenamiento               |

---

## ToDos

1. Interfaz gráfica
2. Video tutorial de uso
3. Catálogo más amplio de voces
4. Voces en lenguas originarias
5. Integración con LLM para chat con voz

```
```
