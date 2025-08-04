import math
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F


import torch
import torch.nn.functional as F
import math

"""
### 1. `init_weights`
**Propósito:** Inicializar los pesos de las **capas convolucionales** en una red.
Esta función se suele aplicar a un modelo completo con `model.apply(init_weights)`.

- `classname = m.__class__.__name__`: Obtiene el nombre de la clase del módulo (ej. "Conv2d").
- `if classname.find("Conv") != -1:`: Comprueba si el nombre de la clase contiene "Conv".
- `m.weight.data.normal_(mean, std)`: Si es una capa convolucional, inicializa sus pesos
  tomando muestras de una **distribución normal** con la media y desviación estándar dadas. Esto
  ayuda a que el entrenamiento comience de manera estable.
"""
def init_weights(m, mean=0.0, std=0.01):
    classname = m.__class__.__name__
    if classname.find("Conv") != -1:
        m.weight.data.normal_(mean, std)

# ---

"""
### 2. `get_padding`
**Propósito:** Calcular el valor de `padding` necesario para una convolución de tipo **"same"**.
Una convolución "same" es aquella donde la dimensión de salida es igual a la de entrada.

- La fórmula `(kernel_size * dilation - dilation) / 2` calcula cuántos píxeles
  se deben añadir a cada lado de la secuencia para mantener su longitud original
  después de la convolución.
"""
def get_padding(kernel_size, dilation=1):
    return int((kernel_size * dilation - dilation) / 2)

# ---

"""
### 3. `convert_pad_shape`
**Propósito:** Transformar el formato de una lista de padding para que sea compatible con `torch.nn.functional.pad`.
PyTorch espera el padding en el formato `(pad_izq_dimN, pad_der_dimN, pad_izq_dimN-1, pad_der_dimN-1, ...)`.

- `l = pad_shape[::-1]`: Invierte el orden de las dimensiones. `F.pad` espera
  el padding de la última dimensión primero.
- `[item for sublist in l for item in sublist]`: Aplana la lista de listas
  (ej. `[[1, 1], [0, 0]]` -> `[1, 1, 0, 0]`) al formato que `F.pad` requiere.
"""
def convert_pad_shape(pad_shape):
    l = pad_shape[::-1]
    pad_shape = [item for sublist in l for item in sublist]
    return pad_shape

# ---

"""
### 4. `intersperse`
**Propósito:** Intercalar un elemento entre cada uno de los elementos de una lista.
Es muy útil en modelos de Texto a Voz (TTS) para insertar tokens de "silencio" o "blanco"
entre los tokens de fonemas.

- `result = [item] * (len(lst) * 2 + 1)`: Crea una lista llena del `item` a intercalar,
  con el tamaño final correcto.
- `result[1::2] = lst`: Coloca los elementos de la lista original `lst` en las
  posiciones impares de la lista `result`.
- **Ejemplo:** `intersperse([1, 2, 3], 0)` -> `[0, 1, 0, 2, 0, 3, 0]`
"""
def intersperse(lst, item):
    result = [item] * (len(lst) * 2 + 1)
    result[1::2] = lst
    return result

# ---

"""
### 5. `kl_divergence`
**Propósito:** Calcular la **Divergencia de Kullback-Leibler (KL)** entre dos distribuciones Gaussianas.
Esta métrica mide cuán diferente es una distribución (Q) de otra de referencia (P).
Es fundamental en los **Autoencoders Variacionales (VAEs)** para regularizar el espacio latente.

- `m_p`, `logs_p`: La media y el logaritmo de la varianza de la distribución P (ej. la posterior).
- `m_q`, `logs_q`: La media y el logaritmo de la varianza de la distribución Q (ej. la prior, usualmente una Normal estándar).
"""
def kl_divergence(m_p, logs_p, m_q, logs_q):
    """KL(P||Q)"""
    kl = (logs_q - logs_p) - 0.5
    kl += (
        0.5 * (torch.exp(2.0 * logs_p) + ((m_p - m_q) ** 2)) * torch.exp(-2.0 * logs_q)
    )
    return kl

# ---

"""
### 6. `rand_gumbel` y `rand_gumbel_like`
**Propósito:** Muestrear de una **distribución Gumbel**.
Esta técnica es clave en el "reparameterization trick" para distribuciones categóricas,
como en el **Gumbel-Softmax**, permitiendo que el gradiente fluya a través de un proceso de muestreo.

- `rand_gumbel`: Genera un tensor de la forma `shape` con valores de Gumbel. El truco
  de sumar/multiplicar `uniform_samples` es para evitar que los valores sean exactamente 0 o 1,
  lo que causaría `-inf` o `+inf` en los logaritmos.
- `rand_gumbel_like`: Una función de conveniencia para generar una muestra de Gumbel con la
  misma forma, tipo de dato y dispositivo que otro tensor `x`.
"""
def rand_gumbel(shape):
    """Sample from the Gumbel distribution, protect from overflows."""
    uniform_samples = torch.rand(shape) * 0.99998 + 0.00001
    return -torch.log(-torch.log(uniform_samples))


def rand_gumbel_like(x):
    g = rand_gumbel(x.size()).to(dtype=x.dtype, device=x.device)
    return g

# ---

"""
### 7. `slice_segments` y `rand_slice_segments`
**Propósito:** Extraer segmentos (trozos) de un tensor de secuencias.
Esto se usa comúnmente para entrenar con porciones más pequeñas de datos largos (como audio),
lo que ahorra memoria y estabiliza el entrenamiento.

- `slice_segments`: Extrae un segmento de tamaño `segment_size` de cada elemento
  en el batch `x`, comenzando en los índices especificados por `ids_str`.
- `rand_slice_segments`: Primero calcula **índices de inicio aleatorios** para cada secuencia
  en el batch y luego usa `slice_segments` para extraer esos trozos aleatorios.
"""
def slice_segments(x, ids_str, segment_size=4):
    ret = torch.zeros_like(x[:, :, :segment_size])
    for i in range(x.size(0)):
        idx_str = ids_str[i]
        idx_end = idx_str + segment_size
        ret[i] = x[i, :, idx_str:idx_end]
    return ret


def rand_slice_segments(x, x_lengths=None, segment_size=4):
    b, d, t = x.size()
    if x_lengths is None:
        x_lengths = t
    ids_str_max = x_lengths - segment_size + 1
    ids_str = (torch.rand([b]).to(device=x.device) * ids_str_max).to(dtype=torch.long)
    ret = slice_segments(x, ids_str, segment_size)
    return ret, ids_str

# ---

"""
### 8. `get_timing_signal_1d` y sus variantes
**Propósito:** Generar y aplicar el **Positional Encoding Sinusoidal** del paper original de Transformer "Attention Is All You Need".
Inyecta información sobre la posición absoluta de cada token en la secuencia.

- `get_timing_signal_1d`: Crea la matriz de codificación posicional usando funciones
  seno y coseno a diferentes frecuencias.
- `add_timing_signal_1d`: **Suma** la señal posicional a la entrada `x`.
- `cat_timing_signal_1d`: **Concatena** la señal posicional a la entrada `x` a lo largo
  de un eje específico.
"""
def get_timing_signal_1d(length, channels, min_timescale=1.0, max_timescale=1.0e4):
    position = torch.arange(length, dtype=torch.float)
    num_timescales = channels // 2
    log_timescale_increment = math.log(float(max_timescale) / float(min_timescale)) / (
        num_timescales - 1
    )
    inv_timescales = min_timescale * torch.exp(
        torch.arange(num_timescales, dtype=torch.float) * -log_timescale_increment
    )
    scaled_time = position.unsqueeze(0) * inv_timescales.unsqueeze(1)
    signal = torch.cat([torch.sin(scaled_time), torch.cos(scaled_time)], 0)
    signal = F.pad(signal, [0, 0, 0, channels % 2])
    signal = signal.view(1, channels, length)
    return signal


def add_timing_signal_1d(x, min_timescale=1.0, max_timescale=1.0e4):
    b, channels, length = x.size()
    signal = get_timing_signal_1d(length, channels, min_timescale, max_timescale)
    return x + signal.to(dtype=x.dtype, device=x.device)


def cat_timing_signal_1d(x, min_timescale=1.0, max_timescale=1.0e4, axis=1):
    b, channels, length = x.size()
    signal = get_timing_signal_1d(length, channels, min_timescale, max_timescale)
    return torch.cat([x, signal.to(dtype=x.dtype, device=x.device)], axis)

# ---

"""
### 9. `subsequent_mask`
**Propósito:** Crear una **máscara causal** para un decodificador de Transformer.
Esta máscara asegura que, al predecir un token, el modelo solo pueda "ver" (atender a)
los tokens anteriores y el token actual, pero no los futuros.

- `torch.tril(...)`: Crea una matriz triangular inferior (lower-triangular), donde
  los elementos por encima de la diagonal principal son cero. Esto implementa la causalidad.
"""
def subsequent_mask(length):
    mask = torch.tril(torch.ones(length, length)).unsqueeze(0).unsqueeze(0)
    return mask

# ---

"""
### 10. `fused_add_tanh_sigmoid_multiply`
**Propósito:** Implementar una **Gated Activation Unit (GAU)** de forma optimizada.
Esta es una función de activación común en modelos como WaveNet.

- `@torch.jit.script`: Compila la función con el Just-In-Time (JIT) compiler de PyTorch
  para que se ejecute más rápido, fusionando las operaciones.
- La función divide la entrada en dos mitades, aplica `tanh` a la primera y `sigmoid`
  a la segunda (la puerta o "gate"), y luego las multiplica.
"""
@torch.jit.script
def fused_add_tanh_sigmoid_multiply(input_a, input_b, n_channels):
    n_channels_int = n_channels[0]
    in_act = input_a + input_b
    t_act = torch.tanh(in_act[:, :n_channels_int, :])
    s_act = torch.sigmoid(in_act[:, n_channels_int:, :])
    acts = t_act * s_act
    return acts

# ---

"""
### 11. `shift_1d`
**Propósito:** Desplazar una secuencia un paso hacia la derecha.
Esto se usa a menudo en los decodificadores autorregresivos, donde la entrada en el
paso de tiempo `t` es la salida del paso `t-1`.

- `F.pad(..., [1, 0])`: Añade un cero al principio de la secuencia.
- `[:, :, :-1]`: Elimina el último elemento de la secuencia. El resultado es que toda
  la secuencia se ha desplazado un lugar.
"""
def shift_1d(x):
    x = F.pad(x, convert_pad_shape([[0, 0], [0, 0], [1, 0]]))[:, :, :-1]
    return x

# ---

"""
### 12. `sequence_mask`
**Propósito:** Crear una máscara booleana a partir de un tensor de longitudes.
Dado un vector de longitudes `[2, 5, 3]`, crea una matriz donde cada fila tiene `True`
hasta la longitud indicada y `False` después.

- `x.unsqueeze(0) < length.unsqueeze(1)`: Aprovecha el broadcasting de PyTorch.
  Compara un vector fila `[0, 1, 2, 3, 4]` con un vector columna `[[2], [5], [3]]`
  para generar la máscara 2D de forma eficiente.
"""
def sequence_mask(length, max_length=None):
    if max_length is None:
        max_length = length.max()
    x = torch.arange(max_length, dtype=length.dtype, device=length.device)
    return x.unsqueeze(0) < length.unsqueeze(1)

# ---

"""
### 13. `generate_path`
**Propósito:** Generar una matriz de alineamiento (path) a partir de duraciones.
Esta función es central en los **alineadores monotónicos** como el de VITS. Convierte
las duraciones predichas para cada token de entrada (ej. fonemas) en una matriz
binaria que alinea la secuencia de entrada con la de salida (ej. espectrograma).

- `cum_duration = torch.cumsum(duration, -1)`: Calcula la suma acumulada para saber
  dónde termina cada token de entrada en la escala de la secuencia de salida.
- El resto de las operaciones usan estas sumas acumuladas para construir una matriz
  (el `path`) donde cada fila de la secuencia de salida se alinea con una única
  columna de la secuencia de entrada.
"""
def generate_path(duration, mask):
    """
    duration: [b, 1, t_x]
    mask: [b, 1, t_y, t_x]
    """
    device = duration.device

    b, _, t_y, t_x = mask.shape
    cum_duration = torch.cumsum(duration, -1)

    cum_duration_flat = cum_duration.view(b * t_x)
    path = sequence_mask(cum_duration_flat, t_y).to(mask.dtype)
    path = path.view(b, t_x, t_y)
    path = path - F.pad(path, convert_pad_shape([[0, 0], [1, 0], [0, 0]]))[:, :-1]
    path = path.unsqueeze(1).transpose(2, 3) * mask
    return path

# ---

"""
### 14. `clip_grad_value_`
**Propósito:** Implementar una forma de **recorte de gradientes (gradient clipping)**.
Esta técnica previene que los gradientes se vuelvan demasiado grandes durante el
entrenamiento (problema de "exploding gradients"), lo que desestabilizaría el modelo.

- `p.grad.data.clamp_(min=-clip_value, max=clip_value)`: Esta es la línea clave.
  Recorta el valor de cada gradiente para que esté dentro del rango `[-clip_value, clip_value]`.
- La función también calcula y devuelve la **norma total** de los gradientes (antes del recorte por valor),
  lo que es útil para monitorear el entrenamiento.
"""
def clip_grad_value_(parameters, clip_value, norm_type=2):
    if isinstance(parameters, torch.Tensor):
        parameters = [parameters]
    parameters = list(filter(lambda p: p.grad is not None, parameters))
    norm_type = float(norm_type)
    if clip_value is not None:
        clip_value = float(clip_value)

    total_norm = 0
    for p in parameters:
        param_norm = p.grad.data.norm(norm_type)
        total_norm += param_norm.item() ** norm_type
        if clip_value is not None:
            p.grad.data.clamp_(min=-clip_value, max=clip_value)
    total_norm = total_norm ** (1.0 / norm_type)
    return total_norm