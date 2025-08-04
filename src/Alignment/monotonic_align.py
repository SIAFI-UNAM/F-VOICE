import numpy as np
import numba
import torch

# El decorador @numba.jit compila esta función a código máquina para un rendimiento máximo.
# nopython=True asegura que no se recurra al intérprete de Python, lo cual es más rápido.
# nogil=True libera el "Global Interpreter Lock" de Python, permitiendo la ejecución
# paralela real en la función que llama a esta.
@numba.jit(nopython=True, nogil=True)
def maximum_path_each(path, value, t_y, t_x, max_neg_val=-1e9):
    """
    Calcula la ruta de máxima verosimilitud para UN solo elemento.
    Esta es la lógica interna que se ejecutará a alta velocidad.
    """
    # Pasada hacia adelante (programación dinámica)
    for y in range(t_y):
        for x in range(max(0, t_x + y - t_y), min(t_x, y + 1)):
            # Valor de la celda de arriba
            if x == y:
                v_cur = max_neg_val
            else:
                v_cur = value[y - 1, x]
            
            # Valor de la celda de arriba a la izquierda
            if x == 0:
                v_prev = 0. if y == 0 else max_neg_val
            else:
                v_prev = value[y - 1, x - 1]
            
            value[y, x] += max(v_prev, v_cur)

    # Pasada hacia atrás (backtracking)
    index = t_x - 1
    for y in range(t_y - 1, -1, -1):
        path[y, index] = 1
        if index != 0 and (index == y or value[y - 1, index] < value[y - 1, index - 1]):
            index -= 1


# El decorador @numba.jit con parallel=True habilita la paralelización automática.
@numba.jit(nopython=True, parallel=True)
def maximum_path_numba(paths, values, t_ys, t_xs):
    """
    Función paralelizada que procesa el batch completo.
    """
    b = paths.shape[0]
    
    # numba.prange distribuye las iteraciones de este bucle entre múltiples hilos (cores).
    # Cada llamada a maximum_path_each(i) se ejecuta en paralelo para diferentes 'i'.
    for i in numba.prange(b):
        maximum_path_each(paths[i], values[i], t_ys[i], t_xs[i])


# Función principal que usa el resto del programa
def maximum_path(neg_cent, mask):
    device = neg_cent.device
    dtype = neg_cent.dtype
    neg_cent = neg_cent.data.cpu().numpy().astype(np.float32)
    path = np.zeros(neg_cent.shape, dtype=np.int32)
    t_t_max = mask.sum(1)[:, 0].data.cpu().numpy().astype(np.int32)
    t_s_max = mask.sum(2)[:, 0].data.cpu().numpy().astype(np.int32)
    maximum_path_numba(path, neg_cent, t_t_max, t_s_max)
    return torch.from_numpy(path).to(device=device, dtype=dtype)