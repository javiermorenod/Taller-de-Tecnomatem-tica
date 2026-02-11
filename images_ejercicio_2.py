import numpy as np
import matplotlib.pyplot as plt
import requests
from PIL import Image
from io import BytesIO
# Importamos scipy.ndimage para realizar la convolución de forma eficiente
import scipy.ndimage as ndimage


def procesar_imagen_con_bordes():
    # ===========================================================
    # 1. OBTENCIÓN Y PREPROCESAMIENTO (Igual que antes)
    # ===========================================================
    url = "https://raw.githubusercontent.com/pytorch/hub/master/images/dog.jpg"
    print(f"Descargando imagen...")
    try:
        response = requests.get(url)
        response.raise_for_status()
        img_pil = Image.open(BytesIO(response.content))
        matriz_rgb = np.array(img_pil)
    except Exception as e:
        print(f"Error: {e}")
        return

    # Escala de Grises (Fórmula de luminancia)
    r, g, b = matriz_rgb[:, :, 0], matriz_rgb[:, :, 1], matriz_rgb[:, :, 2]
    matriz_gris = 0.2989 * r + 0.5870 * g + 0.1140 * b

    # ===========================================================
    # 2. ESTRATEGIA SOLICITADA: UMBRALIZADO PREVIO
    # ===========================================================
    # El enunciado pide: pixeles > 100 llevarlos a 255, el resto a 0.
    # Esto crea una imagen "binaria" (solo blanco y negro puro).
    UMBRAL = 100

    # np.where funciona así: np.where(condicion, valor_si_true, valor_si_false)
    matriz_binaria = np.where(matriz_gris > UMBRAL, 255.0, 0.0)

    # ===========================================================
    # 3. DETECCIÓN DE BORDES (Convolución)
    # ===========================================================
    # Definimos un kernel (filtro) Laplaciano básico.
    # Este filtro busca cambios bruscos de intensidad en todas direcciones.
    kernel_laplaciano = np.array([
        [0, -1,  0],
        [-1, 4, -1],
        [0, -1,  0]
    ])

    # Aplicamos la convolución sobre la imagen BINARIA (según la estrategia del enunciado)
    # ndimage.convolve pasa el kernel por toda la imagen.
    bordes_detectados = ndimage.convolve(matriz_binaria, kernel_laplaciano)

    # La convolución puede dar valores negativos. Para visualizar bordes,
    # nos interesa la magnitud del cambio, así que tomamos el valor absoluto.
    # También nos aseguramos que los valores estén entre 0 y 255.
    matriz_bordes_final = np.clip(np.abs(bordes_detectados), 0, 255)

    # ===========================================================
    # 4. VISUALIZACIÓN
    # ===========================================================
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    # A) Original
    axes[0].imshow(matriz_rgb)
    axes[0].set_title("1. Imagen Original (RGB)")
    axes[0].axis('off')

    # B) Escala de Grises
    axes[1].imshow(matriz_gris, cmap='gray', vmin=0, vmax=255)
    axes[1].set_title("2. Escala de Grises")
    axes[1].axis('off')

    # C) Bordes (Resultado de aplicar filtro a la imagen umbralizada)
    axes[2].imshow(matriz_bordes_final, cmap='gray', vmin=0, vmax=255)
    # Añadimos una nota al título explicando la estrategia usada
    axes[2].set_title(
        f"3. Bordes Detectados\n(Estrategia: Umbral > {UMBRAL} + Filtro Laplaciano)")
    axes[2].axis('off')

    plt.tight_layout()
    print("Mostrando resultados...")
    plt.show()


if __name__ == "__main__":
    procesar_imagen_con_bordes()
