import numpy as np
import matplotlib.pyplot as plt
import requests
from PIL import Image
from io import BytesIO


def descargar_imagen():
    url = "https://raw.githubusercontent.com/pytorch/hub/master/images/dog.jpg"
    response = requests.get(url)
    img_pil = Image.open(BytesIO(response.content))
    return np.array(img_pil)


def generar_ruido_gaussiano(imagen, media=0, sigma=50):
    """
    Agrega ruido normal (gaussiano) a la imagen.
    Sigma controla la intensidad del ruido.
    """
    ruido = np.random.normal(media, sigma, imagen.shape)
    imagen_ruidosa = imagen + ruido
    # Clip para asegurar que los valores estén entre 0 y 255
    return np.clip(imagen_ruidosa, 0, 255).astype(np.uint8)


def generar_ruido_sal_pimienta(imagen, probabilidad=0.05):
    """
    Agrega ruido de impulso:
    - Sal: Píxeles blancos aleatorios (255)
    - Pimienta: Píxeles negros aleatorios (0)
    """
    salida = np.copy(imagen)

    # Matriz de probabilidades aleatorias
    probs = np.random.random(imagen.shape[:2])

    # Aplicar Sal (Blanco) donde la probabilidad sea menor a la mitad del umbral
    salida[probs < (probabilidad / 2)] = 255

    # Aplicar Pimienta (Negro) donde la probabilidad sea mayor a 1 - mitad del umbral
    salida[probs > (1 - probabilidad / 2)] = 0

    return salida


def main():
    # 1. Cargar imagen original
    img_original = descargar_imagen()
    alto, ancho, canales = img_original.shape

    # -----------------------------------------------------------
    # PARTE 1: CANALES Y RECONSTRUCCIÓN ROJA
    # -----------------------------------------------------------

    # Separar canales (son matrices 2D)
    canal_R = img_original[:, :, 0]
    canal_G = img_original[:, :, 1]
    canal_B = img_original[:, :, 2]

    # Reconstruir imagen SOLO con el canal ROJO
    # Creamos una matriz de ceros del mismo tamaño que la original
    img_solo_roja = np.zeros_like(img_original)
    # Asignamos el canal R al índice 0, y dejamos G(1) y B(2) en cero
    img_solo_roja[:, :, 0] = canal_R

    # Visualización Parte 1
    fig1, ax1 = plt.subplots(2, 3, figsize=(12, 8))
    fig1.suptitle(
        "Parte 1: Separación de Canales y Reconstrucción", fontsize=16)

    # Original
    ax1[0, 0].imshow(img_original)
    ax1[0, 0].set_title("Original")
    ax1[0, 0].axis('off')

    # Solo Roja Reconstruida
    ax1[0, 1].imshow(img_solo_roja)
    ax1[0, 1].set_title("Reconstrucción Solo Canal Rojo\n(G=0, B=0)")
    ax1[0, 1].axis('off')

    # Espacio vacío para estética
    ax1[0, 2].axis('off')

    # Canales Individuales (Escala de grises)
    ax1[1, 0].imshow(canal_R, cmap='gray')
    ax1[1, 0].set_title("Canal R (Visualizado en Grises)")
    ax1[1, 0].axis('off')

    ax1[1, 1].imshow(canal_G, cmap='gray')
    ax1[1, 1].set_title("Canal G (Visualizado en Grises)")
    ax1[1, 1].axis('off')

    ax1[1, 2].imshow(canal_B, cmap='gray')
    ax1[1, 2].set_title("Canal B (Visualizado en Grises)")
    ax1[1, 2].axis('off')

    plt.tight_layout()

    # -----------------------------------------------------------
    # PARTE 2: RUIDO GAUSSIANO Y SAL/PIMIENTA
    # -----------------------------------------------------------

    img_gauss = generar_ruido_gaussiano(img_original, sigma=50)
    img_sp = generar_ruido_sal_pimienta(img_original, probabilidad=0.05)

    # Visualización Parte 2
    fig2, ax2 = plt.subplots(1, 3, figsize=(15, 5))
    fig2.suptitle("Parte 2: Comparación de Ruidos", fontsize=16)

    # Original
    ax2[0].imshow(img_original)
    ax2[0].set_title("Original")
    ax2[0].axis('off')

    # Ruido Gaussiano
    ax2[1].imshow(img_gauss)
    ax2[1].set_title("Ruido Gaussiano\n(Variación continua)")
    ax2[1].axis('off')

    # Ruido Sal y Pimienta
    ax2[2].imshow(img_sp)
    ax2[2].set_title("Ruido Sal y Pimienta\n(Píxeles muertos/saturados)")
    ax2[2].axis('off')

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()
