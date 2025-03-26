# pyright: strict

import cv2
from cv2.typing import MatLike
import sys
import numpy as np
from typing import NamedTuple

# definición de tipo:
# type LineLike = np.ndarray[tuple[int, int, int, int], np.dtype[np.int_]] # para python 3.12+
LineLike = np.ndarray[tuple[int, int, int, int], np.dtype[np.int_]]

colors = [
    (0, 0, 255),
    (0, 255, 0),
    (255, 0, 0),
    (255, 255, 0),
    (255, 0, 255),
    (0, 255, 255),
]

def out_name(name: str) -> str:
    """
    Regresa un string con el nombre del archivo inicial seguido de -out y la extensión.
    Ejemplo: imagen1.jpg -> imagen1-out.jpg
    """

    i = name.rfind('.')
    return name[:i] + '-out' + name[i:]

class HoughParams(NamedTuple):
    """
    Empaqueta los parámetros del algoritmo de Hough.
    """

    rho: float
    theta: float
    threshold: int
    minLineLength: float
    maxLineGap: float

class LineDetector:
    """
    Contiene los métodos necesarios para detectar líneas en una imagen.
    """

    def __init__(self):
        pass

    def __lines_too_close(self, a: LineLike, b: LineLike, threshold: float) -> bool:
        """
        Revisa si dos líneas están muy cerca.
        Este método se usa en conjunto con LineDetector.__cleanup().
        """

        def dist(x1: int, y1: int, x2: int, y2: int) -> float:
            """
            Distancia euclidiana entre puntos (x1, y1) y (x2, y2).
            """
            return ((x2 - x1)**2 + (y2 - y1)**2)**0.5

        ax1, ay1, ax2, ay2 = a
        bx1, by1, bx2, by2 = b

        return (
            (
                dist(ax1, ay1, bx1, by1) < threshold and
                dist(ax2, ay2, bx2, by2) < threshold
            ) or (
                dist(ax1, ay1, bx2, by2) < threshold and
                dist(ax2, ay2, bx1, by1) < threshold
            )
        )

    def __cleanup(self, lines: MatLike) -> MatLike:
        """
        Revisa todos los pares de líneas. Si hay dos muy cerca, elimina una del par.
        Esto se hace con el propósito de mantener la imagen final limpia:

        Si la línea original se ve así:
        ```
        . . . . . . . . . . . . . . .
        . . . . . . . . . . . . . . .
        . . . # # # # # # # # # . . .
        . . . . . . . . . . . . . . .
        . . . . . . . . . . . . . . .
        ```

        ```
        Los contornos se forman así:
        . . . . . . . . . . . . . . .
        . . + - - - - - - - - - + . .
        . . | . . . . . . . . . | . .
        . . + - - - - - - - - - + . .
        . . . . . . . . . . . . . . .
        ```
        De manera que por una línea en la imagen original, se tienen cuatro líneas en el contorno.
        Las líneas más cortas son descartadas por el argumento `minLineLength`, pero que para
        las largas hay que realizar un procesamiento adicional para eliminar una de estas.

        Ese procesamiento adicional se realiza en este método.
        """

        while True:
            change = False

            for i in range(len(lines)):
                for j in range(len(lines)):
                    if i == j: continue

                    if self.__lines_too_close(lines[i][0], lines[j][0], 15):
                        lines = np.delete(lines, j, 0)
                        change = True
                        break

                if change: break
            if not change:
                break

        return lines

    def detect(
        self,
        source: MatLike,
        hough_params: HoughParams,
        show_image: bool = True,
        save_image: bool = False,
    ):
        """
        Detecta las líneas.
        """

        out = source.copy()
        gray = cv2.cvtColor(source, cv2.COLOR_BGR2GRAY)
        edges = cv2.Canny(gray, 70, 60)

        rho, theta, threshold, minLineLength, maxLineGap = [*hough_params]
        lines = cv2.HoughLinesP(
            edges,
            rho,
            theta,
            int(threshold),
            minLineLength=minLineLength,
            maxLineGap=maxLineGap,
        )
        lines = self.__cleanup(lines)

        i = 0
        for line in lines:
            x1, y1, x2, y2 = line[0]
            cv2.line(out, (x1, y1), (x2, y2), colors[i % len(colors)], 2)

        if show_image:
            cv2.imshow('lines', out)
            cv2.imshow('original', source)
            cv2.waitKey(0)

        if save_image:
            cv2.imwrite(out_name(sys.argv[1]), out)


if __name__ == '__main__':
    ld = LineDetector()
    params = HoughParams(
        rho=0.5,
        theta=np.pi / 180,
        threshold=20,
        minLineLength=30,
        maxLineGap=5
    )
    source = cv2.imread(sys.argv[1])

    ld.detect(source, params)
