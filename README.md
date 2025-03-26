# Práctica #: Transformada de Hough

La transformada de Hough sirve para detectar líneas rectas, incluso si están cortadas. A continuación se presentan una serie de casos de prueba con los que se explorará la transformada de Hough.

## Instrucción

A partir del siguiente código, modifica los parámetros de la transformada de Hoguh de manera que se obtengan las líneas continuas deseadas para cada caso. Muestra tus resultados, anota los parámetros que usaste y justifica los cambios. Posteriormente, responde las siguientes preguntas:

1. ¿Qué es el parámetro *ρ* (rho)?
2. ¿Qué es el parámetro *θ* (theta)?
3. ¿Qué es el parámetro *threshold*?
3. ¿Qué es el parámetro *minLineLength*?
3. ¿Qué es el parámetro *maxLineGap*?

NOTA: Para cada caso se presentan dos imágenes. la imagen con las líneas coloreadas es el resultado que se espera obtener.

```python
# pyright: strict

import cv2
from cv2.typing import MatLike
import sys
import numpy as np
from typing import NamedTuple

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
    i = name.rfind('.')
    return name[:i] + '-out' + name[i:]

class HoughParams(NamedTuple):
    rho: float
    theta: float
    threshold: int
    minLineLength: float
    maxLineGap: float

class LineDetector:
    def __init__(self):
        pass

    def lines_too_close(self, a: LineLike, b: LineLike, threshold: float):
        def dist(x1: int, y1: int, x2: int, y2: int) -> float:
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

    def cleanup(self, lines: MatLike):
        while True:
            change = False

            for i in range(len(lines)):
                for j in range(len(lines)):
                    if i == j: continue

                    if self.lines_too_close(lines[i][0], lines[j][0], 15):
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
        lines = self.cleanup(lines)

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
```

## Caso 1

![ejemplo-1](ejemplo1.jpg)
![ejemplo-1-deseado](ejemplo1-deseado.jpg)

| $\rho$ | $\theta$ | threshold | minLineLength | maxLineGap |
| - | - | - | - | - |
| $\ $ | |  |  |  |

## Caso 2

![ejemplo-2](ejemplo2.jpg)
![ejemplo-2-deseado](ejemplo2-deseado.jpg)

| $\rho$ | $\theta$ | threshold | minLineLength | maxLineGap |
| - | - | - | - | - |
| $\ $ | |  |  |  |

## Caso 3

![ejemplo-3](ejemplo3.jpg)
![ejemplo-3-deseado](ejemplo3-deseado.jpg)

| $\rho$ | $\theta$ | threshold | minLineLength | maxLineGap |
| - | - | - | - | - |
| $\ $ | |  |  |  |
