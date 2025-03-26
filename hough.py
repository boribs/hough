import cv2
import sys
import numpy as np
from typing import NamedTuple

colors = [
    (0, 0, 255),
    (0, 255, 0),
    (255, 0, 0),
    (255, 255, 0),
    (255, 0, 255),
    (0, 255, 255),
]

def out(name):
    i = name.rfind('.')
    return name[:i] + '-out' + name[i:]

class HoughParams(NamedTuple):
    rho: float
    theta: float
    threshold: float
    minLineLength: float
    maxLineGap: float

class LineDetector:
    def __init__(self):
        pass

    def lines_too_close(self, a, b, threshold):
        def dist(x1, y1, x2, y2):
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

    def cleanup(self, lines):
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
        source,
        hough_params,
        canny_thresholds = (70, 60),
        show_image = True,
        save_image = False,
    ):
        out = source.copy()
        gray = cv2.cvtColor(source, cv2.COLOR_BGR2GRAY)
        edges = cv2.Canny(gray, *canny_thresholds)

        lines = cv2.HoughLinesP(edges, *hough_params)
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
            cv2.imwrite(out(sys.argv[1], out))

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
