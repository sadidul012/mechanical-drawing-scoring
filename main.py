import cv2
import numpy as np
import matplotlib.pyplot as plt
from pdf2image import convert_from_path

pages = convert_from_path('data/001_Grundblech.pdf', dpi=300)
img = np.array(pages[0])
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
edges = cv2.Canny(gray, 50, 150, apertureSize=3)
lines = cv2.HoughLinesP(edges, 1, np.pi / 180, threshold=100, minLineLength=50, maxLineGap=10)
if lines is not None:
    for line in lines:
        x1, y1, x2, y2 = line[0]
        cv2.line(img, (x1, y1), (x2, y2), (0, 255, 0), 2)

plt.figure(figsize=(12, 10))
# cv2.imwrite('output.png', img)
plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
plt.axis('off')
plt.show()
