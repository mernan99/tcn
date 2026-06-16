from PIL import Image
import numpy as np

# Image size
width = 2048
height = 2048

# Black background = dead cells
img = np.zeros((height, width), dtype=np.uint8)

# One glider pattern
# 0 1 0
# 0 0 1
# 1 1 1
glider = np.array([
    [0, 255, 0],
    [0, 0, 255],
    [255, 255, 255]
], dtype=np.uint8)

# 16 positions for the gliders
# Keep them far apart so they do not overlap
positions = [
    (100, 100),   (300, 100),   (500, 100),   (700, 100),
    (100, 300),   (300, 300),   (500, 300),   (700, 300),
    (100, 500),   (300, 500),   (500, 500),   (700, 500),
    (100, 700),   (300, 700),   (500, 700),   (700, 700),
]

# Place each glider
for x, y in positions:
    img[y:y+3, x:x+3] = glider

# Save as PNG
Image.fromarray(img).save("cg_16_gliders.png")

print("Saved cg_16_gliders.png with 16 glider patterns")