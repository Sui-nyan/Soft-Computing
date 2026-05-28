import random
import math

WIDTH, HEIGHT = 800, 600

# Colors
BLACK = (0, 0, 0)
WHITE = (255, 255, 255)
RED = (255, 0, 0)
GREEN = (0, 255, 0)
BLUE = (0, 0, 255)
YELLOW = (255, 255, 0)

for _ in range(10):
    x = random.randint(20, WIDTH - 20)
    y = random.randint(20, HEIGHT - 20)
    print(f"({x},{y}),",end=" ")
print()
print()
for _ in range(10):
    x = random.randint(0, WIDTH)
    y = random.randint(0, HEIGHT)
    print(f"({x},{y}),",end=" ")
print()
print()
for _ in range(10):
    radius = random.randint(15, 30)
    print(f"{radius},", end=" ")
print()
print()
for _ in range(10):
    speed_x = random.uniform(-2, 2)
    speed_y = random.uniform(-2, 2)

    print(f"({speed_x},{speed_y}),",end=" ")