import cv2
import numpy as np
import random
import os

class Shape:
    def __init__(self, width, height, channels, image):
        self.width = width
        self.height = height
        self.channels = channels
        self.image = image
    
    def generate_position(self):
        return random.randint(self.width // 4, 3 * self.width // 4), random.randint(self.height // 4, 3 * self.height // 4)
    
    def random_color(self, grayscale=False):
        if grayscale:
            gray_value = random.randint(150, 255)
            return (gray_value, gray_value, gray_value)
        return tuple(random.randint(0, 255) for _ in range(3))
    
    def draw_circle(self, radius_range):
        center = self.generate_position()
        radius = random.randint(*radius_range)
        color = self.random_color(grayscale=True)
        cv2.circle(self.image, center, radius, color, -1, lineType=cv2.LINE_AA)
    
    def draw_triangle(self, side_length_1, side_length_2):
        x1, y1 = self.generate_position()
        side_1 = random.randint(*side_length_1)
        side_2 = random.randint(*side_length_2)
        x2, y2 = x1 + side_1, y1
        x3, y3 = x1 + side_2 // 2, y1 - side_2
        pts = np.array([[x1, y1], [x2, y2], [x3, y3]], np.int32).reshape((-1, 1, 2))
        color = self.random_color(grayscale=True)
        cv2.fillPoly(self.image, [pts], color, lineType=cv2.LINE_AA)
    
    def draw_rectangle(self, length_range, width_range):
        x1, y1 = self.generate_position()
        width = random.randint(*width_range)
        length = random.randint(*length_range)
        x2, y2 = x1 + width, y1 + length
        color = self.random_color(grayscale=True)
        cv2.rectangle(self.image, (x1, y1), (x2, y2), color, -1, lineType=cv2.LINE_AA)

class Generator(Shape):
    def __init__(self, shape_type, quantity, folder_name, width=64, height=64, channels=3):
        self.image = np.zeros((height, width, channels), dtype=np.uint8)
        super().__init__(width, height, channels, self.image)
        self.shape_type = shape_type
        self.quantity = quantity
        self.folder_name = folder_name
        os.makedirs(self.folder_name, exist_ok=True)

    def save_image(self, filename):
        cv2.imwrite(filename, self.image)

    def generate(self):
        for i in range(1, self.quantity + 1):
            self.image.fill(0)
            if self.shape_type == "circle":
                self.draw_circle((10, 25))
            elif self.shape_type == "triangle":
                self.draw_triangle((10, 25), (10, 25))
            elif self.shape_type == "rectangle":
                self.draw_rectangle((15, 30), (15, 30))
            else:
                raise ValueError("Unknown shape type!")

            filename = os.path.join(self.folder_name, f"{self.shape_type}_{i}.png")
            self.save_image(filename)

        print(f'{self.quantity} images saved in "{self.folder_name}" folder.')

def main():
    generator = Generator("triangle", 2, "Test_Shape")  # shape_type, quantity, folder_name
    generator.generate()

if __name__ == "__main__":
    main()
