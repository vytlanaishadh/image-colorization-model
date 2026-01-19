import cv2
import numpy as np
from tensorflow.keras.preprocessing.image import ImageDataGenerator

class DataPreprocessing:
    def __init__(self, target_size=(256, 256), augmentations=None):
        self.target_size = target_size
        self.augmentations = augmentations if augmentations else ImageDataGenerator()

    def lab_color_space_conversion(self, image):
        # Convert the image to LAB color space
        return cv2.cvtColor(image, cv2.COLOR_BGR2LAB)

    def resize_image(self, image):
        # Resize the image to the target size
        return cv2.resize(image, self.target_size)

    def preprocess_image(self, image):
        # Convert to LAB and resize the image
        image = self.lab_color_space_conversion(image)
        return self.resize_image(image)

    def augment_data(self, images):
        # Apply augmentations to the dataset
        return self.augmentations.flow(images)
