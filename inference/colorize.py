import cv2
import numpy as np

class ImageColorizer:
    def __init__(self, model):
        self.model = model

    def preprocess(self, image):
        # Preprocess the input image: resize, normalize, etc.
        image = cv2.resize(image, (224, 224))  # Example size
        image = image.astype('float32') / 255.0  # Normalize to [0, 1]
        return image

    def predict(self, grayscale_image):
        # Process the grayscale image and predict the colorized version
        processed_image = self.preprocess(grayscale_image)
        predicted_colors = self.model.predict(np.expand_dims(processed_image, axis=0))  # Add batch dimension
        return predicted_colors[0]

    def colorize(self, grayscale_image):
        # Make colorization call
        colorized_image = self.predict(grayscale_image)
        return colorized_image

if __name__ == '__main__':
    model = load_model('path_to_model')  # Load your trained model
    colorizer = ImageColorizer(model)

    # Example usage with a grayscale image
    gray_image = cv2.imread('path_to_grayscale_image', cv2.IMREAD_GRAYSCALE)
    colorized_image = colorizer.colorize(gray_image)
    cv2.imwrite('colorized_output_image.jpg', colorized_image * 255)  # Save the colorized output