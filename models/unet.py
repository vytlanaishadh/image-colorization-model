import tensorflow as tf
from tensorflow.keras import layers, Model

class UNet:
    def __init__(self, input_size=(256, 256, 1)):
        self.input_size = input_size

    def build(self):
        inputs = layers.Input(self.input_size)

        # Encoder
        c1 = self.conv_block(inputs, 64)
        p1 = layers.MaxPooling2D((2, 2))(c1)

        c2 = self.conv_block(p1, 128)
        p2 = layers.MaxPooling2D((2, 2))(c2)

        c3 = self.conv_block(p2, 256)
        p3 = layers.MaxPooling2D((2, 2))(c3)

        c4 = self.conv_block(p3, 512)
        p4 = layers.MaxPooling2D((2, 2))(c4)

        # Bottleneck
        bottleneck = self.conv_block(p4, 1024)

        # Decoder
        u5 = layers.Conv2DTranspose(512, (2, 2), strides=(2, 2), padding='same')(bottleneck)
        u5 = layers.concatenate([u5, c4])
        c5 = self.conv_block(u5, 512)

        u6 = layers.Conv2DTranspose(256, (2, 2), strides=(2, 2), padding='same')(c5)
        u6 = layers.concatenate([u6, c3])
        c6 = self.conv_block(u6, 256)

        u7 = layers.Conv2DTranspose(128, (2, 2), strides=(2, 2), padding='same')(c6)
        u7 = layers.concatenate([u7, c2])
        c7 = self.conv_block(u7, 128)

        u8 = layers.Conv2DTranspose(64, (2, 2), strides=(2, 2), padding='same')(c7)
        u8 = layers.concatenate([u8, c1])
        c8 = self.conv_block(u8, 64)

        outputs = layers.Conv2D(3, (1, 1), activation='sigmoid')(c8)

        return Model(inputs=inputs, outputs=outputs)

    def conv_block(self, input_tensor, filters):
        x = layers.Conv2D(filters, (3, 3), padding='same')(input_tensor)
        x = layers.BatchNormalization()(x)
        x = layers.Activation('relu')(x)
        x = layers.Conv2D(filters, (3, 3), padding='same')(x)
        x = layers.BatchNormalization()(x)
        x = layers.Activation('relu')(x)
        return x

# To use the U-Net model:
# model = UNet().build()
# model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
