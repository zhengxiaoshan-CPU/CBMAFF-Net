# model.py

import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.layers import (
    Input,
    Conv1D,
    Conv2D,
    MaxPooling1D,
    MaxPooling2D,
    Flatten,
    Dense,
    Dropout,
    BatchNormalization,
    Bidirectional,
    LSTM,
    Concatenate,
)
from tensorflow.keras.models import Model


def create_model(shift_c_shape, type_c_shape, shift_h_shape, type_h_shape, num_classes=6):
    """
    Builds and compiles the TensorFlow/Keras model.

    Args:
        shift_c_shape (tuple): Shape of the shift_c input data (timesteps, channels).
        type_c_shape (tuple): Shape of the type_c input data (height, width, channels).
        shift_h_shape (tuple): Shape of the shift_h input data (timesteps, channels).
        type_h_shape (tuple): Shape of the type_h input data (height, width, channels).
        num_classes (int): Number of output classes.

    Returns:
        tensorflow.keras.models.Model: Compiled Keras model ready for training.
    """

    # CNN extraction for C shift features
    input_shift_c = Input(shape=shift_c_shape, name='shift_c_input')
    x1 = Conv1D(64, 3, activation='relu')(input_shift_c)
    x1 = BatchNormalization()(x1)
    x1 = MaxPooling1D(2)(x1)
    x1 = Dropout(0.5)(x1)
    x1 = Bidirectional(LSTM(64, return_sequences=True))(x1)
    x1 = Dropout(0.5)(x1)
    x1 = Flatten()(x1)

    # CNN extraction for C type features
    input_type_c = Input(shape=type_c_shape, name='type_c_input')
    x2 = Conv2D(64, (3, 3), activation='relu')(input_type_c)
    x2 = BatchNormalization()(x2)
    x2 = MaxPooling2D((2, 2))(x2)
    x2 = Dropout(0.5)(x2)
    x2 = Flatten()(x2)

    # CNN extraction for H shift features
    input_shift_h = Input(shape=shift_h_shape, name='shift_h_input')
    x3 = Conv1D(64, 3, activation='relu')(input_shift_h)
    x3 = BatchNormalization()(x3)
    x3 = MaxPooling1D(2)(x3)
    x3 = Dropout(0.5)(x3)
    x3 = Bidirectional(LSTM(64, return_sequences=True))(x3)
    x3 = Dropout(0.5)(x3)
    x3 = Flatten()(x3)

    # CNN extraction for H type features
    input_type_h = Input(shape=type_h_shape, name='type_h_input')
    x4 = Conv2D(64, (3, 3), activation='relu')(input_type_h)
    x4 = BatchNormalization()(x4)
    x4 = MaxPooling2D((2, 2))(x4)
    x4 = Dropout(0.5)(x4)
    x4 = Flatten()(x4)

    # Concatenate features
    concatenated_c = Concatenate(name='concatenated_c')([x1, x2])
    concatenated_h = Concatenate(name='concatenated_h')([x3, x4])

    # Attention mechanism for C features
    attention_c = layers.Attention()([concatenated_c, concatenated_c])

    # Attention mechanism for H features
    attention_h = layers.Attention()([concatenated_h, concatenated_h])

    # Final concatenation
    final_features = Concatenate(name='final_concatenation')(
        [concatenated_c, concatenated_h, attention_c, attention_h]
    )

    # Fully connected layers
    dense = Dense(128, activation='relu')(final_features)
    output = Dense(num_classes, activation='softmax', name='output')(dense)

    # Build the model
    model = Model(
        inputs=[input_shift_c, input_type_c, input_shift_h, input_type_h],
        outputs=output,
        name='MultiInputModel',
    )

    # Compile the model
    model.compile(
        optimizer='adam',
        loss='categorical_crossentropy',
        metrics=['accuracy'],
    )

    return model
