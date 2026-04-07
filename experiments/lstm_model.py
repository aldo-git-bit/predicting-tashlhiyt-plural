#!/usr/bin/env python3
"""
Bi-LSTM architecture for Tashlhiyt plural prediction.

Implements a character-level bidirectional LSTM classifier.
"""

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from lstm_utils import MacroF1


def build_bilstm(
    vocab_size,
    max_len,
    num_classes,
    embedding_dim=32,
    lstm_units=64,
    dropout=0.3,
    recurrent_dropout=0.0,
    l2_reg=0.0
):
    """
    Build a Bidirectional LSTM model for sequence classification.

    Architecture:
        Embedding → Bi-LSTM → Dropout → Dense (Softmax)

    Args:
        vocab_size: Size of character vocabulary
        max_len: Maximum sequence length
        num_classes: Number of output classes
        embedding_dim: Dimension of character embeddings
        lstm_units: Number of LSTM hidden units (per direction)
        dropout: Dropout rate after LSTM
        recurrent_dropout: Dropout rate for LSTM recurrent connections
        l2_reg: L2 regularization coefficient

    Returns:
        keras.Model: Compiled model
    """
    # Input layer
    inputs = keras.Input(shape=(max_len,), name='char_input')

    # Embedding layer
    x = layers.Embedding(
        input_dim=vocab_size,
        output_dim=embedding_dim,
        input_length=max_len,
        name='embedding'
    )(inputs)

    # Bidirectional LSTM
    x = layers.Bidirectional(
        layers.LSTM(
            lstm_units,
            return_sequences=False,  # Many-to-one
            recurrent_dropout=recurrent_dropout,
            kernel_regularizer=keras.regularizers.l2(l2_reg) if l2_reg > 0 else None,
            name='lstm'
        ),
        name='bidirectional_lstm'
    )(x)

    # Dropout for regularization
    if dropout > 0:
        x = layers.Dropout(dropout, name='dropout')(x)

    # Output layer
    outputs = layers.Dense(
        num_classes,
        activation='softmax',
        name='output'
    )(x)

    # Build model
    model = keras.Model(inputs=inputs, outputs=outputs, name='BiLSTM')

    # Compile with metrics
    model.compile(
        optimizer='adam',
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy', MacroF1(num_classes)]
    )

    return model


def build_combined_model(
    vocab_size,
    max_len,
    num_features,
    num_classes,
    embedding_dim=32,
    lstm_units=64,
    dropout=0.3,
    recurrent_dropout=0.0,
    l2_reg=0.0
):
    """
    Build a Combined model: Bi-LSTM + Hand-crafted features.

    Architecture:
        [Char input → Embedding → Bi-LSTM] ⊕ [Feature input] → Dense → Softmax

    Args:
        vocab_size: Size of character vocabulary
        max_len: Maximum sequence length
        num_features: Number of hand-crafted features
        num_classes: Number of output classes
        embedding_dim: Dimension of character embeddings
        lstm_units: Number of LSTM hidden units (per direction)
        dropout: Dropout rate
        recurrent_dropout: Dropout rate for LSTM recurrent connections
        l2_reg: L2 regularization coefficient

    Returns:
        keras.Model: Compiled model with two inputs
    """
    # Input 1: Character sequences
    char_input = keras.Input(shape=(max_len,), name='char_input')

    # LSTM branch
    x1 = layers.Embedding(
        input_dim=vocab_size,
        output_dim=embedding_dim,
        input_length=max_len,
        name='embedding'
    )(char_input)

    x1 = layers.Bidirectional(
        layers.LSTM(
            lstm_units,
            return_sequences=False,
            recurrent_dropout=recurrent_dropout,
            kernel_regularizer=keras.regularizers.l2(l2_reg) if l2_reg > 0 else None,
            name='lstm'
        ),
        name='bidirectional_lstm'
    )(x1)

    if dropout > 0:
        x1 = layers.Dropout(dropout, name='lstm_dropout')(x1)

    # Input 2: Hand-crafted features
    feat_input = keras.Input(shape=(num_features,), name='feat_input')

    # Concatenate LSTM output and features
    combined = layers.Concatenate(name='concatenate')([x1, feat_input])

    # Optional dense layer before output
    # x = layers.Dense(64, activation='relu', name='dense_hidden')(combined)
    # if dropout > 0:
    #     x = layers.Dropout(dropout, name='final_dropout')(x)

    # Output layer
    outputs = layers.Dense(
        num_classes,
        activation='softmax',
        name='output'
    )(combined)

    # Build model
    model = keras.Model(
        inputs=[char_input, feat_input],
        outputs=outputs,
        name='Combined_BiLSTM'
    )

    # Compile (without custom metric to avoid TF 2.20 bug)
    model.compile(
        optimizer='adam',
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )

    return model


def get_model_summary(model):
    """
    Get formatted model summary.

    Args:
        model: Keras model

    Returns:
        str: Model summary string
    """
    import io
    stream = io.StringIO()
    model.summary(print_fn=lambda x: stream.write(x + '\n'))
    return stream.getvalue()


if __name__ == '__main__':
    # Test model building
    print("Testing Bi-LSTM model...")

    # Example parameters
    vocab_size = 50
    max_len = 20
    num_classes = 2

    # Build baseline model
    model = build_bilstm(vocab_size, max_len, num_classes)
    print("\n=== Baseline Bi-LSTM ===")
    print(get_model_summary(model))

    # Build combined model
    num_features = 169  # Example: Morph+Phon features for medial_a
    combined_model = build_combined_model(vocab_size, max_len, num_features, num_classes)
    print("\n=== Combined Bi-LSTM + Features ===")
    print(get_model_summary(combined_model))

    print("\n✓ Models built successfully!")
