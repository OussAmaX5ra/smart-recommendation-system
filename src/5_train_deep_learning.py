"""
Step 5: Train Deep Learning Model
Neural Collaborative Filtering with user and item embeddings
Uses TensorFlow/Keras
"""

import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, Model
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from sklearn.model_selection import train_test_split
import config
from utils import print_section, load_pickle


def load_training_data():
    """Load training data with encoders"""
    print_section("Loading Training Data")

    train_df = pd.read_csv(config.TRAIN_DATA_PATH)
    val_df = pd.read_csv(config.VAL_DATA_PATH)

    user_encoder = load_pickle(config.USER_ENCODER_PATH)
    product_encoder = load_pickle(config.PRODUCT_ENCODER_PATH)

    print(f"✓ Train data: {train_df.shape}")
    print(f"✓ Validation data: {val_df.shape}")
    print(f"✓ Unique users: {len(user_encoder.classes_)}")
    print(f"✓ Unique products: {len(product_encoder.classes_)}")

    return train_df, val_df, user_encoder, product_encoder


def prepare_dl_data(train_df, val_df):
    """
    Prepare data for deep learning model
    Create positive and negative samples
    """
    print_section("Preparing DL Data")

    # Use reordered as target (1 if reordered, 0 otherwise)
    train_df['target'] = train_df['reordered'].fillna(0).astype(int)
    val_df['target'] = val_df['reordered'].fillna(0).astype(int)

    print(f"Train target distribution:")
    print(train_df['target'].value_counts())

    print(f"\nValidation target distribution:")
    print(val_df['target'].value_counts())

    return train_df, val_df


def create_ncf_model(n_users, n_items, embedding_dim=64, hidden_units=[128, 64], dropout_rate=0.3):
    """
    Create Neural Collaborative Filtering model
    Architecture: User & Item Embeddings -> Concatenate -> Dense Layers -> Output
    """
    print_section("Creating NCF Model")

    # Input layers
    user_input = layers.Input(shape=(1,), name='user_input')
    item_input = layers.Input(shape=(1,), name='item_input')

    # Embedding layers
    user_embedding = layers.Embedding(
        n_users,
        embedding_dim,
        name='user_embedding',
        embeddings_regularizer=keras.regularizers.l2(1e-6)
    )(user_input)
    user_embedding = layers.Flatten()(user_embedding)

    item_embedding = layers.Embedding(
        n_items,
        embedding_dim,
        name='item_embedding',
        embeddings_regularizer=keras.regularizers.l2(1e-6)
    )(item_input)
    item_embedding = layers.Flatten()(item_embedding)

    # Concatenate embeddings
    concat = layers.Concatenate()([user_embedding, item_embedding])

    # Dense layers
    x = concat
    for units in hidden_units:
        x = layers.Dense(units, activation='relu')(x)
        x = layers.BatchNormalization()(x)
        x = layers.Dropout(dropout_rate)(x)

    # Output layer
    output = layers.Dense(1, activation='sigmoid', name='output')(x)

    # Create model
    model = Model(inputs=[user_input, item_input], outputs=output)

    print("\nModel Architecture:")
    model.summary()

    return model


def create_sequence_model(n_users, n_items, sequence_length=10):
    """
    Create sequence-based model using GRU for temporal patterns
    Alternative architecture for capturing user purchase sequences
    """
    print_section("Creating Sequence Model")

    # User input
    user_input = layers.Input(shape=(1,), name='user_input')
    user_embedding = layers.Embedding(n_users, config.DL_EMBEDDING_DIM)(user_input)
    user_embedding = layers.Flatten()(user_embedding)

    # Product sequence input
    sequence_input = layers.Input(shape=(sequence_length,), name='sequence_input')
    sequence_embedding = layers.Embedding(n_items, config.DL_EMBEDDING_DIM)(sequence_input)

    # GRU layer for sequence
    gru_out = layers.GRU(config.RNN_UNITS, return_sequences=False)(sequence_embedding)

    # Target product input
    target_input = layers.Input(shape=(1,), name='target_input')
    target_embedding = layers.Embedding(n_items, config.DL_EMBEDDING_DIM)(target_input)
    target_embedding = layers.Flatten()(target_embedding)

    # Concatenate all
    concat = layers.Concatenate()([user_embedding, gru_out, target_embedding])

    # Dense layers
    x = layers.Dense(128, activation='relu')(concat)
    x = layers.Dropout(0.3)(x)
    x = layers.Dense(64, activation='relu')(x)
    x = layers.Dropout(0.3)(x)

    # Output
    output = layers.Dense(1, activation='sigmoid')(x)

    model = Model(inputs=[user_input, sequence_input, target_input], outputs=output)

    print("\nSequence Model Architecture:")
    model.summary()

    return model


def compile_model(model, learning_rate=0.001):
    """Compile the model"""
    optimizer = keras.optimizers.Adam(learning_rate=learning_rate)

    model.compile(
        optimizer=optimizer,
        loss='binary_crossentropy',
        metrics=[
            'accuracy',
            keras.metrics.AUC(name='auc'),
            keras.metrics.Precision(name='precision'),
            keras.metrics.Recall(name='recall')
        ]
    )

    print(f"✓ Model compiled with Adam optimizer (lr={learning_rate})")

    return model


def create_callbacks():
    """Create training callbacks"""
    callbacks = [
        EarlyStopping(
            monitor='val_loss',
            patience=5,
            restore_best_weights=True,
            verbose=1
        ),
        ModelCheckpoint(
            config.DL_MODEL_PATH,
            monitor='val_auc',
            mode='max',
            save_best_only=True,
            verbose=1
        ),
        ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.5,
            patience=3,
            min_lr=1e-7,
            verbose=1
        )
    ]

    return callbacks


def train_model(model, train_df, val_df):
    """Train the model"""
    print_section("Training Model")

    # Prepare input data
    X_train = [
        train_df['user_encoded'].values,
        train_df['product_encoded'].values
    ]
    y_train = train_df['target'].values

    X_val = [
        val_df['user_encoded'].values,
        val_df['product_encoded'].values
    ]
    y_val = val_df['target'].values

    print(f"Training samples: {len(y_train)}")
    print(f"Validation samples: {len(y_val)}")

    # Create callbacks
    callbacks = create_callbacks()

    # Train model
    history = model.fit(
        X_train,
        y_train,
        batch_size=config.DL_BATCH_SIZE,
        epochs=config.DL_EPOCHS,
        validation_data=(X_val, y_val),
        callbacks=callbacks,
        verbose=1
    )

    return history


def evaluate_model(model, val_df):
    """Evaluate the trained model"""
    print_section("Evaluating Model")

    X_val = [
        val_df['user_encoded'].values,
        val_df['product_encoded'].values
    ]
    y_val = val_df['target'].values

    results = model.evaluate(X_val, y_val, verbose=1)

    print("\nEvaluation Results:")
    metrics = ['loss', 'accuracy', 'auc', 'precision', 'recall']
    for metric, value in zip(metrics, results):
        print(f"  {metric}: {value:.4f}")

    return results


def generate_predictions(model, train_df, n_users, n_items):
    """
    Generate predictions for all user-item pairs
    (for top products per user)
    """
    print_section("Generating Predictions for All User-Item Pairs")

    # Get unique users from training data
    unique_users = train_df['user_encoded'].unique()
    unique_products = train_df['product_encoded'].unique()

    print(f"Generating predictions for {len(unique_users)} users and {len(unique_products)} products...")

    # Create a dictionary to store predictions
    user_predictions = {}

    # Process in batches to avoid memory issues
    batch_size = 10000

    for user_idx, user_enc in enumerate(unique_users):
        if (user_idx + 1) % 100 == 0:
            print(f"  Processed {user_idx + 1}/{len(unique_users)} users")

        # Create input for all products for this user
        user_array = np.full(len(unique_products), user_enc)
        product_array = unique_products

        # Get predictions
        predictions = model.predict(
            [user_array, product_array],
            batch_size=batch_size,
            verbose=0
        ).flatten()

        # Sort products by prediction score
        sorted_indices = np.argsort(predictions)[::-1]
        sorted_products = unique_products[sorted_indices]
        sorted_scores = predictions[sorted_indices]

        # Store top predictions
        user_predictions[user_enc] = list(zip(sorted_products, sorted_scores))

    print(f"✓ Generated predictions for {len(user_predictions)} users")

    return user_predictions


def save_model_and_predictions(model, user_predictions):
    """Save model and predictions"""
    print_section("Saving Model and Predictions")

    # Model is already saved by ModelCheckpoint callback
    print(f"✓ Model saved to: {config.DL_MODEL_PATH}")

    # Save predictions
    import pickle
    predictions_path = config.DL_MODEL_PATH.replace('.h5', '_predictions.pkl')
    with open(predictions_path, 'wb') as f:
        pickle.dump(user_predictions, f)
    print(f"✓ Predictions saved to: {predictions_path}")


def main():
    """Main DL training pipeline"""
    print("\n" + "=" * 60)
    print("  STEP 5: TRAIN DEEP LEARNING MODEL")
    print("=" * 60)

    # Set random seeds for reproducibility
    np.random.seed(config.RANDOM_SEED)
    tf.random.set_seed(config.RANDOM_SEED)

    # Load data
    train_df, val_df, user_encoder, product_encoder = load_training_data()

    # Prepare data
    train_df, val_df = prepare_dl_data(train_df, val_df)

    # Get number of unique users and items
    n_users = len(user_encoder.classes_)
    n_items = len(product_encoder.classes_)

    # Create model
    model = create_ncf_model(
        n_users=n_users,
        n_items=n_items,
        embedding_dim=config.DL_EMBEDDING_DIM,
        hidden_units=config.DL_HIDDEN_UNITS,
        dropout_rate=config.DL_DROPOUT_RATE
    )

    # Compile model
    model = compile_model(model, learning_rate=config.DL_LEARNING_RATE)

    # Train model
    history = train_model(model, train_df, val_df)

    # Evaluate model
    results = evaluate_model(model, val_df)

    # Generate predictions
    user_predictions = generate_predictions(model, train_df, n_users, n_items)

    # Save model and predictions
    save_model_and_predictions(model, user_predictions)

    print("\n" + "=" * 60)
    print("  STEP 5 COMPLETED SUCCESSFULLY!")
    print("=" * 60 + "\n")


if __name__ == "__main__":
    main()