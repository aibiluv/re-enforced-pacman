import random
import numpy as np
import tensorflow as tf
import tensorflow.keras as tf_keras
import tf_keras.models as models
import pygame
from pynput.keyboard import Key
import gc

# Ensure TensorFlow eager execution
tf.config.run_functions_eagerly(True)

# Configure TensorFlow to limit thread usage
tf.config.threading.set_intra_op_parallelism_threads(1)
tf.config.threading.set_inter_op_parallelism_threads(1)

# Key mapping
key_mapping = {
    0: (pygame.K_UP, "up", Key.up),
    1: (pygame.K_DOWN, "down", Key.down),
    2: (pygame.K_LEFT, "left", Key.left),
    3: (pygame.K_RIGHT, "right", Key.right),
    4: None  # Representing DO_NOTHING
}

def create_model(model_name='pacman', create_new=True, input_size=None):
    if create_new:
        # Clear previous Keras session to free up memory
        from tensorflow.keras import backend as K
        K.clear_session()

        model = models.Sequential([
            # Add the first convolutional layer
            tf_keras.layers.Conv2D(32, kernel_size=(3, 3), activation='relu', input_shape=input_size),  # Adjust input_shape to match your state dimensions
            tf_keras.layers.MaxPooling2D(pool_size=(2, 2)),
            # Add another convolutional layer
            tf_keras.layers.Conv2D(64, (3, 3), activation='relu'),
            tf_keras.layers.MaxPooling2D(pool_size=(2, 2)),
            tf_keras.layers.Conv2D(128, (3, 3), activation='relu'),  # Adding an extra conv layer
            tf_keras.layers.MaxPooling2D(pool_size=(2, 2)),
            # Flatten the output of the conv layers to feed into the dense layer
            tf_keras.layers.Flatten(),
            tf_keras.layers.Dense(256, activation='relu'),
            # Dense layer for further processing
            tf_keras.layers.Dense(128, activation='relu'),
            tf_keras.layers.Dense(64, activation='relu'),
            # Output layer: one output for each possible action
            tf_keras.layers.Dense(5)
        ],
        name=model_name)

        # Compile the model with an optimizer and loss function for training
        model.compile(optimizer='adam', loss=tf.keras.losses.MeanSquaredError())
        return model
    else:
        # Load an existing model
        return models.load_model(model_name)

def get_one_hot_encoding(action):
    one_hot_encoded = [
        int(action[pygame.K_UP]),    # Index 0 for UP
        int(action[pygame.K_DOWN]),  # Index 1 for DOWN
        int(action[pygame.K_LEFT]),  # Index 2 for LEFT
        int(action[pygame.K_RIGHT])  # Index 3 for RIGHT
    ]
    # Add DO_NOTHING state, which is True if no directional keys are pressed
    do_nothing = 1 if not any(one_hot_encoded) else 0
    one_hot_encoded.append(do_nothing)
    return one_hot_encoded

def train_conv_step_batch(model, states, next_states, actions, rewards, dones):
    print(f'Starting batch training of length {len(states)}')
    one_hot_actions = np.array([get_one_hot_encoding(action) for action in actions])

    # Ensure the states are properly shaped as needed by the CNN
    states_array = np.array(states)  # Assuming states are already correctly shaped
    next_states_array = np.array(next_states)

    current_q_values = model.predict(states_array)
    next_q_values = model.predict(next_states_array)
    
    max_next_qs = np.max(next_q_values, axis=1)
    not_dones = 1 - np.array(dones).flatten()
    target_q_values = np.array(rewards) + 0.95 * max_next_qs * not_dones
    targets = current_q_values.copy()
    
    for i, action in enumerate(one_hot_actions):
        action_index = np.argmax(action)
        if action_index >= targets.shape[1]:  # Safety check
            print(f"Index out of bounds: {action_index} for targets shape {targets.shape[1]}")
            continue
        targets[i, action_index] = target_q_values[i]

    with tf.GradientTape() as tape:
        # Forward pass
        predicted_q_values = model(states_array, training=True)
        # Calculate loss
        mse = tf.keras.losses.MeanSquaredError()
        loss = mse(targets, predicted_q_values)

    # Compute gradients and apply them to update the model
    grads = tape.gradient(loss, model.trainable_variables)
    model.optimizer.apply_gradients(zip(grads, model.trainable_variables))
    
    # Manually invoke garbage collection
    gc.collect()
    
    return loss.numpy()
