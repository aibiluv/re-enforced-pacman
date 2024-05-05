import random
import numpy as np
import tensorflow as tf
from tf_keras.models import Sequential
import tf_keras.models as models
import tf_keras
import pygame
from pynput.keyboard import Key
key_mapping = {
    0: (pygame.K_UP, "up", Key.up),
    1: (pygame.K_DOWN, "down", Key.down),
    2: (pygame.K_LEFT, "left", Key.left),
    3: (pygame.K_RIGHT, "right", Key.right),
    4: None  # Representing DO_NOTHING
}
def create_model(model_name = 'pacman', create_new = True):
    if create_new:
        model = models.Sequential([
            tf_keras.layers.Dense(64, activation='relu', input_shape=(874,)),
            tf_keras.layers.Dense(64, activation='relu'),
            tf_keras.layers.Dense(5)  # Output layer: one output for each possible action
        ],
        name = model_name)
        model.compile(optimizer='adam', loss=tf.keras.losses.MeanSquaredError())
        return model
    else:
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

def train_step(model, states, next_states, action,reward, done):
    # One-hot encode the four direction keys: [UP, DOWN, LEFT, RIGHT]
    
    one_hot_encoded = get_one_hot_encoding(action)
    print (f"reward: {reward}, done: {done}, action: {one_hot_encoded}")
    current_q_values = model(np.array(states).reshape(1, len(states)))
    next_q_values = model.predict(np.array(next_states).reshape(1, len(next_states)))
    max_next_q = np.max(next_q_values[0])
    target_q_value = reward
    if not done:
        target_q_value += 0.95 * max_next_q
    
    target_q_values = np.copy(current_q_values[0])
    target_q_values[one_hot_encoded.index(1)] = target_q_value
    
    with tf.GradientTape() as tape:
              # Re-predict the Q-values to attach them to the gradient tape
        predicted_q_values = model(np.array(states).reshape(1, len(states)), training=True)
        # Calculate loss
        mse = tf.keras.losses.MeanSquaredError()

        # Calculate the loss
        loss = mse(target_q_values, predicted_q_values[0])

    # Calculate gradients and update model weights
    grads = tape.gradient(loss, model.trainable_variables)
    model.optimizer.apply_gradients(zip(grads, model.trainable_variables))
    return loss.numpy()

def train_step_batch(model, states, next_states, actions, rewards, dones):
    one_hot_actions = np.array([get_one_hot_encoding(action) for action in actions])

    current_q_values = model.predict(np.array(states))
    next_q_values = model.predict(np.array(next_states))
    
    max_next_qs = np.max(next_q_values, axis=1)
    not_dones = 1 - np.array(dones).flatten()
    target_q_values = np.array(rewards )+ 0.95 * max_next_qs * not_dones
    targets = current_q_values.copy()
    
    for i, action in enumerate(one_hot_actions):
        
        action_index = np.argmax(action)
        if action_index >= targets.shape[1]:  # Safety check
            print(f"Index out of bounds: {action_index} for targets shape {targets.shape[1]}")
            continue
        targets[i, action_index] = target_q_values[i]

    with tf.GradientTape() as tape:
        predicted_q_values = model(np.array(states), training=True)
        mse = tf.keras.losses.MeanSquaredError()
        loss = mse(targets, predicted_q_values)

    grads = tape.gradient(loss, model.trainable_variables)
    model.optimizer.apply_gradients(zip(grads, model.trainable_variables))
    return loss.numpy()
