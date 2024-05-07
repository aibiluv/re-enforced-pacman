import random
class ReplayBuffer:
    def __init__(self, capacity):
        self.capacity = capacity
        self.buffer = []
        self.position = 0

    def push(self, states, actions, rewards, next_states, dones):
        # Ensure there is room in the buffer
        if len(self.buffer) < self.capacity:
            self.buffer.append(None)
        # Store the entire batch as a single entry in the buffer
        self.buffer[self.position] = (states, actions, rewards, next_states, dones)
        self.position = (self.position + 1) % self.capacity

    def sample(self):
        return random.choice(self.buffer)

    def __len__(self):
        return len(self.buffer)
