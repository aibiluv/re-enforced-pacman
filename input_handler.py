import pygame
class InputHandler:
    def __init__(self):
        self.simulated_keys = {
            "left": False,
            "right": False,
            "up": False,
            "down": False
        }

    def update_simulated_input(self, direction, state):
        """ Update the state of simulated inputs """
        if direction in self.simulated_keys:
            self.simulated_keys[direction] = state

    def get_key(self, key):
        # Map Pygame keys to directions
        key_map = {
            pygame.K_LEFT: "left",
            pygame.K_RIGHT: "right",
            pygame.K_UP: "up",
            pygame.K_DOWN: "down"
        }
        if key in key_map:
            # Return the real key press or the simulated state
            return pygame.key.get_pressed()[key] or self.simulated_keys[key_map[key]]
        return False
