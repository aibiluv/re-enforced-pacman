import pygame
class InputHandler:
    def __init__(self, ai = False):
        self.simulated_keys = {
            pygame.K_LEFT: False,
            pygame.K_RIGHT: False,
            pygame.K_UP: False,
            pygame.K_DOWN: False
        }
        self.ai = ai

    def update_simulated_input(self, key):
        """ Update the state of simulated inputs """
        for k in self.simulated_keys:
            self.simulated_keys[k] = (k == key)

    def get_key(self, key):
        if key in self.simulated_keys:
            if pygame.key.get_pressed()[key]:
                self.update_simulated_input(key)

        return self.simulated_keys[key]

    def get_pressed(self):
        if not self.ai:
            return pygame.key.get_pressed()
        return self.simulated_keys