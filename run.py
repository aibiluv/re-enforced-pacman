import os
import random
import threading
import time
import numpy as np
from input_handler import InputHandler
from ml.replay_buffer import ReplayBuffer
from ml.train import create_model, train_conv_step_batch, key_mapping
from ml.training_executor import TrainingExecutor
import pygame
from pygame.locals import *
from constants import *
from pacman import Pacman
from nodes import NodeGroup
from pellets import PelletGroup
from ghosts import GhostGroup
from fruit import Fruit
from pauser import Pause
from text import TextGroup
from sprites import LifeSprites
from sprites import MazeSprites
from mazedata import MazeData
from vector import Vector2
import tf_keras.models as models
from pynput.keyboard import Controller
keyboard = Controller()
# Create a lock object


games_played = 0
class GameController(object):
    def __init__(self):
        pygame.init()
        self.screen = pygame.display.set_mode(SCREENSIZE, 0, 32)
        self.background = None
        self.background_norm = None
        self.background_flash = None
        self.clock = pygame.time.Clock()
        self.fruit = None
        self.pause = Pause(True)
        self.level = 0
        self.lives = 1
        self.score = 0
        self.reward = 0
        self.textgroup = TextGroup()
        self.lifesprites = LifeSprites(self.lives)
        self.flashBG = False
        self.flashTime = 0.2
        self.flashTimer = 0
        self.fruitCaptured = []
        self.fruitNode = None
        self.mazedata = MazeData()
        self.batch_states = []
        self.batch_next_states = []
        self.batch_actions = []
        self.batch_rewards = []
        self.batch_dones = []
        self.batch_size = 1000 
        self.replay_buffer = ReplayBuffer(1000)


        

    def setup_ml_params(self):
        self.matrixes_buffer = []  # Buffer to accumulate matrices
        self.buffer_size = 1000  # Number of matrices to accumulate before writing to file
        self.save_directory = "channel_files"
        self.ensure_directory_exists()
        self.last_save_time = time.time()  # Initialize the last save time
        self.save_interval = 5  # seconds between saves
        self.train = True
        
        self.defaultGrid = np.zeros((NROWS, NCOLS))  # Initialize matrix with zeros       
        self.model = create_model(model_name='pac_man_human_trainer', create_new=False, input_size=(NROWS-5, NCOLS,7))
        self.trainingExecutor= TrainingExecutor(self.model, 1)
        self.defaultGrid = np.zeros((NROWS, NCOLS))  # Initialize matrix with zeros
        self.generate_matrix_with_position(self.defaultGrid, self.mazesprites.wall_vector ,-1)

    def setBackground(self):
        self.background_norm = pygame.surface.Surface(SCREENSIZE).convert()
        self.background_norm.fill(BLACK)
        self.background_flash = pygame.surface.Surface(SCREENSIZE).convert()
        self.background_flash.fill(BLACK)
        self.background_norm = self.mazesprites.constructBackground(self.background_norm, self.level%5)
        self.background_flash = self.mazesprites.constructBackground(self.background_flash, 5)
        self.flashBG = False
        self.background = self.background_norm

    def startGame(self):      
        self.mazedata.loadMaze(self.level)
        self.mazesprites = MazeSprites(self.mazedata.obj.name+".txt", self.mazedata.obj.name+"_rotation.txt")
        self.setBackground()
        self.nodes = NodeGroup(self.mazedata.obj.name+".txt")
        self.mazedata.obj.setPortalPairs(self.nodes)
        self.mazedata.obj.connectHomeNodes(self.nodes)
        self.ai = True
        self.inputHandler = InputHandler(self.ai)
        self.pacman = Pacman(self.nodes.getNodeFromTiles(*self.mazedata.obj.pacmanStart), self.inputHandler)
        self.pellets = PelletGroup(self.mazedata.obj.name+".txt")
        self.ghosts = GhostGroup(self.nodes.getStartTempNode(), self.pacman)

        self.ghosts.pinky.setStartNode(self.nodes.getNodeFromTiles(*self.mazedata.obj.addOffset(2, 3)))
        self.ghosts.inky.setStartNode(self.nodes.getNodeFromTiles(*self.mazedata.obj.addOffset(0, 3)))
        self.ghosts.clyde.setStartNode(self.nodes.getNodeFromTiles(*self.mazedata.obj.addOffset(4, 3)))
        self.ghosts.setSpawnNode(self.nodes.getNodeFromTiles(*self.mazedata.obj.addOffset(2, 3)))
        self.ghosts.blinky.setStartNode(self.nodes.getNodeFromTiles(*self.mazedata.obj.addOffset(2, 0)))

        self.nodes.denyHomeAccess(self.pacman)
        self.nodes.denyHomeAccessList(self.ghosts)
        self.ghosts.inky.startNode.denyAccess(RIGHT, self.ghosts.inky)
        self.ghosts.clyde.startNode.denyAccess(LEFT, self.ghosts.clyde)
        self.mazedata.obj.denyGhostsAccess(self.ghosts, self.nodes)
        
        self.setup_ml_params()

    def startGame_old(self):      
        self.mazedata.loadMaze(self.level)#######
        print("start_game_old")
        self.mazesprites = MazeSprites("maze1.txt", "maze1_rotation.txt")
        self.setBackground()
        self.nodes = NodeGroup("maze1.txt")
        self.nodes.setPortalPair((0,17), (27,17))
        homekey = self.nodes.createHomeNodes(11.5, 14)
        self.nodes.connectHomeNodes(homekey, (12,14), LEFT)
        self.nodes.connectHomeNodes(homekey, (15,14), RIGHT)
        self.pacman = Pacman(self.nodes.getNodeFromTiles(15, 26))
        self.pellets = PelletGroup("maze1.txt")
        self.ghosts = GhostGroup(self.nodes.getStartTempNode(), self.pacman)
        self.ghosts.blinky.setStartNode(self.nodes.getNodeFromTiles(2+11.5, 0+14))
        self.ghosts.pinky.setStartNode(self.nodes.getNodeFromTiles(2+11.5, 3+14))
        self.ghosts.inky.setStartNode(self.nodes.getNodeFromTiles(0+11.5, 3+14))
        self.ghosts.clyde.setStartNode(self.nodes.getNodeFromTiles(4+11.5, 3+14))
        self.ghosts.setSpawnNode(self.nodes.getNodeFromTiles(2+11.5, 3+14))

        self.nodes.denyHomeAccess(self.pacman)
        self.nodes.denyHomeAccessList(self.ghosts)
        self.nodes.denyAccessList(2+11.5, 3+14, LEFT, self.ghosts)
        self.nodes.denyAccessList(2+11.5, 3+14, RIGHT, self.ghosts)
        self.ghosts.inky.startNode.denyAccess(RIGHT, self.ghosts.inky)
        self.ghosts.clyde.startNode.denyAccess(LEFT, self.ghosts.clyde)
        self.nodes.denyAccessList(12, 14, UP, self.ghosts)
        self.nodes.denyAccessList(15, 14, UP, self.ghosts)
        self.nodes.denyAccessList(12, 26, UP, self.ghosts)
        self.nodes.denyAccessList(15, 26, UP, self.ghosts)
 
        
    def generate_matrix_with_position(self,matrix, vectors, value):
        
        for v in vectors:
            grid_x = int(v.x / TILEWIDTH)  # Convert x-coordinate to grid index
            grid_y = int(v.y / TILEHEIGHT)  # Convert y-coordinate to grid index

            
            matrix[grid_y, grid_x] = value  # Set value at specified position to 1

    def get_state_for_conv(self):
        # Initialize channels for each game element
        wall_channel = (self.defaultGrid == -1).astype(int)
        space_channel = (self.defaultGrid == 0).astype(int)
        pellet_channel = np.zeros_like(self.defaultGrid)
        big_pellet_channel = np.zeros_like(self.defaultGrid)
        ghost_channel = np.zeros_like(self.defaultGrid)
        frightened_ghost_channel = np.zeros_like(self.defaultGrid)
        pacman_channel = np.zeros_like(self.defaultGrid)  # Pac-Man channel
        # Set Pac-Man's position in his channel
        pacman_x, pacman_y = int(self.pacman.position.x / TILEWIDTH), int(self.pacman.position.y / TILEHEIGHT)
        pacman_channel[pacman_y, pacman_x] = 1
        # Update channels based on game elements
        for pellet in self.pellets.pelletList:
            x, y = int(pellet.position.x / TILEWIDTH), int(pellet.position.y / TILEHEIGHT)
            if pellet.name == POWERPELLET:
                big_pellet_channel[y, x] = 1
            else:
                pellet_channel[y, x] = 1

        for ghost in self.ghosts:
            x, y = int(ghost.position.x / TILEWIDTH), int(ghost.position.y / TILEHEIGHT)
            if ghost.mode.current == FREIGHT:
                frightened_ghost_channel[y, x] = 1
            else:
                ghost_channel[y, x] = 1

        # Combine all channels into a single multi-channel array
        matrix = np.stack([wall_channel[3:-2], space_channel[3:-2], pellet_channel[3:-2], big_pellet_channel[3:-2], ghost_channel[3:-2], frightened_ghost_channel[3:-2], pacman_channel[3:-2]], axis=-1)
        return matrix        
    
    
    
    def ensure_directory_exists(self):
        """ Ensure the directory for saving channel files exists """
        if not os.path.exists(self.save_directory):
            os.makedirs(self.save_directory)

    def save_channels(self, matrix):
        channels = ['wall', 'space', 'pellet', 'big_pellet', 'ghost', 'frightened_ghost', 'pacman']
        for i, channel_name in enumerate(channels):
            filename = os.path.join(self.save_directory, f"{channel_name}.txt")
            # Open the file in append mode, or create it if it does not exist
            with open(filename, 'ab') as f:
                np.savetxt(f, matrix[:, :, i], fmt="%d")
                f.write(b"\n")

 
    def release_all_keys(self):
        """Release all keys in the key_mapping that are not None."""
        for key_info in key_mapping.values():
            if key_info is not None:
                keyboard.release(key_info[2]) 

    def ai_action(self, state, eps=0.1):

        if np.random.rand() < eps:
            keys = list(key_mapping.keys())
            random_key = random.choice(keys)
            action_key = key_mapping[random_key]
        else:
            action_space = self.model.predict(np.array(state).reshape(1, *state.shape))
            #print (f"action space: {action_space}")
            active_index = np.argmax(action_space)
            action_key = key_mapping[active_index]
          
        
        self.release_all_keys()
        if action_key is not None: 
            
            self.inputHandler.update_simulated_input(action_key[0])
            
        else:
            # DO_NOTHING or handle it accordingly
            pass
            #print("No action required")
    
    def update_training_params(self, current_state, next_state, done):
        action = self.inputHandler.get_pressed()  # Assuming this captures the action correctly
        


        self.batch_states.append(current_state)
        self.batch_next_states.append(next_state)
        self.batch_actions.append(action)
        self.batch_rewards.append(self.reward)
        
        self.batch_dones.append(done)
    def set_reward(self, r):
        self.reward = r

    def update(self):
        
        dt = self.clock.tick(20) / 1000.0
        

        current_state = self.get_state_for_conv()
        self.textgroup.update(dt)
        self.pellets.update(dt)
        if not self.pause.paused:
            self.ghosts.update(dt)      
            if self.fruit is not None:
                self.fruit.update(dt)
            self.checkPelletEvents()
            self.checkGhostEvents()
            self.checkFruitEvents()

        done = np.array([self.pellets.isEmpty() or not self.pacman.alive])   
        if self.pacman.alive:
            if not self.pause.paused:
                self.pacman.update(dt, self.set_reward) 
                next_state = self.get_state_for_conv()               
                if (self.train):
                    self.update_training_params(current_state, next_state, done)
                    self.reward = 0
                    if len(self.batch_states) >= self.batch_size:                
                        thread_batch = (
                            list(self.batch_states),
                            list(self.batch_next_states),
                            list(self.batch_actions),
                            list(self.batch_rewards),
                            list(self.batch_dones)
                        )
                        self.replay_buffer.push(list(self.batch_states),
                            list(self.batch_next_states),
                            list(self.batch_actions),
                            list(self.batch_rewards),
                            list(self.batch_dones))

                        # Clear the main batch    
                        self.batch_states = []
                        self.batch_next_states = []
                        self.batch_actions = []
                        self.batch_rewards = []
                        self.batch_dones = []

                            # Clear the main batch    
                        self.trainingExecutor.submit(thread_batch, replay_batch=False)
        
                        sampled_batch = self.replay_buffer.sample()
        
                        # Submit the replay training task to the executor
                        self.trainingExecutor.submit(sampled_batch, replay_batch=True)

                        sampled_batch_2 = self.replay_buffer.sample()
        
                        # Submit the replay training task to the executor
                        self.trainingExecutor.submit(sampled_batch_2, replay_batch=True)

                        sampled_batch_3 = self.replay_buffer.sample()
        
                        # Submit the replay training task to the executor
                        self.trainingExecutor.submit(sampled_batch_3, replay_batch=True)

                if self.ai:
                    self.ai_action(next_state)
                    

        else:
            self.pacman.update(dt)
            next_state = self.get_state_for_conv()
            if(self.train):
                self.update_training_params(current_state, next_state, done)
                self.reward = 0
                
            
        
        if self.flashBG:
            self.flashTimer += dt
            if self.flashTimer >= self.flashTime:
                self.flashTimer = 0
                if self.background == self.background_norm:
                    self.background = self.background_flash
                else:
                    self.background = self.background_norm
         
        afterPauseMethod = self.pause.update(dt)
        if afterPauseMethod is not None:
            afterPauseMethod()
        self.checkEvents()
        self.render()

    def checkEvents(self):
        for event in pygame.event.get():
            if event.type == QUIT:
                print("Qutting")
                if self.train:
                    #flushing batch
                    if(len(self.batch_states)>0):
                        thread_batch = (
                            list(self.batch_states),
                            list(self.batch_next_states),
                            list(self.batch_actions),
                            list(self.batch_rewards),
                            list(self.batch_dones)
                        )

                        self.trainingExecutor.submit(thread_batch, replay_batch=False)
                    self.trainingExecutor.shutdown()
                exit()
            elif event.type == KEYDOWN:
                if event.key == K_SPACE:
                    if self.pacman.alive:
                        self.pause.setPause(playerPaused=True)
                        if not self.pause.paused:
                            self.textgroup.hideText()
                            self.showEntities()
                        else:
                            self.textgroup.showText(PAUSETXT)
                            #self.hideEntities()

    def checkPelletEvents(self):
        pellet = self.pacman.eatPellets(self.pellets.pelletList)
        if pellet:
            self.pellets.numEaten += 1
            self.updateScore(pellet.points)
            self.reward = pellet.points
            if self.pellets.numEaten == 30:
                self.ghosts.inky.startNode.allowAccess(RIGHT, self.ghosts.inky)
            if self.pellets.numEaten == 70:
                self.ghosts.clyde.startNode.allowAccess(LEFT, self.ghosts.clyde)
            
            self.pellets.pelletList.remove(pellet)
            if pellet.name == POWERPELLET:
                self.ghosts.startFreight()
                
            if self.pellets.isEmpty():
                self.flashBG = True
                self.hideEntities()
                self.pause.setPause(pauseTime=3, func=self.nextLevel)
    def press_button_delayed(self, key, delay):
        # Wait for 2 seconds
        time.sleep(delay)
        # Press and release the spacebar
        pygame.event.post(pygame.event.Event(pygame.KEYDOWN, key = key))
        

    def saveScore(self):
        global games_played 
        filename = "game_scores.txt"
        
        # Check if the file exists and append if it does
        mode = 'a' if os.path.exists(filename) else 'w'
        
        with open(filename, mode) as file:
            file.write(f"Game {games_played + 1}: Score = {self.score}\n")
        
        # Increment the games_played counter
        games_played += 1
        print(f"Score saved. Total games played: {games_played}")
    def checkGhostEvents(self):
        for ghost in self.ghosts:
            if self.pacman.collideGhost(ghost):
                if ghost.mode.current is FREIGHT:
                    self.pacman.visible = False
                    ghost.visible = False
                    self.updateScore(ghost.points)   
                    self.reward = ghost.points              
                    self.textgroup.addText(str(ghost.points), WHITE, ghost.position.x, ghost.position.y, 8, time=1)
                    self.ghosts.updatePoints()
                    self.pause.setPause(pauseTime=1, func=self.showEntities)
                    ghost.startSpawn()
                    self.nodes.allowHomeAccess(ghost)
                elif ghost.mode.current is not SPAWN:
                    if self.pacman.alive:
                        self.lives -=  1
                        self.updateScore(-100)
                        self.reward = -100
                        self.lifesprites.removeImage()
                        self.pacman.die()               
                        self.ghosts.hide()
                        if self.lives <= 0:
                            self.textgroup.showText(GAMEOVERTXT)
                            self.pause.setPause(pauseTime=0, func=self.restartGame)
                            thread = threading.Thread(target=self.press_button_delayed, args=(pygame.K_SPACE, 2))
                            thread.start()        
                            self.saveScore()                    
                        else:
                            self.pause.setPause(pauseTime=0, func=self.resetLevel)
                            thread = threading.Thread(target=self.press_button_delayed, args=(pygame.K_SPACE, 2))
                            thread.start()

                            
    
    def checkFruitEvents(self):
        if self.pellets.numEaten == 50 or self.pellets.numEaten == 140:
            if self.fruit is None:
                self.fruit = Fruit(self.nodes.getNodeFromTiles(9, 20), self.level)
                print(self.fruit)
        if self.fruit is not None:
            if self.pacman.collideCheck(self.fruit):
                self.updateScore(self.fruit.points)
                self.textgroup.addText(str(self.fruit.points), WHITE, self.fruit.position.x, self.fruit.position.y, 8, time=1)
                fruitCaptured = False
                for fruit in self.fruitCaptured:
                    if fruit.get_offset() == self.fruit.image.get_offset():
                        fruitCaptured = True
                        break
                if not fruitCaptured:
                    self.fruitCaptured.append(self.fruit.image)
                self.fruit = None
            elif self.fruit.destroy:
                self.fruit = None

    def showEntities(self):
        self.pacman.visible = True
        self.ghosts.show()

    def hideEntities(self):
        self.pacman.visible = False
        self.ghosts.hide()

    def nextLevel(self):
        self.showEntities()
        self.level += 1
        self.pause.paused = True
        self.startGame()
        self.textgroup.updateLevel(self.level)

    def restartGame(self):
        self.lives = 1
        self.level = 0
        self.pause.paused = True
        self.fruit = None
        self.startGame()
        self.score = 0
        self.textgroup.updateScore(self.score)
        self.textgroup.updateLevel(self.level)
        self.textgroup.showText(READYTXT)
        self.lifesprites.resetLives(self.lives)
        self.fruitCaptured = []

    def resetLevel(self):
        self.pause.paused = True
        self.pacman.reset()
        self.ghosts.reset()
        self.fruit = None
        self.textgroup.showText(READYTXT)

    def updateScore(self, points):
        self.score += points
        self.textgroup.updateScore(self.score)

    def render(self):
        self.screen.blit(self.background, (0, 0))
        #self.nodes.render(self.screen)
        self.pellets.render(self.screen)
        if self.fruit is not None:
            self.fruit.render(self.screen)
        self.pacman.render(self.screen)
        self.ghosts.render(self.screen)
        self.textgroup.render(self.screen)

        for i in range(len(self.lifesprites.images)):
            x = self.lifesprites.images[i].get_width() * i
            y = SCREENHEIGHT - self.lifesprites.images[i].get_height()
            self.screen.blit(self.lifesprites.images[i], (x, y))

        for i in range(len(self.fruitCaptured)):
            x = SCREENWIDTH - self.fruitCaptured[i].get_width() * (i+1)
            y = SCREENHEIGHT - self.fruitCaptured[i].get_height()
            self.screen.blit(self.fruitCaptured[i], (x, y))

        pygame.display.update()


if __name__ == "__main__":
    game = GameController()
    game.startGame()
    while True:
        game.update()



