import numpy as np
import gym

class snake:
    def __init__(self, body, direction, representation):
        self.map = map
        self.direction = np.array(direction)
        self.body = [ np.array(b) for b in body]
        self.repr_head = representation[0]
        self.repr_tail = representation[1]
        self.alive = True

def printmap(map):
    char_map = {0: ' ', 1: '#', 2: 'S', 
                3: 's', 4: 'M', 5: 'm'}

    # Convert numbers to characters
    char_matrix = np.vectorize(char_map.get)(map)

    # Print in grid format
    for row in char_matrix:
        print(' '.join(row))


input2direction = {None:None, 0:np.array([0,1]), 1:np.array([-1,0]), 2:np.array([0,-1]), 3:np.array([1,0])}
class game(gym.Env):

    def __init__(self, boardsize=15, timelimit=50, verbose=False):
        super(game, self).__init__()
        self.action_space = gym.spaces.Discrete(4)  # Actions: [UP, DOWN, STAY]
        self.observation_space = gym.spaces.Box(low=0, high=1, shape=(boardsize+2,boardsize+2,6), dtype=np.uint8)
        self.boardsize = boardsize
        self.verbose = verbose
        self.timelimit = timelimit
        self.reset()

    def reset(self):
        self.p1 = snake(body = 5*[(3,5)], direction=(0,1), representation=[2,3])
        self.p2 = snake(body = 5*[(9,5)], direction=(0,1), representation=[4,5])
        self.map = self.blankmap()
        self.history = []
        self.time = 0
        return self._get_obs()

    def _get_obs(self):
        return np.eye(6)[self.map]


    def blankmap(self):
        map = np.zeros((self.boardsize, self.boardsize), dtype=np.int16)
        map = np.pad(map,pad_width=((1,1),(1,1)),constant_values=1)
        return map
    

    def step(self, input1, input2):
        self.history.append( (input1, input2))
        input1 = input2direction[input1]
        input2 = input2direction[input2]

        reward1, reward2, done = 0,0,False

        p1,p2 = self.p1, self.p2
        if input1 is not None: p1.direction = input1
        if input2 is not None: p2.direction = input2
        
        #check collision
        if self.collision(p1, p2, self.map):p1.alive = False
        if self.collision(p2, p1, self.map):p2.alive = False
        
        if p1.alive and not p2.alive:
            reward1=1; reward2=-1; done=True
        if p2.alive and not p1.alive:
            reward1=-1; reward2=1; done=True
        if not p1.alive and not p2.alive:
            reward1 = 0; reward2 = 0; done=True
        if p1.alive and p2.alive:
            reward1 = 0; reward2 = 0; done=False
        if self.time >= self.timelimit:
            reward1 = 0; reward2 = 0; done=True 

        #update snakes
        for player in [p1,p2]:
            if player.alive:
                player.body.pop()
                player.body.insert(0, player.body[0]+player.direction)
        

        #draw
        map = self.blankmap()
        for snake in [p1, p2]:
            map[tuple(snake.body[0])] = snake.repr_head
            tail = snake.body[1:]
            for part in tail:
                map[tuple(part)] = snake.repr_tail

        self.map = map
        if self.verbose: printmap(map)
        self.time+=1
        return self._get_obs(), reward1, reward2, done, {'time':self.time}

    def collision(self, p1, p2, map):
        p1nextheadlocation = p1.direction+p1.body[0]
        p2nextheadlocation = p2.direction+p2.body[0]
        
        if map[ tuple(p1nextheadlocation)]!=0 or (p2nextheadlocation==p1nextheadlocation).all():
            print()
            return True
        else:
            return False