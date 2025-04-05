import numpy as np

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
                3: 's', 8: 'M', 9: 'm'}

    # Convert numbers to characters
    char_matrix = np.vectorize(char_map.get)(map)

    # Print in grid format
    for row in char_matrix:
        print(' '.join(row))


class game:
    #TODO 
    #def _get_obs(self): 
    #     returns np array input to model
    #def step(self, action1, action2)


    def __init__(self):
        self.p1 = snake(body = 10*[(1,1)], direction=(0,1), representation=[2,3])
        self.p2 = snake(body = 5*[(8,1)], direction=(0,1), representation=[4,5])
        self.map = self.blankmap()

    def get_obs(self):
        return np.eye(6)[self.map]


    def blankmap(self):
        map = np.zeros((30,30))
        map = np.pad(map,pad_width=((1,1),(1,1)),constant_values=1)
        return map

    def step(self, input1, input2):
        #return self._get_obs(), reward1, reward2, done, {}
        reward1, reward2, done = 0,0,False

        p1,p2 = self.p1, self.p2
        if input1 is not None: p1.direction = input1
        if input2 is not None: p2.direction = input2
        
        #check collision
        if self.collision(p1, p2, self.map):p1.alive = False
        if self.collision(p2, p1, self.map):p2.alive = False
        
        if p1.alive and not p2.alive:
            reward1=1, reward2=-1, done=True
        if p2.alive and not p1.alive:
            reward1=-1, reward2=1, done=True
        if not p1.alive and not p2.alive:
            reward1 = 0, reward2 = 0, done=True
        if p1.alive and p2.alive:
            reward1 = 0, reward2 = 0. done=False

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
        return self._get_obs(), reward1, reward2, done, {}

    def collision(self, p1, p2, map):
        p1nextheadlocation = p1.direction+p1.body[0]
        p2nextheadlocation = p2.direction+p2.body[0]
        
        if map[ tuple(p1nextheadlocation)]!=0 or (p2nextheadlocation==p1nextheadlocation).all():
            print()
            return True
        else:
            return False

#import time
#game = game()
#for i in range(100):
    #game.update()
    #time.sleep(.5)

