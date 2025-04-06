
import random
import snakegame
from blessed import Terminal

import numpy as np

term = Terminal()

game = snakegame.game()

with term.fullscreen(), term.hidden_cursor(), term.cbreak():

    while True:
        key = term.inkey(0.1)
        code = key.code if key else None
        newdir = None
        if code == term.KEY_UP:
            newdir=1
        elif code == term.KEY_DOWN:
            newdir=3
        elif code == term.KEY_LEFT:
            newdir=2
        elif code == term.KEY_RIGHT:
            newdir=0

        game.step(newdir, newdir)

        char_map = {0: ' ', 1: '#', 2: 'S', 3: 's', 4: 'M', 5: 'm'}
        char_matrix = np.vectorize(char_map.get)(game.map)
        for i,row in enumerate(char_matrix):
            #print(str(row))
            print(term.move(i, 0) + ''.join(row))
            #print(term.move(i, 0) + 'A')

    print(term.move(term.height//2, term.width//2 - 5) + "Game Over!")
    print(term.move(term.height//2 + 1, term.width//2 - 8) + f"Final Score: {score}")
    term.inkey()