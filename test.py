from blessed import Terminal

term = Terminal()

# Print "Hello, world!" at column 10, row 5
print(term.move_xy(0, 0) + "Hello, world!")