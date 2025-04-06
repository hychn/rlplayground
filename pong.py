
import gym
import numpy as np

class TwoPlayerPong(gym.Env):
    def __init__(self):
        super(TwoPlayerPong, self).__init__()
        self.action_space = gym.spaces.Discrete(3)  # Actions: [UP, DOWN, STAY]
        self.observation_space = gym.spaces.Box(low=0, high=1, shape=(4,), dtype=np.float32)
        self.ball_x = 0.5
        self.ball_y = 0.5
        self.ball_velocity = 0.01
        self.paddle1_y = 0.5
        self.paddle2_y = 0.5

    def reset(self):
        # Reset ball and paddles
        self.ball_x = 0.5
        self.ball_y = 0.5
        self.ball_velocity = 0.01 * (1 if np.random.rand() > 0.5 else -1)
        self.paddle1_y = 0.5
        self.paddle2_y = 0.5
        return self._get_obs()

    def _get_obs(self):
        # Return state: [ball_x, ball_y, paddle1_y, paddle2_y]
        return np.array([self.ball_x, self.ball_y, self.paddle1_y, self.paddle2_y])

    def step(self, action1, action2):
        # Update paddles based on actions (player 1 and 2)
        self.paddle1_y = np.clip(self.paddle1_y + (0.1 if action1 == 0 else -0.1 if action1 == 1 else 0), 0, 1)
        self.paddle2_y = np.clip(self.paddle2_y + (0.1 if action2 == 0 else -0.1 if action2 == 1 else 0), 0, 1)

        # Update ball position
        self.ball_x += self.ball_velocity

        # Check for collisions with paddles
        if (self.ball_x <= 0.1 and abs(self.ball_y - self.paddle1_y) < 0.2) or \
           (self.ball_x >= 0.9 and abs(self.ball_y - self.paddle2_y) < 0.2):
            self.ball_velocity *= -1  # Reverse direction

        # Calculate rewards
        done = False
        reward1 = 0
        reward2 = 0

        if self.ball_x < 0:
            reward1 = -1  # Player 1 loses
            reward2 = 1   # Player 2 wins
            done = True
        elif self.ball_x > 1:
            reward1 = 1   # Player 1 wins
            reward2 = -1  # Player 2 loses
            done = True

        return self._get_obs(), reward1, reward2, done, {}
    

from keras.models import Sequential
from keras.layers import Dense, Conv2D, MaxPooling2D, Flatten
from keras.optimizers import Adam

class DQNAgent:
    def __init__(self, state_size, action_size):
        self.state_size = state_size
        self.action_size = action_size
        self.model = self._build_model()

    def _build_model(self):
        model = Sequential()


        model.add(Dense(24, input_dim=self.state_size, activation='relu'))
        model.add(Dense(24, activation='relu'))
        model.add(Dense(self.action_size, activation='linear'))
        model.compile(loss='mse', optimizer=Adam(learning_rate=0.001))
        return model

    def act(self, state, epsilon=0.1):
        if np.random.rand() <= epsilon:
            return np.random.choice(self.action_size)
        q_values = self.model.predict(state.reshape(1, -1), verbose=0)
        return np.argmax(q_values[0])
    


env = TwoPlayerPong()
state_size = env.observation_space.shape[0]
action_size = env.action_space.n

# Initialize agents
agent1 = DQNAgent(state_size, action_size)
agent2 = DQNAgent(state_size, action_size)

num_episodes = 1000
batch_size = 32

for episode in range(num_episodes):
    state = env.reset()
    total_reward1 = 0
    total_reward2 = 0
    done = False

    while not done:
        # Agents choose actions
        action1 = agent1.act(state)
        action2 = agent2.act(state)

        # Take actions and observe next state
        next_state, reward1, reward2, done, _ = env.step(action1, action2)

        # Store experiences (optional: use replay buffers)
        # Update models (simple Q-learning update)
        target1 = reward1 + 0.95 * np.max(agent1.model.predict(next_state.reshape(1, -1), verbose=0))
        target2 = reward2 + 0.95 * np.max(agent2.model.predict(next_state.reshape(1, -1), verbose=0))
        agent1.model.fit(state.reshape(1, -1), np.array([[target1]]), verbose=0)
        agent2.model.fit(state.reshape(1, -1), np.array([[target2]]), verbose=0)

        state = next_state
        total_reward1 += reward1
        total_reward2 += reward2

    print(f"Episode: {episode}, Reward1: {total_reward1}, Reward2: {total_reward2}")