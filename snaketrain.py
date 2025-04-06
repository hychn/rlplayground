import snakegame
import gym
import numpy as np
import tensorflow as tf
import keras
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
        model.add(Conv2D(64, (3, 3), activation='relu', padding='same', input_shape=self.state_size))
        model.add(Conv2D(64, (3, 3), activation='relu', padding='same'))
        model.add(MaxPooling2D((2, 2)))
        model.add(Conv2D(128, (3, 3), activation='relu', padding='same'))
        model.add(Conv2D(128, (3, 3), activation='relu', padding='same'))
        model.add(MaxPooling2D((2, 2)))
    
        # More layers...
        model.add(Flatten())



        model.add(Dense(24, input_dim=self.state_size, activation='relu'))
        model.add(Dense(24, activation='relu'))
        model.add(Dense(self.action_size, activation='linear'))
        model.compile(loss='mse', optimizer=Adam(learning_rate=0.001))
        return model

    def act(self, state, epsilon=0.1):
        if np.random.rand() <= epsilon:
            return np.random.choice(self.action_size)
        
        if len(state.shape) <4:
            state = tf.expand_dims(state, axis=0)
        q_values = self.model.predict(state, verbose=0)
        return np.argmax(q_values[0])
    


env = snakegame.game(verbose=False)
state_size = env.observation_space.shape
action_size = env.action_space.n

# Initialize agents
agent1 = DQNAgent(state_size, action_size)
agent2 = DQNAgent(state_size, action_size)
#LOAD
agent1.model = keras.models.load_model('agent1.keras'); agent2.model = keras.models.load_model('agent2.keras')

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
        next_state, reward1, reward2, done, info = env.step(action1, action2)

        # Store experiences (optional: use replay buffers)
        # Update models (simple Q-learning update)
        if len(next_state.shape) <4: next_state = tf.expand_dims(next_state, axis=0)
        target1 = reward1 + 0.95 * np.max(agent1.model.predict(next_state, verbose=0))
        target2 = reward2 + 0.95 * np.max(agent2.model.predict(next_state, verbose=0))
        if len(state.shape) <4: state = tf.expand_dims(state, axis=0)
        agent1.model.fit(state, np.array([[target1]]), verbose=0)
        agent2.model.fit(state, np.array([[target2]]), verbose=0)

        state = next_state
        total_reward1 += reward1
        total_reward2 += reward2

    if episode % 50 ==0:
        agent1.model.save('agent1.keras')
        agent2.model.save('agent2.keras')

    print(f"Episode: {episode}, Reward1: {total_reward1}, Reward2: {total_reward2}, GAME LENGTH: {info['time']}")