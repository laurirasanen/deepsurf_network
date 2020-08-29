"""Module for neural network functionality"""

# =============================================================================
# >> IMPORTS
# =============================================================================
# Python
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import rpyc
from rpyc.utils.server import ThreadedServer


class NetworkService(rpyc.Service):
    class exposed_Network(object):
        __instance = None

        move_actions = 9  # none and 8 directions
        aim_actions = 3  # none and 2 directions

        # The first model makes the predictions for Q-values which are used to
        # make a action.
        model = None

        # Build a target model for the prediction of future rewards.
        # The weights of a target model get updated every 10000 steps thus when the
        # loss between the Q-values is calculated the target Q-value is stable.
        model_target = None

        optimizer = keras.optimizers.Adam(learning_rate=0.00025, clipnorm=1.0)

        # Experience replay buffers
        action_history = []
        state_history = []
        state_next_history = []
        rewards_history = []
        done_history = []
        episode_reward_history = []
        running_reward = 0
        episode_count = 0
        action_count = 0

        # Maximum replay length
        max_memory_length = 100000

        # Train the model after 4 actions
        update_after_actions = 4

        # How often to update the target network
        update_target_network = 10000

        # Using huber loss for stability
        loss_function = keras.losses.Huber()

        # Number of frames to take random action and observe output
        epsilon_random_frames = 0  # 50000

        # Number of frames for exploration
        epsilon_greedy_frames = 1000000.0

        # Configuration parameters for the whole setup
        gamma = 0.99  # Discount factor for past rewards
        epsilon = 1.0  # Epsilon greedy parameter
        epsilon_min = 0.1  # Minimum epsilon greedy parameter
        epsilon_max = 1.0  # Maximum epsilon greedy parameter
        epsilon_interval = (
                epsilon_max - epsilon_min
        )  # Rate at which to reduce chance of random action being taken
        batch_size = 32  # Size of batch taken from replay buffer
        max_steps_per_episode = 10000

        def __init__(self):
            self.model = self.create_q_model()
            self.model_target = self.create_q_model()

        def create_q_model(self):
            # Shape
            inputs = layers.Input(shape=(92, 92, 3,))

            # Convolutions
            layer1 = layers.Conv2D(32, 8, strides=4, activation="relu")(inputs)
            layer2 = layers.Conv2D(64, 4, strides=2, activation="relu")(layer1)
            layer3 = layers.Conv2D(64, 3, strides=1, activation="relu")(layer2)

            layer4 = layers.Flatten()(layer3)

            layer5 = layers.Dense(512, activation="relu")(layer4)
            move_actions = layers.Dense(self.move_actions, activation="linear")(layer5)

            layer6 = layers.Dense(512, activation="relu")(layer4)
            aim_actions = layers.Dense(self.aim_actions, activation="linear")(layer6)

            return keras.Model(inputs=inputs, outputs=(move_actions, aim_actions,))

        def exposed_get_action(self, state: tuple):
            self.action_count += 1

            # Use epsilon-greedy for exploration
            if self.action_count < self.epsilon_random_frames or self.epsilon > np.random.rand(1)[0]:
                # Take random action
                action = (np.random.choice(self.move_actions), np.random.choice(self.aim_actions),)
            else:
                # Predict action Q-values
                state_tensor = tf.convert_to_tensor(state)
                action_probs = self.model(state_tensor, training=False)
                # Take best action
                action = tf.argmax(action_probs[0]).numpy()

            # Decay probability of taking random action
            self.epsilon -= self.epsilon_interval / self.epsilon_greedy_frames
            self.epsilon = max(self.epsilon, self.epsilon_min)

            # Save actions and states in replay buffer
            self.action_history.append(action)
            self.state_history.append(state)
            return action

        def exposed_post_action(self, reward: float, state_next: any, done: bool):
            # Save actions and states in replay buffer
            self.state_next_history.append(state_next)
            self.done_history.append(done)
            self.rewards_history.append(reward)

            # Update every fourth frame once batch size is over 32
            if self.action_count % self.update_after_actions == 0 and len(self.done_history) > self.batch_size:
                # Get indices of samples for replay buffers
                indices = np.random.choice(range(len(self.done_history)), size=self.batch_size)

                # Using list comprehension to sample from replay buffer
                state_sample = np.array([tf.convert_to_tensor(self.state_history[i]) for i in indices])
                state_next_sample = np.array([tf.convert_to_tensor(self.state_next_history[i]) for i in indices])
                rewards_sample = [self.rewards_history[i] for i in indices]
                action_sample = [self.action_history[i] for i in indices]
                done_sample = tf.convert_to_tensor(
                    [float(self.done_history[i]) for i in indices]
                )

                # Build the updated Q-values for the sampled future states
                # Use the target model for stability
                future_rewards = self.model_target.predict(state_next_sample)
                # Q value = reward + discount factor * expected future reward
                updated_q_values = rewards_sample + self.gamma * tf.reduce_max(
                    future_rewards, axis=1
                )

                # If final frame set the last value to -1
                updated_q_values = updated_q_values * (1 - done_sample) - done_sample

                # Create a mask so we only calculate loss on the updated Q-values
                masks = tf.one_hot(action_sample, (self.move_actions, self.aim_actions,))

                with tf.GradientTape() as tape:
                    # Train the model on the states and updated Q-values
                    q_values = self.model(state_sample)

                    # Apply the masks to the Q-values to get the Q-value for action taken
                    q_action = tf.reduce_sum(tf.multiply(q_values, masks), axis=1)
                    # Calculate loss between new Q-value and old Q-value
                    loss = self.loss_function(updated_q_values, q_action)

                # Backpropagation
                grads = tape.gradient(loss, self.model.trainable_variables)
                self.optimizer.apply_gradients(zip(grads, self.model.trainable_variables))

            if self.action_count % self.update_target_network == 0:
                # update the the target network with new weights
                # update the the target network with new weights
                self.model_target.set_weights(self.model.get_weights())
                # Log details
                template = "running reward: {:.2f} at episode {}, frame count {}"
                print(template.format(self.running_reward, self.episode_count, self.action_count))

            # Limit the state and reward history
            if len(self.rewards_history) > self.max_memory_length:
                del self.rewards_history[:1]
                del self.state_history[:1]
                del self.state_next_history[:1]
                del self.action_history[:1]
                del self.done_history[:1]

        def exposed_end_episode(self, episode_reward: float):
            # Update running reward to check condition for solving
            self.episode_reward_history.append(episode_reward)
            if len(self.episode_reward_history) > 100:
                del self.episode_reward_history[:1]
            self.running_reward = np.mean(self.episode_reward_history)

            self.episode_count += 1


if __name__ == "__main__":
    server = ThreadedServer(NetworkService, port=18811)
    server.start()
