"""Module for neural network functionality"""

# =============================================================================
# >> IMPORTS
# =============================================================================
# Python
import time
import os
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import rpyc
from rpyc.utils.server import ThreadedServer
import pickle

rpyc.core.protocol.DEFAULT_CONFIG["allow_pickle"] = True


class NetworkService(rpyc.Service):
    class exposed_Network(object):
        __instance = None

        move_actions = 9  # none and 8 directions
        max_aim_action = (
            1.0  # max degrees of rotation per frame when taking random actions
        )
        aim_actions = 201  # 100 values both directions and 0

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
        episode_frame_count = 0
        running_reward = 0
        episode_count = 0
        action_count = 0

        # Maximum replay length
        max_memory_length = 100000

        # Train the model after actions
        update_after_actions = 4096

        # How often to update the target network
        update_target_network = 10000

        # Using huber loss for stability
        loss_function = keras.losses.Huber()

        # Number of frames to take random action and observe output
        epsilon_random_frames = 50000

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
        batch_size = 4096  # Size of batch taken from replay buffer

        def __init__(self):
            self.model = self.create_q_model()
            self.model_target = self.create_q_model()
            print(self.model.summary())

        def create_q_model(self):
            inputs = layers.Input(shape=(193,))

            layer1 = layers.Dense(2048, activation="relu")(inputs)
            layer2 = layers.Dense(2048, activation="relu")(layer1)
            layer3 = layers.Dense(2048, activation="relu")(layer2)

            move_actions = layers.Dense(self.move_actions, activation="softmax")(layer3)
            yaw_action = layers.Dense(self.aim_actions, activation="softmax")(layer3)
            pitch_action = layers.Dense(self.aim_actions, activation="softmax")(layer3)
            jump_action = layers.Dense(2, activation="softmax")(layer3)
            duck_action = layers.Dense(2, activation="softmax")(layer3)

            model = keras.Model(
                inputs=inputs,
                outputs=(
                    move_actions,
                    yaw_action,
                    pitch_action,
                    jump_action,
                    duck_action,
                ),
            )

            # https://stackoverflow.com/questions/49503748/save-and-load-model-optimizer-state
            if len(self.optimizer.get_weights()) == 0 and os.path.isfile(
                "optimizer.pkl"
            ):
                with open("optimizer.pkl", mode="rb") as file:
                    ow = pickle.load(file)
                    grad_vars = model.trainable_weights
                    zero_grads = [tf.zeros_like(w) for w in grad_vars]
                    self.optimizer.apply_gradients(zip(zero_grads, grad_vars))
                    self.optimizer.set_weights(ow)

            if os.path.isfile("weights.h5"):
                model.load_weights("weights.h5")

            if os.path.isfile("settings.pkl"):
                with open("settings.pkl", mode="rb") as file:
                    s = pickle.load(file)
                    self.epsilon = s["epsilon"]
                    self.action_count = s["action_count"]
                    self.episode_count = s["episode_count"]

            return model

        def exposed_get_action(self, state):
            state = pickle.loads(state)
            self.action_count += 1

            # Use epsilon-greedy for exploration
            if (
                self.action_count < self.epsilon_random_frames
                or self.epsilon > np.random.rand(1)[0]
            ):
                # Take random action
                action = (
                    np.random.choice(self.move_actions),
                    np.random.choice(self.aim_actions),
                    np.random.choice(self.aim_actions),
                    np.random.choice(2),
                    np.random.choice(2),
                )
            else:
                # Predict action Q-values
                state_tensor = tf.convert_to_tensor(state)
                state_tensor = tf.expand_dims(state_tensor, 0)
                action_probs = self.model(state_tensor, training=False)
                # Take best action
                action = (
                    tf.math.argmax(action_probs[0][0]).numpy(),
                    tf.math.argmax(action_probs[1][0]).numpy(),
                    tf.math.argmax(action_probs[2][0]).numpy(),
                    tf.math.argmax(action_probs[3][0]).numpy(),
                    tf.math.argmax(action_probs[4][0]).numpy(),
                )

            # Decay probability of taking random action
            self.epsilon -= self.epsilon_interval / self.epsilon_greedy_frames
            self.epsilon = max(self.epsilon, self.epsilon_min)

            # Save actions and states in replay buffer
            self.action_history.append(action)
            self.state_history.append(state)
            # Always return ints -
            # numpy -> int conversion takes about 4ms for each variable in Source.Python
            return (
                int(action[0]),
                int(action[1]),
                int(action[2]),
                int(action[3]),
                int(action[4]),
            )

        # get action without exploration for running
        def exposed_get_action_run(self, state):
            state = pickle.loads(state)

            # Predict action Q-values
            state_tensor = tf.convert_to_tensor(state)
            state_tensor = tf.expand_dims(state_tensor, 0)
            action_probs = self.model(state_tensor, training=False)
            # Take best action
            action = (
                int(tf.math.argmax(action_probs[0][0]).numpy()),
                int(tf.math.argmax(action_probs[1][0]).numpy()),
                int(tf.math.argmax(action_probs[2][0]).numpy()),
                int(tf.math.argmax(action_probs[3][0]).numpy()),
                int(tf.math.argmax(action_probs[4][0]).numpy()),
            )
            return action

        def exposed_post_action(self, reward: float, state_next, done: bool):
            state_next = pickle.loads(state_next)
            # Save actions and states in replay buffer
            self.state_next_history.append(state_next)
            self.done_history.append(done)
            self.rewards_history.append(reward)
            self.episode_frame_count += 1

            # Update
            if (
                self.action_count % self.update_after_actions == 0
                and len(self.done_history) > self.batch_size
            ):
                # Get indices of samples for replay buffers
                indices = np.random.choice(
                    range(len(self.done_history)), size=self.batch_size
                )

                # Using list comprehension to sample from replay buffer
                state_sample = np.array([self.state_history[i] for i in indices])
                # state_sample = [self.state_history[i] for i in indices]
                state_next_sample = np.array(
                    [self.state_next_history[i] for i in indices]
                )
                # state_next_sample = [self.state_next_history[i] for i in indices]
                rewards_sample = [self.rewards_history[i] for i in indices]
                move_action_sample = [self.action_history[i][0] for i in indices]
                yaw_action_sample = [self.action_history[i][1] for i in indices]
                pitch_action_sample = [self.action_history[i][2] for i in indices]
                jump_action_sample = [self.action_history[i][3] for i in indices]
                duck_action_sample = [self.action_history[i][4] for i in indices]
                done_sample = tf.convert_to_tensor(
                    [float(self.done_history[i]) for i in indices]
                )

                # Build the updated Q-values for the sampled future states
                # Use the target model for stability
                state_tensor = tf.convert_to_tensor(state_next_sample)
                future_rewards = self.model_target.predict(state_tensor)
                # Q value = reward + discount factor * expected future reward
                # TODO: this is probably wrong
                updated_q_values = rewards_sample + self.gamma * (
                    tf.math.reduce_max(future_rewards[0], axis=1)
                    + tf.math.reduce_max(future_rewards[1], axis=1)
                    + tf.math.reduce_max(future_rewards[2], axis=1)
                    + tf.math.reduce_max(future_rewards[3], axis=1)
                    + tf.math.reduce_max(future_rewards[4], axis=1)
                )

                # If final frame set the last value to -1
                updated_q_values = updated_q_values * (1 - done_sample) - done_sample

                # Create a mask so we only calculate loss on the updated Q-values
                mask1 = tf.one_hot(move_action_sample, self.move_actions)
                mask2 = tf.one_hot(yaw_action_sample, self.aim_actions)
                mask3 = tf.one_hot(pitch_action_sample, self.aim_actions)
                mask4 = tf.one_hot(jump_action_sample, 2)
                mask5 = tf.one_hot(duck_action_sample, 2)

                with tf.GradientTape() as tape:
                    # Train the model on the states and updated Q-values
                    q_values = self.model(state_sample)

                    # Apply the masks to the Q-values to get the Q-value for action taken
                    q_action1 = tf.reduce_sum(tf.multiply(q_values[0], mask1), axis=1)
                    q_action2 = tf.reduce_sum(tf.multiply(q_values[1], mask2), axis=1)
                    q_action3 = tf.reduce_sum(tf.multiply(q_values[2], mask3), axis=1)
                    q_action4 = tf.reduce_sum(tf.multiply(q_values[3], mask4), axis=1)
                    q_action5 = tf.reduce_sum(tf.multiply(q_values[4], mask5), axis=1)

                    # Calculate loss between new Q-value and old Q-value
                    loss = self.loss_function(updated_q_values, q_action1)
                    loss += self.loss_function(updated_q_values, q_action2)
                    loss += self.loss_function(updated_q_values, q_action3)
                    loss += self.loss_function(updated_q_values, q_action4)
                    loss += self.loss_function(updated_q_values, q_action5)

                # Backpropagation
                grads = tape.gradient(loss, self.model.trainable_variables)
                self.optimizer.apply_gradients(
                    zip(grads, self.model.trainable_variables)
                )

            if self.action_count % self.update_target_network == 0:
                # update the the target network with new weights
                self.model_target.set_weights(self.model.get_weights())
                # Log details
                template = "running reward: {:.2f} at episode {}, frame count {}"
                print(
                    template.format(
                        self.running_reward, self.episode_count, self.action_count
                    )
                )
                # save
                with open("optimizer.pkl", mode="wb") as file:
                    ow = self.optimizer.get_weights()
                    pickle.dump(ow, file)
                # self.model.save("model")
                self.model.save_weights("weights.h5")
                with open("settings.pkl", mode="wb") as file:
                    s = {}
                    s["epsilon"] = self.epsilon
                    s["action_count"] = self.action_count
                    s["episode_count"] = self.episode_count
                    pickle.dump(s, file)

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
            print(
                f"episode {self.episode_count} reward {self.episode_reward_history[-1]}, mean frame {np.mean(self.rewards_history[-self.episode_frame_count:])}"
            )
            self.episode_frame_count = 0


if __name__ == "__main__":
    server = ThreadedServer(NetworkService, port=18811)
    server.start()
