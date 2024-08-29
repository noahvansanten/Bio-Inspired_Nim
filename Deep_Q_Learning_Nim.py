import numpy as np
import matplotlib.pyplot as plt
import math
from nim_env import NimEnv, OptimalPlayer
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
# from keras.utils import plot_model

# Build a map between the state number and the physical game
def ind_to_game(ind):  # Base 10 t0 base 8
    game0 = ind // (8 ** 2)
    ind -= game0 * (8 ** 2)
    game1 = ind // 8
    ind -= game1 * 8
    game2 = ind
    return [game0, game1, game2]


def game_to_ind(game):  # Base 8 to base 10
    ind = game[0] * (8 ** 2) + game[1] * 8 + game[2]
    return ind


# Build a map between action number and physical game
def action_ind_to_game(ind):
    game0 = ind // 7
    ind -= game0 * 7
    game1 = ind
    return [game0 + 1, game1 + 1]


def action_game_to_ind(game):
    ind = (game[0] - 1) * 7 + (game[1] - 1)
    return ind


# Change representation of Q_values from 1D list to 2D array corresponding to heaps
def Q_table_to_game(Q_table, state):
    return np.array([Q_table[state][0:7], Q_table[state][7:14], Q_table[state][14:21]])


# Useful state conversion function
def game_to_NN(game_state):
    NN_rep = np.zeros(9)

    bin_state_0 = np.binary_repr(game_state[0], width=3)
    bin_state_1 = np.binary_repr(game_state[1], width=3)
    bin_state_2 = np.binary_repr(game_state[2], width=3)

    NN_rep = [int(bin_state_0[0]), int(bin_state_0[1]), int(bin_state_0[2]),
              int(bin_state_1[0]), int(bin_state_1[1]), int(bin_state_1[2]),
              int(bin_state_2[0]), int(bin_state_2[1]), int(bin_state_2[2])]

    # NN_rep = tf.convert_to_tensor(NN_rep)
    return NN_rep


def mask_avail_actions():
    temp = np.zeros([512, 21])
    for i in range(512):
        i_8 = np.base_repr(i, base=8)
        while len(i_8) < 3:
            i_8 = "0" + i_8
        #         i_8 = i_8[::-1]
        for idx, j in enumerate(i_8):
            for k in range(8):
                if k < int(j):
                    temp[i, 7 * idx + k] = 1
    return temp


# Define an action policy

def pick_action(game_state, Q_table, epsilon=0):
    # implements greedy policy and returns the action with max. Q-value (given the state).
    # note: when Q-table is filled with zeros, returns a random policy.
    state_ind = game_to_ind(game_state)
    action_scores = Q_table[state_ind]

    if np.random.random() < epsilon:
        action = np.random.choice(np.where(action_scores != -np.inf)[0])  # An action of Q_value -np.inf isn't allowed
    else:
        #         print(game_state)
        #         print(action_scores)
        #         print(np.max(action_scores))
        #         print(np.where(action_scores == np.max(action_scores))[0])
        action = np.random.choice(np.where(action_scores == np.max(action_scores))[0])

    return action


def pick_action_DQN(state_ind, action_probs, mask_avail, epsilon=0):
    if np.random.random() < epsilon:
        action = np.random.choice(np.where(mask_avail[state_ind] != 0)[0])
    else:
        #         print(np.max(action_probs))
        action = np.random.choice(np.where(action_probs == np.max(action_probs))[0])

    return action


env = NimEnv()

# DQN Hyperparameters
gamma = 0.99
buffer_size = 10000
batch_size = 64
alpha = 0.0005
n_inputs = 9
n_actions = 21


# NN architecture
def create_q_model():
    inputs = layers.Input(shape=(n_inputs,))

    layer1 = layers.Dense(128, activation="relu")(inputs)
    layer2 = layers.Dense(128, activation="relu")(layer1)

    actions = layers.Dense(n_actions, activation="linear")(layer2)

    return keras.Model(inputs=inputs, outputs=actions)


optimizer = keras.optimizers.Adam(learning_rate=alpha, clipnorm=1.0)
loss_function = keras.losses.Huber()


def train_with_replay(model, model_target, optimizer,
                      state_history, state_next_history, action_history, rewards_history, done_history, loss_history,
                      batch_size=64, n_actions=21):
    # Random sampling of the replay buffer
    indices = np.random.choice(range(len(done_history)), size=batch_size)
    # Selection of the samples from the drawn indexes
    state_sample = np.array([state_history[i] for i in indices])
    state_next_sample = np.array([state_next_history[i] for i in indices])
    rewards_sample = [rewards_history[i] for i in indices]
    action_sample = [action_history[i] for i in indices]
    done_sample = tf.convert_to_tensor([float(done_history[i]) for i in indices])

    future_rewards = model_target(tf.convert_to_tensor(state_next_sample)).numpy()
    for i in range(batch_size):
        if done_sample[i]:
            future_rewards[i, :] = np.zeros([1, len(future_rewards[i, :])])

    #         print("future_rewards shape =", np.shape(future_rewards))
    #         print("future_rewards =", future_rewards)
    #         print("future max rewards = ", tf.reduce_max(future_rewards,axis=1))
    updated_q_values = rewards_sample + gamma * tf.reduce_max(future_rewards, axis=1)
    # print("updated_q_values = ", updated_q_values)

    masks = tf.one_hot(action_sample, n_actions)
    with tf.GradientTape() as tape:
        q_values = model(state_sample)
        q_action = tf.reduce_sum(tf.multiply(q_values, masks), axis=1)
        loss = loss_function(updated_q_values, q_action)
        loss_history.append(loss)
    #             print("Loss = ", loss)

    grads = tape.gradient(loss, model.trainable_variables)
    optimizer.apply_gradients(zip(grads, model.trainable_variables))


def train_without_replay(model, model_target, optimizer,
                         state, state_next, action, reward, done, loss_history,
                         batch_size_=1, n_action=21):
    future_rewards = model_target(tf.convert_to_tensor([state_next])).numpy()
    if done:
        future_rewards = np.zeros([1, len(future_rewards)])

    #         print("future_rewards shape =", np.shape(future_rewards))
    #         print("future_rewards =", future_rewards)
    #         print("future max rewards = ", tf.reduce_max(future_rewards,axis=1))
    updated_q_values = reward + gamma * tf.reduce_max(future_rewards, axis=1)
    # print("updated_q_values = ", updated_q_values)

    masks = tf.one_hot(action, n_actions)
    with tf.GradientTape() as tape:
        q_values = model(tf.convert_to_tensor([state])).numpy()
        q_action = tf.reduce_sum(tf.multiply(q_values, masks), axis=1)
        loss = loss_function(updated_q_values, q_action)
        loss_history.append(loss)


def test_runs(env, model):
    n_wins_trainer = 0
    n_wins_learner = 0
    for test in range(1000):
        eps_learner = 0
        if test < 500:
            eps_trainer_test = 0
        elif test == 500:  # switch to random trainer
            M_opt = (n_wins_learner - n_wins_trainer) / (n_wins_learner + n_wins_trainer)
            eps_trainer_test = 1
            n_wins_trainer = 0
            n_wins_learner = 0

        env.reset()
        heaps = env.heaps
        if test % 2 == 0:
            Turns[0] = 0
            Turns[1] = 1
        else:
            Turns[0] = 1
            Turns[1] = 0
        player_training = OptimalPlayer(epsilon=eps_trainer_test, player=Turns[0])
        player_learning = Turns[1]

        while not env.end:
            if env.current_player == player_training.player:
                move = player_training.act(heaps)
            else:
                state_NN = game_to_NN(heaps)
                action_probs = model(tf.convert_to_tensor([state_NN]))[0]
                move_ind = pick_action_DQN(game_to_ind(heaps), action_probs, mask_avail, epsilon=eps_learner)
                move = action_ind_to_game(move_ind)
                if not ((env.heap_avail[move[0] - 1])) or not (
                (move[1] <= heaps[move[0] - 1])):  # Check the move isn't allowed
                    n_wins_trainer += 1
                    break

            heaps, end, winner = env.step(move)

            if end:
                if winner == player_training.player:
                    n_wins_trainer += 1
                else:
                    n_wins_learner += 1

                env.reset()
                break

    M_rand = (n_wins_learner - n_wins_trainer) / (n_wins_learner + n_wins_trainer)
    return M_opt, M_rand


model = create_q_model()
model_target = create_q_model()

update_after_games = 1
update_target_network = 500

action_history = []
state_history = []
state_next_history = []
rewards_history = []
done_history = []
loss_history = []

loss_plot = []
game_rewards = []
rewards_plot = []
n_updates = 0
Turns = np.array([0, 1])
mask_avail = mask_avail_actions()

n_episodes = 20000
M_opts = np.zeros(n_episodes // 250)
M_rands = np.zeros(n_episodes // 250)

for episode in range(n_episodes):

    eps_trainer = 0.5
    eps_learner = 0.1

    env.reset()
    heaps = env.heaps
    if episode % 2 == 0:
        Turns[0] = 0
        Turns[1] = 1
    else:
        Turns[0] = 1
        Turns[1] = 0

    player_1 = Turns[0]
    player_2 = Turns[1]
    player_training = OptimalPlayer(epsilon=eps_trainer, player=Turns[0])
    player_learning = Turns[1]
    first_move = True
    #     print("New game")
    while not env.end:
        #         env.render()
        illegal_move = False

        if env.current_player == player_training.player:  # Teacher playing
            move = player_training.act(heaps)
            #             print("Teacher move :", move)
            heaps_next, end, winner = env.step(move)
            state_next_NN = game_to_NN(heaps_next)
            heaps = heaps_next
            agent_reward = env.reward(player=player_learning)
            if not (first_move) and not (end):
                #                 print("state next append : ", state_next_NN)
                state_next_history.append(state_next_NN)
                done_history.append(env.end)
                rewards_history.append(agent_reward)
                if (len(done_history) > batch_size):  # Training
                    train_with_replay(model, model_target, optimizer,
                                      state_history, state_next_history, action_history, rewards_history, done_history,
                                      loss_history)
                    n_updates += 1


        else:  # Student playing
            state_NN = game_to_NN(heaps)
            action_probs = model(tf.convert_to_tensor([state_NN]), training=False)[0]
            move_ind = pick_action_DQN(game_to_ind(heaps), action_probs, mask_avail, epsilon=eps_learner)
            move = action_ind_to_game(move_ind)
            #             print("Student move :", move)
            first_move = False
            if not (env.check_valid(move)):  # Check the move isn't allowed
                illegal_move = True
                env.end = True
                agent_reward = -1
                action_history.append(move_ind)
                state_history.append(state_NN)
                state_next_history.append(game_to_NN([0, 0, 0]))
                done_history.append(env.end)
                rewards_history.append(agent_reward)
                if (len(done_history) > batch_size):  # Training
                    train_with_replay(model, model_target, optimizer,
                                      state_history, state_next_history, action_history, rewards_history, done_history,
                                      loss_history)
                    n_updates += 1
            else:
                heaps_next, end, winner = env.step(move)
                state_next_NN = game_to_NN(heaps_next)
                heaps = heaps_next
                agent_reward = env.reward(player=player_learning)
                action_history.append(move_ind)
                state_history.append(state_NN)

        first_move = False

        if (env.end) and not (illegal_move):  # game over
            #             print("state next append :", state_next_NN)
            state_next_history.append(state_next_NN)
            done_history.append(env.end)
            rewards_history.append(agent_reward)
            if (len(done_history) > batch_size):  # Training
                train_with_replay(model, model_target, optimizer,
                                  state_history, state_next_history, action_history, rewards_history, done_history,
                                  loss_history)
                n_updates += 1

    # End of game here
    game_rewards.append(agent_reward)

    #     if ((episode+1) % update_after_games == 0) and (len(done_history) > batch_size): # Training
    #         train_with_replay(model, model_target, optimizer,
    #                           state_history, state_next_history, action_history, rewards_history, done_history, loss_history)
    #         n_updates += 1

    if (episode + 1) % update_target_network == 0:
        model_target.set_weights(model.get_weights())

    # logging loss rewards and test runs
    if (episode + 1) % 250 == 0:
        loss_plot.append(np.mean(loss_history[len(loss_history) - n_updates:len(loss_history)]))
        n_updates = 0
        rewards_plot.append(np.mean(game_rewards))
        game_rewards = []

        #         M_opt, M_rand = test_runs(env, model)
        #         M_opts[(episode+1)//250 - 1] = M_opt
        #         M_rands[(episode+1)//250 - 1] = M_rand

        print("Progress :", episode + 1, "/", n_episodes)

    if len(rewards_history) > buffer_size:
        del rewards_history[:len(rewards_history) - buffer_size]
        del state_history[:len(state_history) - buffer_size]
        del state_next_history[:len(state_next_history) - buffer_size]
        del action_history[:len(action_history) - buffer_size]
        del done_history[:len(done_history) - buffer_size]

# Plot Reward and loss
plt.figure()
plt.plot(loss_plot)
title = "Average loss history vs expert, eps = 0.1"
plt.title(title)
plt.ylabel("Average loss")

plt.figure()
plt.plot(rewards_plot)
title = "Average reward history vs expert, eps = 0.1"
plt.title(title)
plt.ylabel("Average reward")

eps_min = 0.1
eps_max = 0.9
n_maxes = np.linspace(1,40000,6)

M_opts_graph = []
M_rands_graph = []

for n_max in n_maxes:
    model = create_q_model()
    model_target = create_q_model()

    update_after_games = 1
    update_target_network = 500

    action_history = []
    state_history = []
    state_next_history = []
    rewards_history = []
    done_history = []
    loss_history = []

    loss_plot = []
    game_rewards = []
    rewards_plot = []
    n_updates = 0
    Turns = np.array([0,1])
    mask_avail = mask_avail_actions()

    n_episodes = 20000
    M_opts = np.zeros(n_episodes//250)
    M_rands = np.zeros(n_episodes//250)

    for episode in range(n_episodes):

        eps_trainer = 0.5
        eps_learner = max(eps_min, eps_max*(1 - episode/n_max))

        env.reset()
        heaps = env.heaps
        if episode%2 == 0:
            Turns[0] = 0
            Turns[1] = 1
        else:
            Turns[0] = 1
            Turns[1] = 0

        player_1 = Turns[0]
        player_2 = Turns[1]
        player_training = OptimalPlayer(epsilon=eps_trainer, player=Turns[0])
        player_learning = Turns[1]
        first_move = True
    #     print("New game")
        while not env.end:
    #         env.render()
            illegal_move = False

            if env.current_player == player_training.player: # Teacher playing
                move = player_training.act(heaps)
    #             print("Teacher move :", move)
                heaps_next, end, winner = env.step(move)
                state_next_NN = game_to_NN(heaps_next)
                heaps = heaps_next
                agent_reward = env.reward(player=player_learning)
                if not(first_move) and not(end):
    #                 print("state next append : ", state_next_NN)
                    state_next_history.append(state_next_NN)
                    done_history.append(env.end)
                    rewards_history.append(agent_reward)
                    if (len(done_history) > batch_size): # Training
                        optimizer = keras.optimizers.Adam(learning_rate=alpha, clipnorm=1.0)
                        train_with_replay(model, model_target, optimizer,
                                          state_history, state_next_history, action_history, rewards_history, done_history, loss_history)
                        n_updates += 1


            else: # Student playing
                state_NN = game_to_NN(heaps)
                action_probs = model(tf.convert_to_tensor([state_NN]), training=False)[0]
                move_ind = pick_action_DQN(game_to_ind(heaps), action_probs, mask_avail, epsilon=eps_learner)
                move = action_ind_to_game(move_ind)
    #             print("Student move :", move)
                first_move = False
                if not(env.check_valid(move)): # Check the move isn't allowed
                    illegal_move = True
                    env.end = True
                    agent_reward = -1
                    action_history.append(move_ind)
                    state_history.append(state_NN)
                    state_next_history.append(game_to_NN([0,0,0]))
                    done_history.append(env.end)
                    rewards_history.append(agent_reward)
                    if (len(done_history) > batch_size): # Training
                        optimizer = keras.optimizers.Adam(learning_rate=alpha, clipnorm=1.0)
                        train_with_replay(model, model_target, optimizer,
                                          state_history, state_next_history, action_history, rewards_history, done_history, loss_history)
                        n_updates += 1
                else:
                    heaps_next, end, winner = env.step(move)
                    state_next_NN = game_to_NN(heaps_next)
                    heaps = heaps_next
                    agent_reward = env.reward(player=player_learning)
                    action_history.append(move_ind)
                    state_history.append(state_NN)

            first_move = False

            if (env.end) and not(illegal_move): # game over
    #             print("state next append :", state_next_NN)
                state_next_history.append(state_next_NN)
                done_history.append(env.end)
                rewards_history.append(agent_reward)
                if (len(done_history) > batch_size): # Training
                    optimizer = keras.optimizers.Adam(learning_rate=alpha, clipnorm=1.0)
                    train_with_replay(model, model_target, optimizer,
                                      state_history, state_next_history, action_history, rewards_history, done_history, loss_history)
                    n_updates += 1

        # End of game here
        game_rewards.append(agent_reward)

        if (episode+1) % update_target_network == 0:
            model_target.set_weights(model.get_weights())

        # logging loss rewards and test runs
        if (episode+1) % 250 == 0:

            loss_plot.append(np.mean(loss_history[len(loss_history)-n_updates:len(loss_history)]))
            n_updates = 0
            rewards_plot.append(np.mean(game_rewards))
            game_rewards = []

            M_opt, M_rand = test_runs(env, model)
            M_opts[(episode+1)//250 - 1] = M_opt
            M_rands[(episode+1)//250 - 1] = M_rand

            print("Progress : n_max =", n_max, ", episode =", episode+1, "/", n_episodes)


        if len(rewards_history) > buffer_size:
            del rewards_history[:len(rewards_history)-buffer_size]
            del state_history[:len(state_history)-buffer_size]
            del state_next_history[:len(state_next_history)-buffer_size]
            del action_history[:len(action_history)-buffer_size]
            del done_history[:len(done_history)-buffer_size]
    M_opts_graph.append(M_opts)
    M_rands_graph.append(M_rands)

# Plot M_opt and M_rand
plt.figure()
legends = []
for idx, M_opts in enumerate(M_opts_graph):
    plt.plot(M_opts)
    legend = "n_max = " + str(round(n_maxes[idx],1))
    legends.append(legend)
title = "M_opts by self-practice with varying n_max"
plt.title(title)
plt.ylabel("Average M_opt")
plt.legend(legends)