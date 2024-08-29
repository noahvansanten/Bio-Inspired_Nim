import numpy as np
import matplotlib.pyplot as plt
import math
from nim_env import NimEnv, OptimalPlayer

env = NimEnv()


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

n_states = 512 # 8*8*8 = 512
n_actions = 21 # 7+7+7 = 21 (but not all actions are always available)

# Reset Q_table
Q_table = np.zeros((n_states, n_actions))

# Reset Q_table
Q_table = np.zeros((n_states, n_actions))

alpha = 0.1
gamma = 0.99
eps_learner = 0.1
eps_trainer = 0.5
n_episodes = 20000

batch_size = 250  # Size of sample to average reward
rewards = np.zeros(n_episodes // batch_size)
reward = 0

n_wins_trainer = 0
n_wins_learner = 0

Turns = np.array([0, 1])

for episode in range(n_episodes):
    env.reset()
    heaps = env.heaps
    # Turns = Turns[np.random.permutation(2)]
    if episode % 2 == 0:
        Turns[0] = 0
        Turns[1] = 1
    else:
        Turns[0] = 1
        Turns[1] = 0
    # print("Turns = ",Turns)
    player_training = OptimalPlayer(epsilon=eps_trainer, player=Turns[0])
    player_learning = Turns[1]

    first_move = True

    while not env.end:
        # env.render()

        if env.current_player == player_training.player:
            move = player_training.act(heaps)
            heaps, end, winner = env.step(move)
            heaps_ind = game_to_ind(heaps)
            # print("Trainer player move :", move)
            if not (first_move):
                next_move_greedy_ind = pick_action(heaps, Q_table, epsilon=0)  # Q-Learning is off-policy
                delta_Q = alpha * (env.reward(player=player_learning) +
                                   gamma * Q_table[heaps_ind, next_move_greedy_ind] -
                                   Q_table[heaps_prev_ind, move_ind])
                Q_table[heaps_prev_ind, move_ind] += delta_Q
        else:
            while 1:  # Loop until we've found an authorised move
                move_ind = pick_action(heaps, Q_table, epsilon=eps_learner)
                move = action_ind_to_game(move_ind)
                if (env.heap_avail[move[0] - 1]) and (move[1] <= heaps[move[0] - 1]):  # Check the move is allowed
                    break
                else:  # Update the Q_table if it isn't
                    Q_table[game_to_ind(heaps), move_ind] = -np.inf
            # print("learner player move :", move)
            heaps_prev = heaps
            heaps_prev_ind = game_to_ind(heaps_prev)
            heaps, end, winner = env.step(move)
            heaps_ind = game_to_ind(heaps)

        first_move = False

        if end:
            if winner == player_training.player:
                n_wins_trainer += 1
            else:
                n_wins_learner += 1

            reward += env.reward(player=player_learning)
            if ((episode + 1) % batch_size) == 0:
                print("Episode : ", (episode + 1), "/", n_episodes)
                rewards[(episode + 1) // batch_size - 1] = reward / batch_size
                reward = 0

            # print("Game over, updating Q_table")
            # Q_table update without using the next move since the game is over
            delta_Q = alpha * (env.reward(player=player_learning) - Q_table[heaps_prev_ind, move_ind])
            Q_table[heaps_prev_ind, move_ind] += delta_Q
            env.reset()
            break

    env.reset()

plt.plot(rewards)
plt.title("eps = 0.1")
plt.ylabel("Average Reward")

alpha = 0.1
gamma = 0.99
eps_learner_min = 0.1
eps_learner_max = 0.8
n_maxes = np.linspace(1, 40000, 6)
eps_trainer = 0.5
n_episodes = 20000

batch_size = 250  # Size of sample to average reward
rewards_graphs = []
M_opt_graphs = []
M_rand_graphs = []

Turns = np.array([0, 1])

print("Progress : 0 /", len(n_maxes))
for idx, n_max in enumerate(n_maxes):
    # Reset Q_table
    Q_table = np.zeros((n_states, n_actions))

    rewards = np.zeros(n_episodes // batch_size)
    reward = 0
    M_opts = np.zeros(n_episodes // batch_size)
    M_rands = np.zeros(n_episodes // batch_size)

    for episode in range(n_episodes):
        eps_learner = max(eps_learner_min, eps_learner_max * (1 - episode / n_max))

        env.reset()
        heaps = env.heaps
        if episode % 2 == 0:
            Turns[0] = 0
            Turns[1] = 1
        else:
            Turns[0] = 1
            Turns[1] = 0
        player_training = OptimalPlayer(epsilon=eps_trainer, player=Turns[0])
        player_learning = Turns[1]

        first_move = True

        while not env.end:
            # env.render()

            if env.current_player == player_training.player:
                move = player_training.act(heaps)
                heaps, end, winner = env.step(move)
                heaps_ind = game_to_ind(heaps)
                # print("Trainer player move :", move)
                if not (first_move):
                    next_move_greedy_ind = pick_action(heaps, Q_table, epsilon=0)  # Q-Learning is off-policy
                    delta_Q = alpha * (env.reward(player=player_learning) +
                                       gamma * Q_table[heaps_ind, next_move_greedy_ind] -
                                       Q_table[heaps_prev_ind, move_ind])
                    Q_table[heaps_prev_ind, move_ind] += delta_Q
            else:
                while 1:  # Loop until we've found an authorised move
                    move_ind = pick_action(heaps, Q_table, epsilon=eps_learner)
                    move = action_ind_to_game(move_ind)
                    if (env.heap_avail[move[0] - 1]) and (move[1] <= heaps[move[0] - 1]):  # Check the move is allowed
                        break
                    else:  # Update the Q_table if it isn't
                        Q_table[game_to_ind(heaps), move_ind] = -np.inf
                # print("learner player move :", move)
                heaps_prev = heaps
                heaps_prev_ind = game_to_ind(heaps_prev)
                heaps, end, winner = env.step(move)
                heaps_ind = game_to_ind(heaps)

            first_move = False

            if end:
                # Q_table update without using the next move since the game is over
                delta_Q = alpha * (env.reward(player=player_learning) - Q_table[heaps_prev_ind, move_ind])
                Q_table[heaps_prev_ind, move_ind] += delta_Q

                reward += env.reward(player=player_learning)
                if ((episode + 1) % batch_size) == 0:
                    rewards[(episode + 1) // batch_size - 1] = reward / batch_size
                    reward = 0

                    # Test runs
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
                            # print("episode", episode+1, ", test", test, "M_opt", M_opt, "switching to eps_trainer = ", eps_trainer)

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
                                while 1:  # Loop until we've found an authorised move
                                    move_ind = pick_action(heaps, Q_table, epsilon=eps_learner)
                                    move = action_ind_to_game(move_ind)
                                    if (env.heap_avail[move[0] - 1]) and (
                                            move[1] <= heaps[move[0] - 1]):  # Check the move is allowed
                                        break
                                    else:  # Update the Q_table if it isn't
                                        Q_table[game_to_ind(heaps), move_ind] = -np.inf

                            heaps, end, winner = env.step(move)

                            if end:
                                if winner == player_training.player:
                                    n_wins_trainer += 1
                                else:
                                    n_wins_learner += 1

                                env.reset()
                                break

                        env.reset()
                    M_rand = (n_wins_learner - n_wins_trainer) / (n_wins_learner + n_wins_trainer)
                    M_opts[(episode + 1) // batch_size - 1] = M_opt
                    M_rands[(episode + 1) // batch_size - 1] = M_rand

                env.reset()
                break

        env.reset()
    print("Progress : ", idx + 1, "/", len(n_maxes))
    rewards_graphs.append(rewards)
    M_opt_graphs.append(M_opts)
    M_rand_graphs.append(M_rands)

# Plot rewards
plt.figure()
legends = []
for idx, rewards in enumerate(rewards_graphs):
    plt.plot(rewards)
    legend = "n_max = " + str(round(n_maxes[idx],1))
    legends.append(legend)
title = "Reward vs expert with decreasing exploration"
plt.title(title)
plt.ylabel("Average Reward")
plt.legend(legends)

# Plot M_opt and M_rand
plt.figure()
legends = []
for idx, M_opts in enumerate(M_opt_graphs):
    plt.plot(M_opts)
    legend = "n_max = " + str(round(n_maxes[idx], 1))
    legends.append(legend)
title = "M_opts vs expert with decreaing exploration"
plt.title(title)
plt.ylabel("Average M_opt")
plt.legend(legends)

# Plot M_opt and M_rand
plt.figure()
legends = []
for idx, M_rands in enumerate(M_rand_graphs):
    plt.plot(M_rands)
    legend = "n_max = " + str(round(n_maxes[idx], 1))
    legends.append(legend)
title = "M_rands vs expert with decreaing exploration"
plt.title(title)
plt.ylabel("Average M_rand")
plt.legend(legends)

alpha = 0.1
gamma = 0.99
eps_learner_min = 0.1
eps_learner_max = 0.8
n_max = 15000
eps_trainers = np.linspace(0, 1, 6)
n_episodes = 20000

batch_size = 250  # Size of sample to average reward
rewards_graphs = []
M_opt_graphs = []
M_rand_graphs = []

Turns = np.array([0, 1])

print("Progress : 0 /", len(eps_trainers))
for idx, eps_trainer in enumerate(eps_trainers):
    # Reset Q_table
    Q_table = np.zeros((n_states, n_actions))

    rewards = np.zeros(n_episodes // batch_size)
    reward = 0
    M_opts = np.zeros(n_episodes // batch_size)
    M_rands = np.zeros(n_episodes // batch_size)

    for episode in range(n_episodes):
        eps_learner = max(eps_learner_min, eps_learner_max * (1 - episode / n_max))

        env.reset()
        heaps = env.heaps
        if episode % 2 == 0:
            Turns[0] = 0
            Turns[1] = 1
        else:
            Turns[0] = 1
            Turns[1] = 0
        player_training = OptimalPlayer(epsilon=eps_trainer, player=Turns[0])
        player_learning = Turns[1]

        first_move = True

        while not env.end:
            # env.render()

            if env.current_player == player_training.player:
                move = player_training.act(heaps)
                heaps, end, winner = env.step(move)
                heaps_ind = game_to_ind(heaps)
                # print("Trainer player move :", move)
                if not (first_move):
                    next_move_greedy_ind = pick_action(heaps, Q_table, epsilon=0)  # Q-Learning is off-policy
                    delta_Q = alpha * (env.reward(player=player_learning) +
                                       gamma * Q_table[heaps_ind, next_move_greedy_ind] -
                                       Q_table[heaps_prev_ind, move_ind])
                    Q_table[heaps_prev_ind, move_ind] += delta_Q
            else:
                while 1:  # Loop until we've found an authorised move
                    move_ind = pick_action(heaps, Q_table, epsilon=eps_learner)
                    move = action_ind_to_game(move_ind)
                    if (env.heap_avail[move[0] - 1]) and (move[1] <= heaps[move[0] - 1]):  # Check the move is allowed
                        break
                    else:  # Update the Q_table if it isn't
                        Q_table[game_to_ind(heaps), move_ind] = -np.inf
                # print("learner player move :", move)
                heaps_prev = heaps
                heaps_prev_ind = game_to_ind(heaps_prev)
                heaps, end, winner = env.step(move)
                heaps_ind = game_to_ind(heaps)

            first_move = False

            if end:
                # Q_table update without using the next move since the game is over
                delta_Q = alpha * (env.reward(player=player_learning) - Q_table[heaps_prev_ind, move_ind])
                Q_table[heaps_prev_ind, move_ind] += delta_Q

                reward += env.reward(player=player_learning)
                if ((episode + 1) % batch_size) == 0:
                    rewards[(episode + 1) // batch_size - 1] = reward / batch_size
                    reward = 0

                    # Test runs
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
                            # print("episode", episode+1, ", test", test, "M_opt", M_opt, "switching to eps_trainer = ", eps_trainer)

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
                                while 1:  # Loop until we've found an authorised move
                                    move_ind = pick_action(heaps, Q_table, epsilon=eps_learner)
                                    move = action_ind_to_game(move_ind)
                                    if (env.heap_avail[move[0] - 1]) and (
                                            move[1] <= heaps[move[0] - 1]):  # Check the move is allowed
                                        break
                                    else:  # Update the Q_table if it isn't
                                        Q_table[game_to_ind(heaps), move_ind] = -np.inf

                            heaps, end, winner = env.step(move)

                            if end:
                                if winner == player_training.player:
                                    n_wins_trainer += 1
                                else:
                                    n_wins_learner += 1

                                env.reset()
                                break

                        env.reset()
                    M_rand = (n_wins_learner - n_wins_trainer) / (n_wins_learner + n_wins_trainer)
                    M_opts[(episode + 1) // batch_size - 1] = M_opt
                    M_rands[(episode + 1) // batch_size - 1] = M_rand

                env.reset()
                break

        env.reset()
    print("Progress : ", idx + 1, "/", len(eps_trainers))
    rewards_graphs.append(rewards)
    M_opt_graphs.append(M_opts)
    M_rand_graphs.append(M_rands)

# Plot M_opt and M_rand
plt.figure()
legends = []
for idx, M_opts in enumerate(M_opt_graphs):
    plt.plot(M_opts)
    legend = "eps_trainer = " + str(round(eps_trainers[idx], 1))
    legends.append(legend)
title = "M_opts vs expert with fixed eps_trainer"
plt.title(title)
plt.ylabel("Average M_opt")
plt.legend(legends)

# Plot M_opt and M_rand
plt.figure()
legends = []
for idx, M_rands in enumerate(M_rand_graphs):
    plt.plot(M_rands)
    legend = "eps_trainer = " + str(round(eps_trainers[idx], 1))
    legends.append(legend)
title = "M_rands vs expert with fixed eps_trainer"
plt.title(title)
plt.ylabel("Average M_rand")
plt.legend(legends)

eps_of_n = False  # Choose whether we want to vary epsilon or n_max (Questtion 7 or Question 8)

alpha = 0.1
gamma = 0.99
eps_min = 0.1
eps_max = 0.8
n_maxes = np.linspace(1, 40000, 6)
eps_consts = np.linspace(0, 1, 6)
eps_consts = [0.0]

n_episodes = 20000
batch_size = 250  # Size of intervals to compute M_opt and M_rand
M_opt_graphs = []
M_rand_graphs = []

Turns = np.array([0, 1])

if eps_of_n:
    variables = n_maxes
else:
    variables = eps_consts

print("Progress : 0 /", len(variables))
for idx, variable in enumerate(variables):
    # Reset Q_table
    Q_table = np.zeros((n_states, n_actions))

    M_opts = np.zeros(n_episodes // batch_size)
    M_rands = np.zeros(n_episodes // batch_size)

    for episode in range(n_episodes):
        if eps_of_n:  # variable = n_max
            eps = max(eps_min, eps_max * (1 - episode / variable))
        else:  # variable = eps_const
            eps = variable

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

        first_move = True

        while not env.end:
            # Action Selection
            if env.current_player == player_1:
                while 1:  # Loop until we've found an authorised move
                    move_1_ind = pick_action(heaps, Q_table, epsilon=eps)
                    move = action_ind_to_game(move_1_ind)
                    if (env.heap_avail[move[0] - 1]) and (move[1] <= heaps[move[0] - 1]):  # Check the move is allowed
                        break
                    else:  # Update the Q_table if it isn't
                        Q_table[game_to_ind(heaps), move_1_ind] = -np.inf
                heaps_prev = heaps
                heaps_prev_1_ind = game_to_ind(heaps_prev)
                heaps, end, winner = env.step(move)
                heaps_ind = game_to_ind(heaps)
                if not (first_move):
                    next_move_greedy_ind = pick_action(heaps, Q_table, epsilon=0)  # Q-Learning is off-policy
                    delta_Q = alpha * (env.reward(player=player_2) +
                                       gamma * Q_table[heaps_ind, next_move_greedy_ind] -
                                       Q_table[heaps_prev_2_ind, move_2_ind])
                    Q_table[heaps_prev_2_ind, move_2_ind] += delta_Q

            elif env.current_player == player_2:
                while 1:  # Loop until we've found an authorised move
                    move_2_ind = pick_action(heaps, Q_table, epsilon=eps)
                    move = action_ind_to_game(move_2_ind)
                    if (env.heap_avail[move[0] - 1]) and (move[1] <= heaps[move[0] - 1]):  # Check the move is allowed
                        break
                    else:  # Update the Q_table if it isn't
                        Q_table[game_to_ind(heaps), move_2_ind] = -np.inf
                heaps_prev = heaps
                heaps_prev_2_ind = game_to_ind(heaps_prev)
                heaps, end, winner = env.step(move)
                heaps_ind = game_to_ind(heaps)
                if not (first_move):
                    next_move_greedy_ind = pick_action(heaps, Q_table, epsilon=0)  # Q-Learning is off-policy
                    delta_Q = alpha * (env.reward(player=player_1) +
                                       gamma * Q_table[heaps_ind, next_move_greedy_ind] -
                                       Q_table[heaps_prev_1_ind, move_1_ind])
                    Q_table[heaps_prev_1_ind, move_1_ind] += delta_Q

            first_move = False

            # End of game
            if end:
                # Q_table update without using the next move since the game is over
                delta_Q_1 = alpha * (env.reward(player=player_1) - Q_table[heaps_prev_1_ind, move_1_ind])
                delta_Q_2 = alpha * (env.reward(player=player_2) - Q_table[heaps_prev_2_ind, move_2_ind])
                Q_table[heaps_prev_1_ind, move_1_ind] += delta_Q_1
                Q_table[heaps_prev_2_ind, move_2_ind] += delta_Q_2

                if ((episode + 1) % batch_size) == 0:
                    # Test runs
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
                            # print("episode", episode+1, ", test", test, "M_opt", M_opt, "switching to eps_trainer = ", eps_trainer)

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
                                while 1:  # Loop until we've found an authorised move
                                    move_ind = pick_action(heaps, Q_table, epsilon=eps_learner)
                                    move = action_ind_to_game(move_ind)
                                    if (env.heap_avail[move[0] - 1]) and (
                                            move[1] <= heaps[move[0] - 1]):  # Check the move is allowed
                                        break
                                    else:  # Update the Q_table if it isn't
                                        Q_table[game_to_ind(heaps), move_ind] = -np.inf

                            heaps, end, winner = env.step(move)

                            if end:
                                if winner == player_training.player:
                                    n_wins_trainer += 1
                                else:
                                    n_wins_learner += 1

                                env.reset()
                                break

                        env.reset()
                    M_rand = (n_wins_learner - n_wins_trainer) / (n_wins_learner + n_wins_trainer)
                    M_opts[(episode + 1) // batch_size - 1] = M_opt
                    M_rands[(episode + 1) // batch_size - 1] = M_rand

                env.reset()
                break

        env.reset()
    print("Progress : ", idx + 1, "/", len(variables))
    M_opt_graphs.append(M_opts)
    M_rand_graphs.append(M_rands)

if not(eps_of_n):
    # Plot M_opt and M_rand
    plt.figure()
    legends = []
    for idx, M_opts in enumerate(M_opt_graphs):
        plt.plot(M_opts)
        legend = "eps = " + str(round(variables[idx],1))
        legends.append(legend)
    title = "M_opts by self-practice with fixed eps"
    plt.title(title)
    plt.ylabel("Average M_opt")
    plt.legend(legends)

    # Plot M_opt and M_rand
    plt.figure()
    legends = []
    for idx, M_rands in enumerate(M_rand_graphs):
        plt.plot(M_rands)
        legend = "eps = " + str(round(variables[idx],1))
        legends.append(legend)
    title = "M_rands by self-practice with fixed eps"
    plt.title(title)
    plt.ylabel("Average M_rand")
    plt.legend(legends)

else:
    # Plot M_opt and M_rand
    plt.figure()
    legends = []
    for idx, M_opts in enumerate(M_opt_graphs):
        plt.plot(M_opts)
        legend = "n_max = " + str(round(variables[idx],1))
        legends.append(legend)
    title = "M_opts by self-practice with decreasing exploration"
    plt.title(title)
    plt.ylabel("Average M_opt")
    plt.legend(legends)

    # Plot M_opt and M_rand
    plt.figure()
    legends = []
    for idx, M_rands in enumerate(M_rand_graphs):
        plt.plot(M_rands)
        legend = "n_max = " + str(round(variables[idx],1))
        legends.append(legend)
    title = "M_rands by self-practice with decreasing exploration"
    plt.title(title)
    plt.ylabel("Average M_rand")
    plt.legend(legends)

x_label = ["1", "2", "3", "4", "5", "6", "7"]
y_label = ["Heap 1", "Heap 2", "Heap 3"]
for i in range(3):
    env.reset()
    print("Example", i+1)
    env.render()
    state = game_to_ind(env.heaps)
    Q_values = Q_table_to_game(Q_table, state)
    fig, ax = plt.subplots()
    im = ax.imshow(Q_values)
    ax.set_xticks(np.arange(7), labels=x_label)
    ax.set_yticks(np.arange(3), labels=y_label)
    ax.set_xlabel("Number of sticks to remove")
    ax.set_title("Q_values map")
    for j in range(3):
        for k in range(7):
            Q_values[j,k] = round(Q_values[j,k], 3)
            text = ax.text(k, j, Q_values[j,k], ha="center", va="center", color="w")
    print("i =",i)