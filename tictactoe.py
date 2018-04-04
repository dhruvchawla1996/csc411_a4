from __future__ import print_function
from collections import defaultdict
from itertools import count
import numpy as np
import math
import random
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.distributions
from torch.autograd import Variable
import matplotlib.pyplot as plt

# Set seeds
np.random.seed(42)
random.seed(42)
torch.manual_seed(42)

class Environment(object):
    """
    The Tic-Tac-Toe Environment
    """
    # possible ways to win
    win_set = frozenset([(0,1,2), (3,4,5), (6,7,8), # horizontal
                         (0,3,6), (1,4,7), (2,5,8), # vertical
                         (0,4,8), (2,4,6)])         # diagonal
    # statuses
    STATUS_VALID_MOVE = 'valid'
    STATUS_INVALID_MOVE = 'inv'
    STATUS_WIN = 'win'
    STATUS_TIE = 'tie'
    STATUS_LOSE = 'lose'
    STATUS_DONE = 'done'

    def __init__(self):
        self.reset()

    def reset(self):
        """Reset the game to an empty board."""
        self.grid = np.array([0] * 9) # grid
        self.turn = 1                 # whose turn it is
        self.done = False             # whether game is done
        return self.grid

    def render(self):
        """Print what is on the board."""
        map = {0:'.', 1:'x', 2:'o'} # grid label vs how to plot
        print(''.join(map[i] for i in self.grid[0:3]))
        print(''.join(map[i] for i in self.grid[3:6]))
        print(''.join(map[i] for i in self.grid[6:9]))
        print('====')

    def check_win(self):
        """Check if someone has won the game."""
        for pos in self.win_set:
            s = set([self.grid[p] for p in pos])
            if len(s) == 1 and (0 not in s):
                return True
        return False

    def step(self, action):
        """Mark a point on position action."""
        assert type(action) == int and action >= 0 and action < 9
        # done = already finished the game
        if self.done:
            return self.grid, self.STATUS_DONE, self.done
        # action already have something on it
        if self.grid[action] != 0:
            return self.grid, self.STATUS_INVALID_MOVE, self.done
        # play move
        self.grid[action] = self.turn
        if self.turn == 1:
            self.turn = 2
        else:
            self.turn = 1
        # check win
        if self.check_win():
            self.done = True
            return self.grid, self.STATUS_WIN, self.done
        # check tie
        if all([p != 0 for p in self.grid]):
            self.done = True
            return self.grid, self.STATUS_TIE, self.done
        return self.grid, self.STATUS_VALID_MOVE, self.done

    def random_step(self):
        """Choose a random, unoccupied move on the board to play."""
        pos = [i for i in range(9) if self.grid[i] == 0]
        move = random.choice(pos)
        return self.step(move)

    def play_against_random(self, action):
        """Play a move, and then have a random agent play the next move."""
        state, status, done = self.step(action)
        if not done and self.turn == 2:
            state, s2, done = self.random_step()
            if done:
                if s2 == self.STATUS_WIN:
                    status = self.STATUS_LOSE
                elif s2 == self.STATUS_TIE:
                    status = self.STATUS_TIE
                else:
                    raise ValueError("???")
        return state, status, done

class Policy(nn.Module):
    """
    The Tic-Tac-Toe Policy
    """
    def __init__(self, input_size=27, hidden_size=256, output_size=9):
        super(Policy, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        softmax = nn.Softmax()
        return softmax(x)

def select_action(policy, state):
    """Samples an action from the policy at the state."""
    state = torch.from_numpy(state).long().unsqueeze(0)
    state = torch.zeros(3,9).scatter_(0,state,1).view(1,27)
    pr = policy(Variable(state))
    m = torch.distributions.Categorical(pr) 
    action = m.sample()
    log_prob = torch.sum(m.log_prob(action))
    return action.data[0], log_prob

def compute_returns(rewards, gamma=1.0):
    """
    Compute returns for each time step, given the rewards
      @param rewards: list of floats, where rewards[t] is the reward
                      obtained at time step t
      @param gamma: the discount factor
      @returns list of floats representing the episode's returns
          G_t = r_t + \gamma r_{t+1} + \gamma^2 r_{t+2} + ... 

    >>> compute_returns([0,0,0,1], 1.0)
    [1.0, 1.0, 1.0, 1.0]
    >>> compute_returns([0,0,0,1], 0.9)
    [0.7290000000000001, 0.81, 0.9, 1.0]
    >>> compute_returns([0,-0.5,5,0.5,-10], 0.9)
    [-2.5965000000000003, -2.8850000000000002, -2.6500000000000004, -8.5, -10.0]
    """
    G = [0] * len(rewards)

    for i in range(len(rewards)-1, -1, -1):
        if i == len(rewards)-1: G[i] = rewards[i]
        else: G[i] = rewards[i] + gamma * G[i+1]

    return G

def finish_episode(saved_rewards, saved_logprobs, gamma=1.0):
    """Samples an action from the policy at the state."""
    policy_loss = []
    returns = compute_returns(saved_rewards, gamma)
    returns = torch.Tensor(returns)
    # subtract mean and std for faster training
    returns = (returns - returns.mean()) / (returns.std() +
                                            np.finfo(np.float32).eps)
    for log_prob, reward in zip(saved_logprobs, returns):
        policy_loss.append(-log_prob * reward)
    policy_loss = torch.cat(policy_loss).sum()
    policy_loss.backward(retain_graph=True)
    # note: retain_graph=True allows for multiple calls to .backward()
    # in a single step

#TODO: play around with reward values
def get_reward(status):
    """Returns a numeric given an environment status."""
    return {
            Environment.STATUS_VALID_MOVE  : 1,
            Environment.STATUS_INVALID_MOVE: -250,
            Environment.STATUS_WIN         : 500,
            Environment.STATUS_TIE         : -3,
            Environment.STATUS_LOSE        : -3
    }[status]

def train(policy, env, gamma=0.75, log_interval=1000):
    """Train policy gradient."""
    optimizer = optim.Adam(policy.parameters(), lr=0.001)
    scheduler = torch.optim.lr_scheduler.StepLR(
            optimizer, step_size=10000, gamma=0.9)
    running_reward = 0

    episode_axis = []
    return_axis = []

    win_rate_per_episode = []
    loss_rate_per_episode = []
    tie_rate_per_episode = []


    for i_episode in count(1):
        saved_rewards = []
        saved_logprobs = []
        state = env.reset()
        done = False
        while not done:
            action, logprob = select_action(policy, state)
            state, status, done = env.play_against_random(action)
            reward = get_reward(status)
            saved_logprobs.append(logprob)
            saved_rewards.append(reward)

        R = compute_returns(saved_rewards)[0]
        running_reward += R

        finish_episode(saved_rewards, saved_logprobs, gamma)

        if i_episode % log_interval == 0:

            episode_axis.extend([i_episode])
            return_axis.extend([running_reward/log_interval])

            games_won, games_lost, games_tied, invalid_moves = play_games_against_random(policy, env)
            win_rate_per_episode.extend([games_won/100.0])
            loss_rate_per_episode.extend([games_lost/100.0])
            tie_rate_per_episode.extend([games_tied/100.0])

            if i_episode == log_interval:
                first_move_prob_distr = first_move_distr(policy,env).numpy()
            else:
                first_move_prob_distr = np.dstack((first_move_prob_distr, first_move_distr(policy,env).numpy()))

            print('Episode {}\tAverage return: {:.2f}\tGames Won: {}\tGames Lost:{}\tGames Tied:{}\tInvalid Moves:{}'.format(
                i_episode,
                running_reward / log_interval,
                games_won, 
                games_lost,
                games_tied,
                invalid_moves))
            running_reward = 0

        if i_episode % (log_interval) == 0:
            torch.save(policy.state_dict(),
                       "ttt/policy-%d.pkl" % i_episode)

        if i_episode % 1 == 0: # batch_size
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()

        if i_episode == 50000:
            #plot return
            plt.figure()
            plt.plot(episode_axis, return_axis)
            plt.xlabel("episode #")
            plt.ylabel("average return")
            plt.title("Training curve of Tic-Tac-Toe model")
            plt.savefig("figures/part5b_256.png")

            #plot win/loss rates
            plt.figure()
            plt.plot(episode_axis, win_rate_per_episode , label = "win rate")
            plt.plot(episode_axis, loss_rate_per_episode, label = "loss rate")
            plt.plot(episode_axis, tie_rate_per_episode, label = "tie rate")
            plt.xlabel("episode #")
            plt.ylabel("win/loss/tie rates")
            plt.title("Evolution of Win/Loss/Tie rates with training")
            plt.legend()
            plt.savefig("figures/part6.png")

            np.save("first_move_prob_distribution.npy", first_move_prob_distr)

            return

def first_move_distr(policy, env):
    """Display the distribution of first moves."""
    state = env.reset()
    state = torch.from_numpy(state).long().unsqueeze(0)
    state = torch.zeros(3,9).scatter_(0,state,1).view(1,27)
    pr = policy(Variable(state))
    return pr.data


def load_weights(policy, episode):
    """Load saved weights"""
    weights = torch.load("ttt/policy-%d.pkl" % episode)
    policy.load_state_dict(weights)

def play_games_against_random(policy, env, games = 100):
    """Play (100) games against random and return number of games won, lost or tied"""
    games_won, games_lost, games_tied, invalid_moves = 0, 0, 0, 0

    for i in range(games):
        state = env.reset()
        # if i % 19 == 0:
        #     print("Game: %s"%i)
        print("Game: %s"%i)
        done = False

        while not done:
            action, logprob = select_action(policy, state)
            state, status, done = env.play_against_random(action)
            invalid_moves += (1 if status == env.STATUS_INVALID_MOVE else 0)
            # if i % 19 == 0:
            #     env.render()
            env.render()

        if status == env.STATUS_WIN: games_won += 1
        elif status == env.STATUS_LOSE:
            games_lost += 1
            print("!!!!GAME %s LOST!!!!"%i)
        else:
            print("Game %s LOST"%i)
            games_tied += 1

    return games_won, games_lost, games_tied, invalid_moves

def part_7():
    prob_dist = np.load("first_move_prob_distribution.npy")
    episode_axis = np.arange(1000,51000,1000)

    #create 3x3 subplot
    fig = plt.figure(figsize=(10, 10))
    a = fig.add_subplot(3, 3, 1)
    plt.plot(episode_axis, prob_dist[0, 0, :])
    plt.title("cell 0")
    b = fig.add_subplot(3, 3, 2)
    plt.plot(episode_axis, prob_dist[0, 1, :])
    plt.title("cell 1")
    c = fig.add_subplot(3,3,3)
    plt.plot(episode_axis, prob_dist[0, 2, :])
    plt.title("cell 2")
    d = fig.add_subplot(3,3,4)
    plt.plot(episode_axis, prob_dist[0, 3, :])
    plt.title("cell 3")
    plt.ylabel("probability")
    e = fig.add_subplot(3,3,5)
    plt.plot(episode_axis, prob_dist[0, 4, :])
    plt.title("cell 4")
    f = fig.add_subplot(3,3,6)
    plt.plot(episode_axis,prob_dist[0, 5, :])
    plt.title("cell 5")
    g = fig.add_subplot(3,3,7)
    plt.plot(episode_axis,prob_dist[0, 6, :])
    plt.title("cell 6")
    h = fig.add_subplot(3,3,8)
    plt.plot(episode_axis,prob_dist[0, 7, :])
    plt.title("cell 7")
    plt.xlabel("episode")
    i = fig.add_subplot(3,3,9)
    plt.plot(episode_axis,prob_dist[0, 8, :])
    plt.title("cell 8")

    plt.savefig("figures/part7.png")

if __name__ == '__main__':


    import sys
    policy = Policy()
    env = Environment()

    if len(sys.argv) == 1:
        # `python tictactoe.py` to train the agent
        train(policy, env)
    else:
        # `python tictactoe.py <ep>` to print the first move distribution
        # using weightt checkpoint at episode int(<ep>)
        ep = int(sys.argv[1])
        load_weights(policy, ep)
        # print(first_move_distr(policy, env))
        print(play_games_against_random(policy, env))

    #part_7()
