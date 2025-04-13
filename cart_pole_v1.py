import matplotlib.pyplot as plt
import gymnasium as gym
from tqdm import tqdm
import numpy as np

class Policy:
    def __init__(self, weight_dims):
        """
        Initialize the weights randomly between [-1, 1].
        :param weight_dims:
        """
        self.weights = np.random.uniform(low=-1.0,high=1.0, size=weight_dims)

    def act(self, observation):
        """
        Determines the action to take based on a linear policy.
        :param observation: state[Cart Position, Cart Velocity, Pole Angle, Pole Angular Velocity]
        :return:   0: Push cart to the left.
                   1: Push cart to the right.
        """
        if np.dot(self.weights, observation) > 0:
            return 1
        else:
            return 0

    def set_random_weights(self):
        self.weights = np.random.uniform(low=-1.0,high=1.0, size=self.weights.shape)


def run_episode(policy, env, render=False):
    """
    Runs a single episode using the given policy.
    Returns the total accumulated reward (score).
    """
    state, _ = env.reset()
    total_reward = 0
    steps = 0
    done = False

    while not done and steps < 200:
        if render:
            env.render()

        action = policy.act(state)
        state, reward, terminated, truncated, _ = env.step(action)

        total_reward += reward
        done = terminated or truncated
        steps += 1

    return total_reward


def random_search(env, num_episodes=10_000):
    """
    Trains the agent using random search. Samples random weights and keeps
    the best one based on single-episode performance.
    """
    best_score = -float("inf")
    best_weights = None

    policy = Policy(weight_dims=4)

    for _ in tqdm(range(num_episodes), desc="Training"):
        policy.set_random_weights()
        score = run_episode(policy, env, render=False)

        if score > best_score:
            best_score = score
            best_weights = policy.weights.copy()

    best_policy = Policy(weight_dims=4)
    best_policy.weights = best_weights

    return best_policy, best_score


def random_search_trial(env, max_attempts=10_000):
    """
    Runs a single random search trial until a policy reaches a score of 200.
    Returns the number of episodes required.
    """
    episodes_count = 0
    while episodes_count < max_attempts:
        episodes_count += 1
        policy = Policy(weight_dims=4)
        policy.set_random_weights()
        score = run_episode(policy, env, render=False)
        if score >= 200:
            return episodes_count
    return episodes_count


def evaluate_random_search_trials(n_trials, env):
    """
    Evaluates the random search scheme by running n_trials independent trials.
    Returns a list of episode counts required in each trial to reach a score of 200.
    """
    episode_counts = []
    for _ in tqdm(range(n_trials), desc="Evaluating Trials"):
        count = random_search_trial(env)
        episode_counts.append(count)
    return episode_counts


def main():
    train_env = gym.make(id="CartPole-v1")
    eval_env = gym.make(id="CartPole-v1", render_mode="human")

    policy = Policy(weight_dims=4)

    """ Section 3 """
    score = run_episode(policy, eval_env, render=True)
    tqdm.write(f"Agent's score: {score}\n")

    """ Section 4 """
    best_policy, best_score = random_search(train_env, num_episodes=10_000)
    tqdm.write(f"Best training score: {best_score}")
    tqdm.write(f"Best weights: {best_policy.weights}")

    final_score = run_episode(best_policy, eval_env, render=True)
    tqdm.write(f"Final evaluation score: {final_score}")

    """ Section 5  average number of episodes ~ 14 """
    n_trials = 1000
    trial_counts = evaluate_random_search_trials(n_trials, train_env)
    average_episodes = np.mean(trial_counts)
    tqdm.write(f"Average number of episodes required to reach score 200: {average_episodes:.2f}")

    plt.figure(figsize=(10, 6))
    plt.hist(trial_counts, bins=30, edgecolor='black')
    plt.title("Histogram of Episodes Required to Reach a Score of 200")
    plt.xlabel("Number of Episodes")
    plt.ylabel("Frequency")
    plt.show()

    train_env.close()
    eval_env.close()

if __name__ == "__main__":
    main()
