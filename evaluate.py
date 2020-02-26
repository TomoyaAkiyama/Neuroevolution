import os

import torch
import numpy as np

from models.deterministic_policy import DeterministicPolicy
from env_wrapper import EnvWrapper


def main(env_name, seed, individual, args, eval_episodes=10):
    env = EnvWrapper(env_name)

    state_dim = sum(list(env.unwrapped().observation_space.shape))
    action_dim = sum(list(env.unwrapped().action_space.shape))
    hidden_sizes = args['hidden_sizes']
    activation = args['activation']
    layernorm = args['layernorm']

    torch.manual_seed(seed)
    np.random.seed(seed)
    env.seed(seed)

    policy = DeterministicPolicy(state_dim, action_dim, hidden_sizes, -1, activation, layernorm).eval()

    file_dir = os.path.abspath(os.path.dirname(__file__))
    save_dir = os.path.join(
        file_dir,
        'results',
        env_name,
        args['activation'] + ('_LayerNorm' if args['layernorm'] else ''),
        'seed' + str(seed),
    )
    model_path = os.path.join(
        save_dir,
        'learned_model',
        'individual' + str(individual) + '.pth'
    )
    model_state_dict = torch.load(model_path)
    policy.load_state_dict(model_state_dict)

    env.seed(seed + 100)

    episode_rewards = []
    for _ in range(eval_episodes):
        state = env.reset()
        done = False
        sum_rewards = 0
        while not done:
            # env.render()
            action = policy.deterministic_action(torch.tensor(state.reshape(1, -1), dtype=torch.float))
            next_state, reward, done, _ = env.step(action)
            sum_rewards += reward
            state = next_state
        episode_rewards.append(sum_rewards)
        print(f'Episode: {len(episode_rewards)} Sum Rewards: {sum_rewards:.3f}')

    avg_reward = np.mean(episode_rewards)
    print('\n---------------------------------------')
    print(f'Evaluation over {eval_episodes} episodes: {avg_reward:.3f}')
    print('---------------------------------------')


if __name__ == '__main__':
    env_name = 'Swimmer-v2'
    seed = 1
    individual = 1

    args = {
        'hidden_sizes': [400, 300],
        'activation': 'ReLU',
        'layernorm': True,
        'pop_size': 10,
        'elite_fraction': 0.2,
        'cross_prob': 0.0,
        'cross_fraction': 0.3,
        'bias_cross_prob': 0.2,
        'mutation_prob': 0.9,
        'mut_strength': 0.1,
        'mut_fraction': 0.1,
        'super_mut_prob': 0.05,
        'reset_prob': 0.1,
    }

    main(env_name, seed, individual, args)
