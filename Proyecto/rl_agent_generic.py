"""
Agente Q-Learning genérico con soporte para espacios de estados Discrete y Box.
Incluye utilidades para entrenamiento, evaluación y generación de gráficas en memoria.
"""
from __future__ import annotations

import numpy as np
import gymnasium as gym
import io
import os
import pickle
from collections import defaultdict
import matplotlib.pyplot as plt
import base64


class QLearningAgent:
    def __init__(
        self,
        env: gym.Env,
        learning_rate: float = 0.1,
        discount_factor: float = 0.99,
        epsilon: float = 1.0,
        epsilon_decay: float = 0.995,
        epsilon_min: float = 0.01,
        bins: int = 10,
    ) -> None:
        self.env = env
        self.lr = learning_rate
        self.gamma = discount_factor
        self.epsilon = epsilon
        self.epsilon_decay = epsilon_decay
        self.epsilon_min = epsilon_min
        self.bins = bins

        # Detectar tipo de espacio de observación
        self.is_discrete_obs = hasattr(env.observation_space, "n")
        if self.is_discrete_obs:
            # Espacio Discrete: estados son enteros [0..n-1]
            self.n_states = env.observation_space.n
            self.obs_space_bounds = None
        else:
            # Espacio Box: discretizar cada dimensión
            low = np.array(env.observation_space.low, dtype=np.float32)
            high = np.array(env.observation_space.high, dtype=np.float32)
            # Reemplazar infinitos por límites razonables
            low = np.where(np.isfinite(low), low, -4.0)
            high = np.where(np.isfinite(high), high, 4.0)
            self.obs_space_bounds = list(zip(low, high))
            self.n_states = None  # desconocido: usamos tuplas discretizadas

        self.n_actions = env.action_space.n
        self.q_table = defaultdict(lambda: np.zeros(self.n_actions, dtype=np.float32))

        self.history = {
            "episode_rewards": [],
            "episode_lengths": [],
            "epsilon_values": [],
            "avg_rewards": [],
        }

    def _discretize_box(self, observation: np.ndarray) -> tuple[int, ...]:
        discrete_state: list[int] = []
        for obs, (low, high) in zip(observation, self.obs_space_bounds):
            obs = float(np.clip(obs, low, high))
            scaling = (obs - low) / (high - low + 1e-8)
            bin_number = int(scaling * (self.bins - 1))
            bin_number = min(self.bins - 1, max(0, bin_number))
            discrete_state.append(bin_number)
        return tuple(discrete_state)

    def discretize_state(self, observation) -> int | tuple[int, ...]:
        if self.is_discrete_obs:
            # observation puede venir como int o ndarray([int])
            if isinstance(observation, (int, np.integer)):
                return int(observation)
            if isinstance(observation, (list, tuple)) and len(observation) == 1:
                return int(observation[0])
            try:
                return int(np.array(observation).item())
            except Exception:
                return int(observation)
        else:
            return self._discretize_box(np.array(observation, dtype=np.float32))

    def select_action(self, state, training: bool = True) -> int:
        if training and np.random.random() < self.epsilon:
            return self.env.action_space.sample()
        return int(np.argmax(self.q_table[state]))

    def update_q(self, state, action: int, reward: float, next_state, done: bool) -> None:
        current_q = self.q_table[state][action]
        max_next_q = 0.0 if done else float(np.max(self.q_table[next_state]))
        new_q = current_q + self.lr * (reward + self.gamma * max_next_q - current_q)
        self.q_table[state][action] = new_q

    def train(self, episodes: int = 500, verbose: bool = True) -> dict:
        for ep in range(episodes):
            obs, _ = self.env.reset()
            state = self.discretize_state(obs)
            done = False
            ep_reward = 0.0
            ep_len = 0

            while not done:
                action = self.select_action(state, training=True)
                next_obs, reward, terminated, truncated, _ = self.env.step(action)
                done = bool(terminated or truncated)
                next_state = self.discretize_state(next_obs)
                self.update_q(state, action, float(reward), next_state, done)

                ep_reward += float(reward)
                ep_len += 1
                state = next_state

            if self.epsilon > self.epsilon_min:
                self.epsilon *= self.epsilon_decay

            self.history["episode_rewards"].append(ep_reward)
            self.history["episode_lengths"].append(ep_len)
            self.history["epsilon_values"].append(self.epsilon)
            if ep >= 99:
                avg_r = float(np.mean(self.history["episode_rewards"][-100:]))
            else:
                avg_r = float(np.mean(self.history["episode_rewards"]))
            self.history["avg_rewards"].append(avg_r)

            if verbose and (ep + 1) % max(1, episodes // 5) == 0:
                print(
                    f"Episodio {ep+1}/{episodes} | Recompensa media100: {avg_r:.2f} | "
                    f"Epsilon: {self.epsilon:.3f}"
                )
        return self.history

    def evaluate(self, episodes: int = 50) -> dict:
        rewards = []
        lengths = []
        for _ in range(episodes):
            obs, _ = self.env.reset()
            state = self.discretize_state(obs)
            done = False
            ep_reward = 0.0
            ep_len = 0
            while not done:
                action = int(np.argmax(self.q_table[state]))
                next_obs, reward, terminated, truncated, _ = self.env.step(action)
                done = bool(terminated or truncated)
                state = self.discretize_state(next_obs)
                ep_reward += float(reward)
                ep_len += 1
            rewards.append(ep_reward)
            lengths.append(ep_len)
        return {
            "mean_reward": float(np.mean(rewards)),
            "std_reward": float(np.std(rewards)),
            "min_reward": float(np.min(rewards)),
            "max_reward": float(np.max(rewards)),
            "mean_length": float(np.mean(lengths)),
        }

    def save(self, path: str) -> None:
        os.makedirs(os.path.dirname(path), exist_ok=True)
        data = {
            "q_table": dict(self.q_table),
            "lr": self.lr,
            "gamma": self.gamma,
            "epsilon": self.epsilon,
            "epsilon_decay": self.epsilon_decay,
            "epsilon_min": self.epsilon_min,
            "bins": self.bins,
            "is_discrete_obs": self.is_discrete_obs,
            "obs_space_bounds": self.obs_space_bounds,
            "history": self.history,
        }
        with open(path, "wb") as f:
            pickle.dump(data, f)


def _plot_training(history: dict) -> tuple[str, str]:
    plt.style.use('seaborn-v0_8-darkgrid')

    # Fig 1: Recompensas y epsilon
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8))
    episodes = np.arange(1, len(history['episode_rewards']) + 1)
    ax1.plot(episodes, history['episode_rewards'], alpha=0.3, label='Recompensa episodio')
    ax1.plot(episodes, history['avg_rewards'], color='red', linewidth=2, label='Promedio móvil (100)')
    ax1.set_xlabel('Episodio'); ax1.set_ylabel('Recompensa acumulada'); ax1.legend(); ax1.grid(True, alpha=0.3)

    ax2.plot(episodes, history['epsilon_values'], color='green', linewidth=2)
    ax2.set_xlabel('Episodio'); ax2.set_ylabel('Epsilon (ε)'); ax2.grid(True, alpha=0.3)
    buf1 = io.BytesIO()
    plt.tight_layout(); fig.savefig(buf1, format='png', dpi=200, bbox_inches='tight')
    plt.close(fig)

    # Fig 2: Distribuciones
    fig2, (bx1, bx2) = plt.subplots(1, 2, figsize=(12, 5))
    bx1.hist(history['episode_rewards'], bins=30, color='skyblue', edgecolor='black')
    bx1.set_title('Distribución de Recompensas'); bx1.set_xlabel('Recompensa'); bx1.set_ylabel('Frecuencia'); bx1.grid(True, alpha=0.3)

    bx2.hist(history['episode_lengths'], bins=30, color='lightcoral', edgecolor='black')
    bx2.set_title('Distribución de Longitudes'); bx2.set_xlabel('Pasos'); bx2.set_ylabel('Frecuencia'); bx2.grid(True, alpha=0.3)
    buf2 = io.BytesIO()
    plt.tight_layout(); fig2.savefig(buf2, format='png', dpi=200, bbox_inches='tight')
    plt.close(fig2)

    img1 = base64.b64encode(buf1.getvalue()).decode()
    img2 = base64.b64encode(buf2.getvalue()).decode()
    return img1, img2


def train_qlearning(
    env_id: str,
    episodes: int = 500,
    learning_rate: float = 0.1,
    discount_factor: float = 0.99,
    epsilon: float = 1.0,
    epsilon_decay: float = 0.995,
    epsilon_min: float = 0.01,
    bins: int = 10,
) -> dict:
    env = gym.make(env_id)
    agent = QLearningAgent(
        env,
        learning_rate=learning_rate,
        discount_factor=discount_factor,
        epsilon=epsilon,
        epsilon_decay=epsilon_decay,
        epsilon_min=epsilon_min,
        bins=bins,
    )
    history = agent.train(episodes=episodes, verbose=True)
    eval_stats = agent.evaluate(episodes=max(10, episodes // 10))

    # Guardar modelo
    model_dir = os.path.join(os.path.dirname(__file__), 'modelos')
    os.makedirs(model_dir, exist_ok=True)
    model_path = os.path.join(model_dir, f'{env_id.replace("-","_")}_qlearning.pkl')
    agent.save(model_path)

    rewards_plot, distributions_plot = _plot_training(history)

    return {
        "history": history,
        "eval": eval_stats,
        "model_path": model_path,
        "rewards_plot": rewards_plot,
        "distributions_plot": distributions_plot,
        "q_table_size": len(agent.q_table),
        "params": {
            "episodes": episodes,
            "learning_rate": learning_rate,
            "discount_factor": discount_factor,
            "epsilon": epsilon,
            "epsilon_decay": epsilon_decay,
            "epsilon_min": epsilon_min,
            "bins": bins,
            "env_id": env_id,
        },
    }
