"""
Implementación de Agente de Aprendizaje por Refuerzo usando Q-Learning
para el entorno CartPole de OpenAI Gym

Autor: Proyecto Machine Learning
Fecha: Noviembre 2025
"""

import numpy as np
import gymnasium as gym
import matplotlib.pyplot as plt
import pickle
import os
from collections import defaultdict


class QLearningAgent:
    """
    Agente de aprendizaje por refuerzo que implementa el algoritmo Q-Learning.
    
    Parámetros:
        env: Entorno de OpenAI Gym
        learning_rate (float): Tasa de aprendizaje (α) - controla cuánto se actualiza el valor Q
        discount_factor (float): Factor de descuento (γ) - importancia de recompensas futuras
        epsilon (float): Tasa de exploración inicial (ε) - probabilidad de acción aleatoria
        epsilon_decay (float): Tasa de decaimiento de epsilon por episodio
        epsilon_min (float): Valor mínimo de epsilon
        bins (int): Número de divisiones para discretizar el espacio de estados continuo
    """
    
    def __init__(self, env, learning_rate=0.1, discount_factor=0.99, 
                 epsilon=1.0, epsilon_decay=0.995, epsilon_min=0.01, bins=10):
        self.env = env
        self.lr = learning_rate  # α - learning rate
        self.gamma = discount_factor  # γ - discount factor
        self.epsilon = epsilon  # ε - exploration rate
        self.epsilon_decay = epsilon_decay
        self.epsilon_min = epsilon_min
        self.bins = bins
        
        # Tabla Q: mapea (estado, acción) -> valor Q
        self.q_table = defaultdict(lambda: np.zeros(env.action_space.n))
        
        # Límites del espacio de observación para discretización
        self.obs_space_bounds = list(zip(env.observation_space.low, 
                                          env.observation_space.high))
        # Ajustar límites infinitos
        self.obs_space_bounds[1] = (-4.0, 4.0)  # Velocidad del carro
        self.obs_space_bounds[3] = (-4.0, 4.0)  # Velocidad angular del poste
        
        # Historial de entrenamiento
        self.training_history = {
            'episode_rewards': [],
            'episode_lengths': [],
            'epsilon_values': [],
            'avg_rewards': []
        }
    
    def discretize_state(self, observation):
        """
        Convierte el estado continuo en un estado discreto.
        CartPole tiene 4 características continuas que necesitamos discretizar.
        """
        discrete_state = []
        for i, (obs, (low, high)) in enumerate(zip(observation, self.obs_space_bounds)):
            # Limitar valores a los límites
            obs = max(low, min(obs, high))
            # Crear bins y asignar el valor observado a un bin
            scaling = (obs - low) / (high - low)
            bin_number = int(scaling * (self.bins - 1))
            bin_number = min(self.bins - 1, max(0, bin_number))
            discrete_state.append(bin_number)
        
        return tuple(discrete_state)
    
    def select_action(self, state, training=True):
        """
        Selecciona una acción usando política epsilon-greedy.
        - Con probabilidad epsilon: acción aleatoria (exploración)
        - Con probabilidad 1-epsilon: mejor acción conocida (explotación)
        """
        if training and np.random.random() < self.epsilon:
            # Exploración: acción aleatoria
            return self.env.action_space.sample()
        else:
            # Explotación: mejor acción según tabla Q
            return np.argmax(self.q_table[state])
    
    def update_q_value(self, state, action, reward, next_state, done):
        """
        Actualiza el valor Q usando la ecuación de Bellman:
        Q(s,a) = Q(s,a) + α * [r + γ * max(Q(s',a')) - Q(s,a)]
        
        Donde:
        - s: estado actual
        - a: acción tomada
        - r: recompensa recibida
        - s': siguiente estado
        - α: learning rate
        - γ: discount factor
        """
        current_q = self.q_table[state][action]
        
        if done:
            # Si el episodio terminó, no hay valor futuro
            max_next_q = 0
        else:
            # Mejor valor Q posible en el siguiente estado
            max_next_q = np.max(self.q_table[next_state])
        
        # Calcular nuevo valor Q
        new_q = current_q + self.lr * (reward + self.gamma * max_next_q - current_q)
        
        # Actualizar tabla Q
        self.q_table[state][action] = new_q
    
    def train(self, num_episodes=1000, verbose=True):
        """
        Entrena el agente durante el número especificado de episodios.
        
        Args:
            num_episodes (int): Número de episodios de entrenamiento
            verbose (bool): Si mostrar progreso durante el entrenamiento
        
        Returns:
            dict: Historial de entrenamiento con métricas por episodio
        """
        print(f"\n{'='*60}")
        print(f"Iniciando entrenamiento del agente Q-Learning")
        print(f"{'='*60}")
        print(f"Parámetros:")
        print(f"  - Episodios: {num_episodes}")
        print(f"  - Learning rate (α): {self.lr}")
        print(f"  - Discount factor (γ): {self.gamma}")
        print(f"  - Epsilon inicial (ε): {self.epsilon}")
        print(f"  - Epsilon decay: {self.epsilon_decay}")
        print(f"  - Bins: {self.bins}")
        print(f"{'='*60}\n")
        
        for episode in range(num_episodes):
            # Reiniciar entorno
            observation, info = self.env.reset()
            state = self.discretize_state(observation)
            
            episode_reward = 0
            episode_length = 0
            done = False
            
            # Ejecutar episodio
            while not done:
                # Seleccionar y ejecutar acción
                action = self.select_action(state, training=True)
                next_observation, reward, terminated, truncated, info = self.env.step(action)
                done = terminated or truncated
                
                # Discretizar siguiente estado
                next_state = self.discretize_state(next_observation)
                
                # Actualizar tabla Q
                self.update_q_value(state, action, reward, next_state, done)
                
                # Acumular métricas
                episode_reward += reward
                episode_length += 1
                
                # Actualizar estado
                state = next_state
            
            # Decaimiento de epsilon (reducir exploración con el tiempo)
            if self.epsilon > self.epsilon_min:
                self.epsilon *= self.epsilon_decay
            
            # Guardar métricas del episodio
            self.training_history['episode_rewards'].append(episode_reward)
            self.training_history['episode_lengths'].append(episode_length)
            self.training_history['epsilon_values'].append(self.epsilon)
            
            # Calcular recompensa promedio de los últimos 100 episodios
            if episode >= 99:
                avg_reward = np.mean(self.training_history['episode_rewards'][-100:])
                self.training_history['avg_rewards'].append(avg_reward)
            else:
                self.training_history['avg_rewards'].append(
                    np.mean(self.training_history['episode_rewards'])
                )
            
            # Mostrar progreso
            if verbose and (episode + 1) % 100 == 0:
                avg_reward = self.training_history['avg_rewards'][-1]
                print(f"Episodio {episode + 1}/{num_episodes} | "
                      f"Recompensa promedio (últimos 100): {avg_reward:.2f} | "
                      f"Epsilon: {self.epsilon:.4f}")
        
        print(f"\n{'='*60}")
        print(f"Entrenamiento completado!")
        print(f"Recompensa promedio final (últimos 100 episodios): "
              f"{self.training_history['avg_rewards'][-1]:.2f}")
        print(f"{'='*60}\n")
        
        return self.training_history
    
    def evaluate(self, num_episodes=100, render=False):
        """
        Evalúa el agente entrenado sin exploración (epsilon = 0).
        
        Args:
            num_episodes (int): Número de episodios de evaluación
            render (bool): Si renderizar el entorno
        
        Returns:
            dict: Estadísticas de evaluación
        """
        print(f"\nEvaluando agente en {num_episodes} episodios...")
        
        rewards = []
        lengths = []
        
        for episode in range(num_episodes):
            observation, info = self.env.reset()
            state = self.discretize_state(observation)
            
            episode_reward = 0
            episode_length = 0
            done = False
            
            while not done:
                # Solo explotación (sin exploración)
                action = self.select_action(state, training=False)
                next_observation, reward, terminated, truncated, info = self.env.step(action)
                done = terminated or truncated
                
                state = self.discretize_state(next_observation)
                episode_reward += reward
                episode_length += 1
            
            rewards.append(episode_reward)
            lengths.append(episode_length)
        
        evaluation_stats = {
            'mean_reward': np.mean(rewards),
            'std_reward': np.std(rewards),
            'min_reward': np.min(rewards),
            'max_reward': np.max(rewards),
            'mean_length': np.mean(lengths)
        }
        
        print(f"Resultados de evaluación:")
        print(f"  - Recompensa promedio: {evaluation_stats['mean_reward']:.2f} ± "
              f"{evaluation_stats['std_reward']:.2f}")
        print(f"  - Recompensa mínima: {evaluation_stats['min_reward']:.2f}")
        print(f"  - Recompensa máxima: {evaluation_stats['max_reward']:.2f}")
        print(f"  - Longitud promedio: {evaluation_stats['mean_length']:.2f}")
        
        return evaluation_stats
    
    def save_model(self, filepath='modelo_rl_cartpole.pkl'):
        """Guarda la tabla Q y los parámetros del agente."""
        model_data = {
            'q_table': dict(self.q_table),
            'learning_rate': self.lr,
            'discount_factor': self.gamma,
            'bins': self.bins,
            'obs_space_bounds': self.obs_space_bounds,
            'training_history': self.training_history
        }
        
        with open(filepath, 'wb') as f:
            pickle.dump(model_data, f)
        
        print(f"\nModelo guardado en: {filepath}")
    
    def load_model(self, filepath='modelo_rl_cartpole.pkl'):
        """Carga la tabla Q y los parámetros del agente."""
        with open(filepath, 'rb') as f:
            model_data = pickle.load(f)
        
        self.q_table = defaultdict(lambda: np.zeros(self.env.action_space.n), 
                                   model_data['q_table'])
        self.lr = model_data['learning_rate']
        self.gamma = model_data['discount_factor']
        self.bins = model_data['bins']
        self.obs_space_bounds = model_data['obs_space_bounds']
        self.training_history = model_data['training_history']
        
        print(f"\nModelo cargado desde: {filepath}")


def plot_training_results(training_history, save_path='static/'):
    """
    Genera visualizaciones del proceso de entrenamiento.
    
    Args:
        training_history (dict): Historial de entrenamiento del agente
        save_path (str): Ruta donde guardar las gráficas
    """
    os.makedirs(save_path, exist_ok=True)
    
    # Configurar estilo
    plt.style.use('seaborn-v0_8-darkgrid')
    
    # Figura 1: Recompensas por episodio y promedio móvil
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10))
    
    episodes = range(1, len(training_history['episode_rewards']) + 1)
    
    # Subplot 1: Recompensas
    ax1.plot(episodes, training_history['episode_rewards'], 
             alpha=0.3, color='blue', label='Recompensa por episodio')
    ax1.plot(episodes, training_history['avg_rewards'], 
             color='red', linewidth=2, label='Promedio móvil (100 episodios)')
    ax1.set_xlabel('Episodio', fontsize=12)
    ax1.set_ylabel('Recompensa acumulada', fontsize=12)
    ax1.set_title('Evolución de la Recompensa durante el Entrenamiento', 
                  fontsize=14, fontweight='bold')
    ax1.legend(loc='lower right')
    ax1.grid(True, alpha=0.3)
    
    # Subplot 2: Tasa de exploración (epsilon)
    ax2.plot(episodes, training_history['epsilon_values'], 
             color='green', linewidth=2)
    ax2.set_xlabel('Episodio', fontsize=12)
    ax2.set_ylabel('Epsilon (ε)', fontsize=12)
    ax2.set_title('Decaimiento de la Tasa de Exploración (Epsilon)', 
                  fontsize=14, fontweight='bold')
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(os.path.join(save_path, 'rl_training_rewards.png'), 
                dpi=300, bbox_inches='tight')
    print(f"Gráfica de recompensas guardada en: {save_path}rl_training_rewards.png")
    plt.close()
    
    # Figura 2: Distribución de recompensas y longitudes de episodios
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    
    # Histograma de recompensas
    ax1.hist(training_history['episode_rewards'], bins=30, 
             color='skyblue', edgecolor='black', alpha=0.7)
    ax1.axvline(np.mean(training_history['episode_rewards']), 
                color='red', linestyle='--', linewidth=2, 
                label=f'Media: {np.mean(training_history["episode_rewards"]):.2f}')
    ax1.set_xlabel('Recompensa acumulada', fontsize=12)
    ax1.set_ylabel('Frecuencia', fontsize=12)
    ax1.set_title('Distribución de Recompensas', fontsize=14, fontweight='bold')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Histograma de longitudes
    ax2.hist(training_history['episode_lengths'], bins=30, 
             color='lightcoral', edgecolor='black', alpha=0.7)
    ax2.axvline(np.mean(training_history['episode_lengths']), 
                color='red', linestyle='--', linewidth=2, 
                label=f'Media: {np.mean(training_history["episode_lengths"]):.2f}')
    ax2.set_xlabel('Longitud del episodio (pasos)', fontsize=12)
    ax2.set_ylabel('Frecuencia', fontsize=12)
    ax2.set_title('Distribución de Longitudes de Episodios', 
                  fontsize=14, fontweight='bold')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(os.path.join(save_path, 'rl_training_distributions.png'), 
                dpi=300, bbox_inches='tight')
    print(f"Gráfica de distribuciones guardada en: {save_path}rl_training_distributions.png")
    plt.close()


def main():
    """
    Función principal para entrenar y evaluar el agente Q-Learning.
    """
    # Crear entorno CartPole
    env = gym.make('CartPole-v1')
    
    # Crear agente con parámetros optimizados
    agent = QLearningAgent(
        env=env,
        learning_rate=0.1,      # α - qué tan rápido aprende
        discount_factor=0.99,   # γ - importancia del futuro
        epsilon=1.0,            # ε - exploración inicial (100%)
        epsilon_decay=0.995,    # decaimiento de exploración
        epsilon_min=0.01,       # exploración mínima (1%)
        bins=10                 # discretización del espacio de estados
    )
    
    # Entrenar agente
    training_history = agent.train(num_episodes=1000, verbose=True)
    
    # Evaluar agente
    evaluation_stats = agent.evaluate(num_episodes=100)
    
    # Generar visualizaciones
    plot_training_results(training_history, save_path='static/')
    
    # Guardar modelo
    agent.save_model('modelo_rl_cartpole.pkl')
    
    # Cerrar entorno
    env.close()
    
    print("\n¡Proceso completado exitosamente!")
    print(f"El agente ha sido entrenado y los resultados han sido guardados.")


if __name__ == '__main__':
    main()
