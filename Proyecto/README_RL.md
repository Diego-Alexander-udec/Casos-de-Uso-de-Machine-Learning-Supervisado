# Implementación de Aprendizaje por Refuerzo con Q-Learning

## Descripción General

Este módulo implementa un agente de Aprendizaje por Refuerzo utilizando el algoritmo Q-Learning para resolver el problema de control CartPole-v1 de OpenAI Gymnasium.

## Contenido Teórico

### 1. Definición del Aprendizaje por Refuerzo

El Aprendizaje por Refuerzo (RL) es un paradigma de aprendizaje automático donde un agente aprende a tomar decisiones mediante la interacción con un entorno. A diferencia del aprendizaje supervisado (que requiere ejemplos etiquetados) y el no supervisado (que busca patrones ocultos), el RL aprende mediante prueba y error, recibiendo recompensas o penalizaciones por sus acciones.

**Diferencias principales:**
- **Supervisado:** Aprende de ejemplos con respuestas correctas predefinidas
- **No supervisado:** Descubre estructura en datos sin etiquetas
- **Por Refuerzo:** Aprende de la experiencia mediante recompensas y castigos

### 2. Componentes del Modelo RL

#### Agente
Sistema inteligente que toma decisiones. En nuestro caso, es el algoritmo Q-Learning que aprende a mantener el poste vertical.

#### Entorno
Mundo donde opera el agente. Usamos CartPole-v1, un sistema físico simulado donde un poste está balanceado sobre un carro móvil.

#### Estados
Situaciones en las que puede encontrarse el sistema:
- Posición del carro
- Velocidad del carro
- Ángulo del poste
- Velocidad angular del poste

#### Acciones
Decisiones que puede tomar el agente:
- Acción 0: Empujar carro hacia la izquierda
- Acción 1: Empujar carro hacia la derecha

#### Recompensas
Señales numéricas que evalúan las acciones:
- +1 por cada paso que el poste permanece vertical
- 0 cuando el episodio termina (fallo)

#### Política
Estrategia del agente: función que mapea estados a acciones. Es lo que el agente aprende a optimizar.

### 3. Principios del Ciclo de Aprendizaje

#### Exploración vs. Explotación
- **Exploración:** Probar acciones aleatorias para descubrir nuevas estrategias
- **Explotación:** Usar el conocimiento actual para maximizar recompensas
- **Balance:** Controlado por epsilon (ε) que decae con el tiempo

#### Retorno Acumulado
No solo importa la recompensa inmediata, sino la suma total de recompensas futuras:

```
G_t = R_{t+1} + R_{t+2} + R_{t+3} + ...
```

#### Descuento Temporal (γ)
Las recompensas futuras valen menos que las inmediatas:

```
G_t = R_{t+1} + γ*R_{t+2} + γ²*R_{t+3} + ...
```

Donde 0 ≤ γ ≤ 1. Un γ cercano a 1 valora más el futuro.

### 4. Algoritmos Principales

#### Q-Learning (Watkins & Dayan, 1992)
Algoritmo off-policy que aprende la función de valor-acción óptima:

```
Q(s,a) ← Q(s,a) + α[r + γ * max_a' Q(s',a') - Q(s,a)]
```

**Características:**
- No requiere modelo del entorno
- Converge a la política óptima
- Usa tabla para almacenar valores Q
- **Aplicaciones:** Robótica, navegación, juegos, control industrial

#### SARSA (State-Action-Reward-State-Action)
Algoritmo on-policy que actualiza basándose en la acción realmente tomada:

```
Q(s,a) ← Q(s,a) + α[r + γ * Q(s',a') - Q(s,a)]
```

**Características:**
- Más cauteloso que Q-Learning
- Mejor para entornos estocásticos
- Considera la política actual
- **Aplicaciones:** Sistemas con alto riesgo, control adaptativo

#### Deep Q-Network (DQN) (Mnih et al., 2015)
Combina Q-Learning con redes neuronales profundas:

**Características:**
- Usa redes neuronales en lugar de tablas
- Maneja espacios de estados muy grandes
- Experience replay para estabilidad
- Target network para reducir varianza
- **Aplicaciones:** Videojuegos complejos (Atari), sistemas de control avanzados, conducción autónoma

### 5. Buenas Prácticas

#### Estabilidad del Aprendizaje
- **Experience Replay:** Almacenar experiencias pasadas y reproducirlas aleatoriamente
- **Target Networks:** Usar una red separada para calcular objetivos (DQN)
- **Gradient Clipping:** Limitar magnitud de actualizaciones

#### Tasa de Exploración
```python
# Decaimiento de epsilon
epsilon = max(epsilon_min, epsilon * epsilon_decay)
```
- Iniciar con alta exploración (ε = 1.0)
- Reducir gradualmente (decay = 0.995)
- Mantener exploración mínima (ε_min = 0.01)

#### Manejo de Recompensas
- **Reward Shaping:** Diseñar recompensas intermedias que guíen al agente
- **Reward Clipping:** Normalizar recompensas para estabilidad
- **Evitar recompensas dispersas:** Proporcionar feedback frecuente

#### Convergencia
Indicadores de buen aprendizaje:
- Recompensas promedio crecientes
- Reducción de varianza en episodios tardíos
- Estabilización cerca del óptimo

#### Generalización
- Usar discretización apropiada (no muy fina ni muy gruesa)
- Validar en episodios diferentes a los de entrenamiento
- Considerar transfer learning para problemas similares

## Implementación Práctica

### Requisitos
```bash
pip install gymnasium numpy matplotlib pickle
```

### Estructura del Código

#### Clase QLearningAgent
```python
class QLearningAgent:
    def __init__(self, env, learning_rate=0.1, discount_factor=0.99, 
                 epsilon=1.0, epsilon_decay=0.995, epsilon_min=0.01, bins=10)
```

**Parámetros:**
- `learning_rate (α)`: Velocidad de aprendizaje (0.1 = 10% por actualización)
- `discount_factor (γ)`: Importancia del futuro (0.99 = muy importante)
- `epsilon (ε)`: Tasa de exploración inicial (1.0 = 100% aleatorio)
- `epsilon_decay`: Factor de reducción por episodio (0.995 = -0.5% cada vez)
- `epsilon_min`: Exploración mínima (0.01 = 1% siempre aleatorio)
- `bins`: Divisiones del espacio continuo (10 = 10,000 estados discretos)

### Entrenamiento

```bash
python Proyecto/rl_agent_cartpole.py
```

**Proceso de entrenamiento:**
1. Inicialización de la tabla Q (vacía)
2. Por cada episodio:
   - Resetear entorno
   - Mientras no termine:
     - Seleccionar acción (ε-greedy)
     - Ejecutar acción
     - Observar recompensa y nuevo estado
     - Actualizar tabla Q
   - Decaer epsilon
3. Guardar modelo y generar gráficas

**Salidas generadas:**
- `modelo_rl_cartpole.pkl`: Modelo entrenado
- `static/rl_training_rewards.png`: Evolución de recompensas
- `static/rl_training_distributions.png`: Histogramas de rendimiento

### Visualización en Flask

Ejecutar la aplicación web:
```bash
python app.py
```

Navegar a: `http://localhost:5000/caso_practico_refuerzo`

## Resultados Esperados

### Métricas de Éxito
- **Recompensa promedio > 450:** Excelente convergencia
- **Recompensa promedio 300-450:** Aprendizaje parcial
- **Recompensa promedio < 300:** Requiere más entrenamiento

### Interpretación de Gráficas

#### Gráfica de Recompensas
- **Fase inicial (0-200 episodios):** Alta variabilidad, exploración dominante
- **Fase intermedia (200-600):** Mejora gradual, inicio de explotación
- **Fase final (600-1000):** Estabilización, política óptima aprendida

#### Gráfica de Epsilon
- Decaimiento exponencial desde 1.0 hasta ~0.01
- Refleja la transición de exploración a explotación

## Referencias (APA 7)

Sutton, R. S., & Barto, A. G. (2018). *Reinforcement learning: An introduction* (2nd ed.). MIT Press. https://mitpress.mit.edu/9780262039246/reinforcement-learning/

Mnih, V., Kavukcuoglu, K., Silver, D., Rusu, A. A., Veness, J., Bellemare, M. G., Graves, A., Riedmiller, M., Fidjeland, A. K., Ostrovski, G., Petersen, S., Beattie, C., Sadik, A., Antonoglou, I., King, H., Kumaran, D., Wierstra, D., Legg, S., & Hassabis, D. (2015). Human-level control through deep reinforcement learning. *Nature*, 518(7540), 529–533. https://doi.org/10.1038/nature14236

Watkins, C. J. C. H., & Dayan, P. (1992). Q-learning. *Machine Learning*, 8(3), 279–292. https://doi.org/10.1007/BF00992698

## Autores

Proyecto de Machine Learning - Universidad de Cundinamarca
Noviembre 2025

## Licencia

Este proyecto es de uso educativo.
