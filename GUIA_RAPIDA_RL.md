# Guía Rápida: Aprendizaje por Refuerzo - Q-Learning

## ⚡ Inicio Rápido (5 minutos)

### Paso 1: Instalar Dependencias
```powershell
cd e:\Github\Casos-de-Uso-de-Machine-Learning-Supervisado
pip install gymnasium numpy matplotlib Flask
```

### Paso 2: Entrenar el Agente
```powershell
python Proyecto\rl_agent_cartpole.py
```

**Salida esperada:**
```
====================================================================
Iniciando entrenamiento del agente Q-Learning
====================================================================
Parámetros:
  - Episodios: 1000
  - Learning rate (α): 0.1
  - Discount factor (γ): 0.99
  - Epsilon inicial (ε): 1.0
  ...

Episodio 100/1000 | Recompensa promedio (últimos 100): 23.45 | Epsilon: 0.6050
Episodio 200/1000 | Recompensa promedio (últimos 100): 87.32 | Epsilon: 0.3660
...
Episodio 1000/1000 | Recompensa promedio (últimos 100): 487.25 | Epsilon: 0.0100

====================================================================
Entrenamiento completado!
====================================================================
```

### Paso 3: Ver Resultados en la Web
```powershell
python app.py
```

Abrir navegador: **http://localhost:5000/caso_practico_refuerzo**

---

## 📋 Verificación Rápida

### ✅ Archivos que deberían generarse:
- [ ] `Proyecto/modelo_rl_cartpole.pkl` (~100KB)
- [ ] `Proyecto/static/rl_training_rewards.png`
- [ ] `Proyecto/static/rl_training_distributions.png`

### ✅ Métricas de éxito:
- Recompensa promedio final: **> 450** ✅
- Estados explorados: **~2000-3000**
- Tiempo de entrenamiento: **2-5 minutos**

---

## 🎯 Navegación en la Web

1. **Conceptos Básicos:** `/conceptos_refuerzo`
   - Teoría del Aprendizaje por Refuerzo
   - Componentes del sistema
   - Algoritmos principales
   - Referencias académicas (APA 7)

2. **Caso Práctico:** `/caso_practico_refuerzo`
   - Descripción del entorno CartPole
   - Parámetros del algoritmo
   - Resultados del entrenamiento
   - Gráficas interactivas
   - Análisis de convergencia

---

## 🔧 Solución de Problemas

### Problema: "gymnasium" no encontrado
```powershell
pip install gymnasium
```

### Problema: Error de matplotlib
```powershell
pip install --upgrade matplotlib
```

### Problema: Puerto 5000 ocupado
En `app.py`, cambiar última línea:
```python
app.run(debug=True, port=5001)
```

---

## 📖 Documentación Completa

Para más detalles, consultar:
- **`Proyecto/README_RL.md`** - Documentación técnica completa
- **`RESUMEN_RL.md`** - Resumen del trabajo realizado

---

## 🎓 Conceptos Clave Implementados

| Concepto | Implementación |
|----------|----------------|
| **Q-Learning** | Tabla Q con actualización de Bellman |
| **ε-greedy** | Exploración decreciente (1.0 → 0.01) |
| **Discretización** | 10 bins × 4 dimensiones = 10K estados |
| **Visualización** | Matplotlib + Bootstrap 5 |
| **Persistencia** | Pickle para guardar modelo |

---

## 💡 Próximos Pasos Sugeridos

1. **Experimentar con parámetros:**
   - Cambiar `learning_rate` (0.05, 0.2)
   - Ajustar `bins` (5, 15, 20)
   - Modificar `epsilon_decay` (0.99, 0.995, 0.999)

2. **Probar otros entornos:**
   - MountainCar-v0
   - Acrobot-v1
   - LunarLander-v2

3. **Implementar mejoras:**
   - SARSA algorithm
   - Experience Replay
   - Deep Q-Network (DQN)

---

**¡Listo para empezar! 🚀**
