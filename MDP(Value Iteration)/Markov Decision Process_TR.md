<h1 align='center'>Markov Decision Process</h1>

<h2 align='center'>（2024学年秋季学期）</h2>

**课程名称：强化学习与博弈论                   			             批改人：**

| 实验      | MDP                     | 专业（方向） | 计算机科学与技术 |
| --------- | ----------------------- | ------------ | ---------------- |
| **学号**  | **21307244**            | **姓名**     | **钟宇**         |
| **Email** | **tue2dayzz@gmail.com** | **完成日期** | **2024.12.24**   |

[toc]

## 一、实验内容

Solve the Maze Problem using Policy Iteration or Value Iteration

<img src="/Users/Caleb/Library/Application Support/typora-user-images/image-20250103122213618.png" alt="image-20250103122213618" style="zoom:67%;" />



## 二、实验环境

- 操作系统：$\text{Ubuntu 22.04}$
- 编程语言：$\text{Python}$
- IDE：Pycharm CE



## 三、实验过程

### 实验方法

本次实验采用 **Value Iteration** 的方法进行迷宫问题的解决



### 算法简述

#### Markov Decision Process

马尔可夫决策过程 (Markov Decision Process, MDP) 是一种用于序列决策的数学模型，广泛应用于强化学习、机器人控制、资源管理等领域。MDP 的核心思想是通过定义状态、动作、奖励和转移概率，来建模智能体与环境之间的交互，并寻找最优策略以最大化累积奖励。

1. **MDP 基本要素**：一个 MDP 通常由以下五元组定义：
   - **状态空间 (S)：**表示智能体可能处于的所有状态的集合。
   - **动作空间 (A)：**表示智能体在每个状态下可以采取的所有动作的集合。
   - **转移概率 (P)：**表示智能体在状态 s 下采取动作 a 后转移到状态 s' 的概率，记为 P(s'|s, a)。
   - **奖励函数 (R)：**表示智能体在状态 s 下采取动作 a 后转移到状态 s' 所获得的即时奖励，记为 R(s, a, s')。
   - **折扣因子 (γ)：**用于权衡当前奖励和未来奖励的重要性，取值范围为 [0, 1]。

2. **策略与价值函数**
   - **策略 (π)：**定义了智能体在每个状态下选择动作的方式，可以是确定性的，也可以是随机性的。
   - **状态价值函数 (V<sub>π</sub>(s))：**表示从状态 s 开始，遵循策略 π 所能获得的期望累积奖励。
   - **动作价值函数 (Q<sub>π</sub>(s, a))：**表示在状态 s 下采取动作 a 后，遵循策略 π 所能获得的期望累积奖励。

3. **最优策略与贝尔曼方程**

   MDP 的目标是找到最优策略 π\*，使得在任意状态下，遵循该策略都能获得最大的期望累积奖励。最优策略对应的状态价值函数和动作价值函数分别记为 V* 和 Q*。

   贝尔曼方程是 MDP 的核心方程，它描述了状态价值函数和动作价值函数之间的关系：

   - **贝尔曼期望方程:**
     $$
     \begin{align}
     V_\pi(s) &= \sum_{a\in A}\pi(a|s)\sum_{s' \in S}P(s'|s,a)[R(s,a,s')+\gamma V_\pi(s')]\\
     Q_\pi(s,a) &= \sum_{s' \in S}P(s'|s,a)[R(s,a,s')+\gamma V_\pi(s')] 
     \end{align}
     $$

   - **贝尔曼最优方程:**
     $$
     \begin{align}
     V^*(s) &= max_{a \in A}Q^*(s,a)\\
     Q^*(s,a) &= \sum_{s' \in S}P(s'|s,a)[R(s,a,s')+\gamma V^*(s')] 
     \end{align}
     $$

#### Value Iterate

值迭代是一种基于动态规划的 MDP 求解算法，用于寻找最优策略。其核心思想是通过迭代更新状态价值函数，直到收敛到最优价值函数，进而推导出最优策略。值迭代的特点是直接对价值函数进行优化，而不需要显式地维护策略。

**算法原理**

值迭代基于**贝尔曼最优方程**，通过不断迭代更新状态价值函数 V(s)，使其逐步逼近最优价值函数 V*(s)。具体更新公式如下：
$$
V_{k+1}(s)=\mathop\max_{a \in A}\sum_{s' \in S}P(s'|s,a)[R(s,a,s')+\gamma V_k(s')]
$$
其中：

- V<sub>k</sub>(s) 是第 k 次迭代时状态 s 的价值函数估计值。
- γ 是折扣因子，用于平衡当前奖励和未来奖励的重要性。
- P(s'|s, a) 是状态转移概率。
- R(s, a, s') 是即时奖励函数。

每次迭代中，算法会遍历所有状态 s∈S，并计算在当前状态下所有可能动作 a∈A 的期望价值，然后选择使期望价值最大化的动作来更新 V(s)。



**算法流程**

值迭代算法的具体步骤如下：

1. **初始化：**

   - 对所有状态 s∈S，初始化价值函数 V<sub>0</sub>(s)=0（或其他任意值）。
   - 设置收敛阈值 θ（一个较小的正数），用于判断算法是否收敛。

2. **迭代更新：**

   对于每次迭代 k：

   - 对所有状态 s∈S：计算 $V_{k+1}(s)=\mathop{\max}_{a \in A}\sum_{s' \in S}P(s'|s,a)[R(s,a,s')+\gamma V_k(s')]$
   - 检查收敛性：如果 $max_{s \in S}|V_{k+1}(s)-V_k(s)| < \theta$，则停止迭代。

3. **输出最优策略：**

   根据收敛后的最优价值函数 V\*，计算最优策略 π*：
   $$
   \pi^*(s)=arg\mathop\max_{a \in A}\sum_{s' \in S}P(s'|s,a)[R(s,a,s')+\gamma V^*(s')]
   $$



### 代码实现

#### MDP Modeling

```python
# Modeling maze
self.maze = [
    [0, 0, 0, 0, 0, 0, 0, 0],
    [0, 1, 1, 1, 1, 1, 1, 0],
    [1, 1, 0, 0, 1, 0, 1, 0],
    [0, 1, 1, 0, 0, 1, 1, 0],
    [0, 0, 1, 1, 0, 1, 0, 0],
    [0, 1, 0, 1, 0, 1, 1, 0],
    [0, 1, 1, 1, 1, 0, 1, 1],
    [0, 0, 0, 0, 0, 0, 0, 0]
]
self.start = (2, 0)
self.end = (6, 7)
```

本次实验是使用值迭代的方法解决迷宫问题，因此首先我们需要对于迷宫进行建模：

实验使用的迷宫使用用二维的 $0-1$ 矩阵进行建模，矩阵上的每一个位置坐标对应 MDP 问题中的一个状态

- 0 表示障碍物（不可通过）
- 1 表示可通行的路径
- `start` 为起点坐标 `(2, 0)`
- `end` 为终点坐标 `(6, 7)`



```python
# Parameter settings
self.gamma = 0.9  # Discount factor
self.reward = -1  # Reward for each step
self.threshold = 0.0001  # Convergence threshold
```

模型的参数设置如下：

- **折扣因子 (`gamma`)：**用于权衡当前奖励和未来奖励的重要性，取值范围为 `[0, 1]`。`gamma` 越接近 1，表示未来奖励越重要。
- **每一步的奖励 (`reward`)：**每走一步的即时奖励为 `-1`
- **收敛阈值 (`threshold`)：**用于判断值迭代是否收敛。当价值函数的变化小于该阈值时，停止迭代



```python
# Action definitions: up, down, left, right
self.actions = [(-1, 0), (1, 0), (0, -1), (0, 1)]
```

同时，根据题目的要求，定义了智能体可以采取的四种动作：

- 上 `(-1, 0)`
- 下 `(1, 0)`
- 左 `(0, -1)`
- 右 `(0, 1)`



```python
# Value matrix V
self.V = np.full((len(self.maze), len(self.maze[0])), -np.inf)
self.initialize()

def initialize(self):
    for i in range(len(self.maze)):
        for j in range(len(self.maze[0])):
            if self.maze[i][j] == 1:
                self.V[i][j] = 0
```

初始化模型的**价值矩阵 V**，用于存储每个状态的价值函数。

初始时，所有可通行位置的价值设为 `0`，障碍物的价值设为 `-∞`（表示不可达）。



#### Value Iteration

根据算法简述部分关于 Value Iteration 算法流程的介绍，转换为实际代码如下：

```python
def is_valid(self, state):
    x, y = state
    return 0 <= x < len(self.maze) and 0 <= y < len(self.maze[0]) and self.maze[x][y] == 1

def get_next_state(self, state, action):
    return (state[0] + action[0], state[1] + action[1])

def value_iteration(self):
    while True:
        delta = 0
        for i in range(len(self.maze)):
            for j in range(len(self.maze[0])):
                if (i, j) == self.end or self.maze[i][j] == 0:
                    continue
                # Calculate the maximum value for the current state
                max_value = float('-inf')
                for action in self.actions:
                    next_state = self.get_next_state((i, j), action)
                    if self.is_valid(next_state):
                        max_value = max(max_value, self.reward + self.gamma * self.V[next_state])

                # Update the value of the current state
                delta = max(delta, abs(self.V[i][j] - max_value))
                self.V[i][j] = max_value
        if delta < self.threshold:
            break
```

- `is_valid` 函数检查给定的状态 `(x, y)` 是否在迷宫范围内且是可通行的
  - 如果状态有效，返回 `True`
  - 否则返回 `False`
- `get_next_state` 函数则根据当前状态 `state` 和动作 `action`，计算下一个状态
- `value_iteration` 函数为值迭代算法逻辑实现的主要函数，其通过值迭代更新价值矩阵 `V`，直到收敛，具体来说，函数
  - 遍历所有状态 `(i, j)`。
  - 对于每个状态，计算所有可能动作的期望价值，并选择最大值作为当前状态的新价值。
  - 更新价值矩阵 `V`，并记录最大变化值 `delta`。
  - 当 `delta` 小于收敛阈值 `threshold` 时，停止迭代。



#### Find Path 

在迭代完价值矩阵 V 之后，我们遍可以通过价值矩阵中每一个状态对应的价值来构建迷宫的路径

```python
def find_path(self):
    path = [self.start]
    state = self.start
    while state != self.end:
        max_value = float('-inf')
        next_state = None
        for action in self.actions:
            new_state = self.get_next_state(state, action)
            if self.is_valid(new_state) and self.reward + self.gamma * self.V[new_state] > max_value:
                max_value = self.reward + self.gamma * self.V[new_state]
                next_state = new_state
        if next_state is None:
            print("No valid path found!")
            return []
        path.append(next_state)
        state = next_state
    return path
```

根据收敛后的价值矩阵 `V`，从起点到终点选择最大价值的动作，生成路径，具体步骤如下：

1. 从起点开始，选择使 `reward + gamma * V[next_state]` 最大的动作。
2. 将下一个状态加入路径，并更新当前状态。
3. 重复上述步骤，直到到达终点。



#### Visualize Path

```python
def visualize_path(self, path):
    # Visualize the maze and the path
    fig, ax = plt.subplots()

    # Draw the maze
    maze_array = np.array(self.maze)
    ax.imshow(maze_array, cmap='binary', interpolation='nearest')

    # Mark the start and end points
    ax.plot(self.start[1], self.start[0], 'go', markersize=10, label='Start')  # Start point in green
    ax.plot(self.end[1], self.end[0], 'ro', markersize=10, label='End')  # End point in red

    # Draw the path
    if path:
        path_x = [p[1] for p in path]
        path_y = [p[0] for p in path]
        ax.plot(path_x, path_y, 'b-', linewidth=2, label='Path')  # Path in blue

    # Set grid lines
    ax.set_xticks(np.arange(-0.5, len(self.maze[0]), 1), minor=True)
    ax.set_yticks(np.arange(-0.5, len(self.maze), 1), minor=True)
    ax.grid(which='minor', color='black', linestyle='-', linewidth=2)
    ax.tick_params(which='minor', size=0)

    # Show legend
    ax.legend()

    # Display the plot
    plt.title("Maze with Path")
    plt.show()
```

- **迷宫：**
  - 白色格子表示障碍物。
  - 黑色格子表示可通行路径。
- **起点和终点：**
  - 绿色圆点表示起点。
  - 红色圆点表示终点。
- **路径：**
  - 蓝色线条表示从起点到终点的路径



## 四、实验结果

最终算法输出的路径如下：

![image-20250103145441664](/Users/Caleb/Library/Application Support/typora-user-images/image-20250103145441664.png)



迷宫路径的可视化结果如下：

![Maze with Path](/Users/Caleb/Documents/Senior Course/Reinforcement Learning/MDP/Maze with Path.png)
