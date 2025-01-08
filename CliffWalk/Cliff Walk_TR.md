<h1 align='center'>Cliff Walk</h1>

<h2 align='center'>（2024学年秋季学期）</h2>

**课程名称：强化学习与博弈论                   			                       批改人：**

| 实验      | 基于Cliff Walk例子实现SARSA、 Q-learning算法 | 专业（方向） | 计算机科学与技术 |
| --------- | -------------------------------------------- | ------------ | ---------------- |
| **学号**  | **21307244**                                 | **姓名**     | **钟宇**         |
| **Email** | **tue2dayzz@gmail.com**                      | **完成日期** | **2024.12.28**   |

[toc]

## 一、实验内容

基于 Cliff Walk 例子实现 SARSA、Q-Learning 算法

<img src="/Users/Caleb/Library/Application Support/typora-user-images/image-20250103161208342.png" alt="image-20250103161208342" style="zoom:50%;" />



## 二、实验环境

- 操作系统：$\text{Ubuntu 22.04}$
- 编程语言：$\text{Python}$
- IDE：Pycharm CE



## 三、实验过程

### 实验方法

本次实验分别采用 SARSA 以及 Q-Learning 的算法来实现经典的强化学习问题 Cliff Walk



### 算法简述

SARSA 和 Q-learning 是两种经典的 **时序差分学习（Temporal Difference Learning, TD Learning）** 算法，属于强化学习中的无模型（Model-Free）方法。它们通过学习动作价值函数 Q(s,a) 优化策略，适用于解决马尔可夫决策过程（MDP）问题，相比于实验一中实现的值迭代算法，这两种算法都遵循以下三个特点：

- **无模型学习：**
  - SARSA 和 Q-learning 都不需要事先知道环境的转移概率和奖励函数，而是通过与环境的交互来学习。
- **动作价值函数 Q(s,a)：**
  - 两者都通过学习 Q(s,a) 来评估在状态 s 下采取动作 a 的长期累积奖励。
- **时序差分更新：**
  - 利用当前奖励和下一状态的估计值来更新 Q(s,a)，结合了动态规划和蒙特卡罗方法的优点。



#### SARSA

**算法思想：**

SARSA（State-Action-Reward-State-Action）是一种 **在策略（On-Policy）** 算法，即它在学习过程中遵循的策略与优化的策略是相同的。SARSA 通过以下公式更新 Q(s,a)：
$$
Q(s_t,a_t) \leftarrow Q(s_t,a_t)+\alpha[r_{t+1}+ \gamma Q(s_{t+1},a_{t+1})-Q(s_t,a_t)]
$$
其中：

- s<sub>t</sub>, a<sub>t</sub>：当前状态和动作。
- r<sub>t+1</sub>：执行动作 a<sub>t</sub> 后获得的即时奖励。
- s<sub>t+1</sub>, a<sub>t+1</sub>：下一状态和动作（根据当前策略选择）。
- α：学习率，控制更新的步长。
- γ：折扣因子，权衡当前奖励和未来奖励的重要性。



**算法流程：**

1. 初始化 Q(s,a) 为任意值。
2. 在每个时间步 t：
   - 根据当前策略（如 ϵ-贪心策略）选择动作 a<sub>t</sub>。
   - 执行动作 a<sub>t</sub>，观察奖励 r<sub>t+1</sub> 和下一状态 s<sub>t+1</sub>。
   - 根据当前策略选择下一动作 a<sub>t+1</sub>
   - 更新 Q(s<sub>t</sub>, a<sub>t</sub>) 使用上述公式。
   - 转移到下一状态 s<sub>t+1</sub>，并设置 a<sub>t</sub>=a<sub>t+1</sub>。
3. 重复上述过程直到收敛。



**特点：**

- **在策略学习：**SARSA 学习的是当前策略下的 Q 函数，因此更注重实际执行的动作。
- **更保守：**由于考虑了实际执行的动作，SARSA 在学习过程中更注重安全性，适合对风险敏感的任务。



#### Q-Learning

**算法思想：**

Q-learning 是一种 **离策略（Off-Policy）** 算法，即它在学习过程中优化的策略与遵循的策略可以不同。Q-learning 通过以下公式更新 Q(s,a)：
$$
Q(s_t,a_t) \leftarrow Q(s_t,a_t)+\alpha[r_{t+1}+ \gamma \mathop\max_{a'}Q(s_{t+1},a')-Q(s_t,a_t)]
$$
其中：

- $max⁡_{a'}Q(s_{t+1},a')$：在下一状态 s<sub>t+1</sub>下选择使 Q 值最大的动作。



**算法流程：**

1. 初始化 Q(s,a) 为任意值。
2. 在每个时间步 t：
   - 根据当前策略（如 ϵ-贪心策略）选择动作 。
   - 执行动作 a<sub>t</sub>，观察奖励 r<sub>t+1</sub> 和下一状态 s<sub>t+1</sub>。
   - 更新 Q(s<sub>t</sub>, a<sub>t</sub>) 使用上述公式。
   - 转移到下一状态 s<sub>t+1</sub>。
3. 重复上述过程直到收敛。



**特点：**

- **离策略学习：**Q-learning 直接学习最优策略的 Q 函数，而不依赖于当前策略。
- **更激进：**由于总是选择使 Q 值最大的动作进行更新，Q-learning 更倾向于探索最优路径，但可能忽略风险。



### 代码实现

#### Cliff Walk Modeling(Environment)

```python
class CliffWalkingEnv:
    def __init__(self):
        self.rows = 4  # Number of rows in the grid
        self.cols = 12  # Number of columns in the grid
        self.start = (3, 0)  # Starting position
        self.goal = (3, 11)  # Goal position
        self.cliff = [(3, i) for i in range(1, 11)]  # Cliff area
        self.actions = ['U', 'D', 'L', 'R']  # Possible actions
        self.action_dict = {'U': (-1, 0), 'D': (1, 0), 'L': (0, -1), 'R': (0, 1)}  # Action effects
```

根据实验内容给出的悬崖示意图，`CliffWalkingEnv` 类定义了悬崖行走（Cliff Walking）的环境，包括网格大小、起点、终点、悬崖区域以及动作空间。

- `actions = ['U', 'D', 'L', 'R']`：动作空间，分别表示上、下、左、右。
- `action_dict`：动作对应的坐标变换



```python
def step(self, state, action):
    row, col = state
    delta_row, delta_col = self.action_dict[action]
    next_state = (row + delta_row, col + delta_col)

    # Boundary check
    if next_state[0] < 0 or next_state[0] >= self.rows or next_state[1] < 0 or next_state[1] >= self.cols:
        next_state = state  # Stay in place if out of bounds

    # Check if the next state is in the cliff area
    if next_state in self.cliff:
        return self.start, -100, False  # Return to start with a penalty of -100
    elif next_state == self.goal:
        return next_state, 0, True  # Reach the goal with a reward of 0
    else:
        return next_state, -1, False  # Normal step with a reward of -1

def reset(self):
    """Reset the environment to the starting state."""
    return self.start
```

`step` 方法根据当前状态和动作，计算下一个状态，具体来说，函数会检查

1. 边界：如果超出网格范围，则留在原地，状态不变。
2. 悬崖区域：如果进入悬崖，返回起点并给予 -100 的惩罚。
3. 终点：如果到达终点，返回 0 奖励并结束回合。
4. 其他情况：执行动作进入下一个状态并返回 -1 奖励。

`reset` 方法将环境重置到起点



#### SARSA

```python
class SarsaAgent:
    def __init__(self, env, alpha=0.1, gamma=1.0, epsilon=0.1):
        self.env = env  # Environment
        self.alpha = alpha  # Learning rate
        self.gamma = gamma  # Discount factor
        self.epsilon = epsilon  # Exploration rate
        self.q_table = np.zeros((env.rows, env.cols, len(env.actions)))  # Initialize Q-table

    def choose_action(self, state):
        """Epsilon-greedy policy."""
        if np.random.rand() < self.epsilon:
            return np.random.choice(self.env.actions)  # Explore: choose a random action
        else:
            return self.env.actions[np.argmax(self.q_table[state])]  # Exploit: choose the best action

    def learn(self, num_episodes):
        """Train the agent using the SARSA algorithm."""
        rewards = []  # Store rewards for each episode
        all_paths = []  # Store paths for each episode

        for episode in range(num_episodes):
            state = self.env.reset()  # Reset the environment
            action = self.choose_action(state)  # Choose an initial action
            total_reward = 0
            path = [state]  # Store the path taken in this episode

            while True:
                next_state, reward, done = self.env.step(state, action)  # Take a step
                next_action = self.choose_action(next_state)  # Choose the next action
                total_reward += reward
                path.append(next_state)

                # SARSA update
                current_action_index = self.env.actions.index(action)
                next_action_index = self.env.actions.index(next_action)
                self.q_table[state][current_action_index] += self.alpha * (reward + self.gamma * self.q_table[next_state][next_action_index] - [.q_table[state]current_action_index])

                state, action = next_state, next_action  # Move to the next state and action
                if done:
                    break

            rewards.append(total_reward)
            all_paths.append(path)  # Save the path for visualization

        return rewards, all_paths
```

`SarsaAgent` 类实现了 SARSA 算法，用于训练智能体在悬崖行走环境中找到最优路径。

首先进行类的初始化：

- 接收环境对象 (`env`) 和超参数（学习率 `alpha`、折扣因子 `gamma`、探索率 `epsilon`）。
- 初始化 Q 表 (`q_table`)，用于存储每个状态-动作对的价值。

**`choose_action` 方法**为智能体在某一状态下的动作选择策略，本次实验中使用 ϵ-贪心策略选择动作

- 以概率 ϵ 随机选择动作进行探索
- 以概率 1−ϵ 选择当前 Q 值最大的动作

**`learn` 方法**则实现了使用 SARSA 算法训练智能体的具体逻辑。一共进行 `num_episodes` 个轮次进行迭代更新，每个回合开始时，环境被重置，智能体选择一个初始动作。

- `total_reward` 用于累计当前回合的总奖励。
- `path` 用于记录当前回合的状态序列

每个轮次中的算法流程与算法简述部分介绍的类似：

1. 初始化状态和动作。

2. 执行动作，获得下一个状态、奖励和是否结束的标志。

3. 使用 ϵ -贪心策略选择下一个动作。

4. 更新 Q 值：
   $$
   Q(s_t,a_t) \leftarrow Q(s_t,a_t)+\alpha[r_{t+1}+ \gamma Q(s_{t+1},a_{t+1})-Q(s_t,a_t)]
   $$

5. 转移到下一个状态和动作。

6. 如果环境返回 `done=True`，表示当前智能体已到达悬崖终点，退出时间步循环。

每个回合的总奖励和路径被存储到 `rewards` 和 `all_paths` 中。



#### Q-Learning

```python
# Q-learning Agent
class QLearningAgent:
    def __init__(self, env, alpha=0.1, gamma=1.0, epsilon=0.1):
        self.env = env  # Environment
        self.alpha = alpha  # Learning rate
        self.gamma = gamma  # Discount factor
        self.epsilon = epsilon  # Exploration rate
        self.q_table = np.zeros((env.rows, env.cols, len(env.actions)))  # Initialize Q-table

    def choose_action(self, state):
        """Epsilon-greedy policy."""
        if np.random.rand() < self.epsilon:
            return np.random.choice(self.env.actions)  # Explore: choose a random action
        else:
            return self.env.actions[np.argmax(self.q_table[state])]  # Exploit: choose the best action

    def learn(self, num_episodes):
        """Train the agent using the Q-learning algorithm."""
        rewards = []  # Store rewards for each episode
        all_paths = []  # Store paths for each episode

        for episode in range(num_episodes):
            state = self.env.reset()  # Reset the environment
            total_reward = 0
            path = [state]  # Store the path taken in this episode

            while True:
                action = self.choose_action(state)  # Choose an action
                next_state, reward, done = self.env.step(state, action)  # Take a step
                total_reward += reward
                path.append(next_state)

                # Q-learning update
                current_action_index = self.env.actions.index(action)
                self.q_table[state][current_action_index] += self.alpha * (reward + self.gamma * np.max(self.q_table[next_state]) - self.q_table[state][current_action_index])

                state = next_state  # Move to the next state
                if done:
                    break

            rewards.append(total_reward)
            all_paths.append(path)  # Save the path for visualization

        return rewards, all_paths
```

`QLearningAgent` 类实现了 Q-Learning 算法，用于训练智能体在悬崖行走环境中找到最优路径。

其算法流程与 SARSA 算法类似，唯一的不同点为其直接学习最优策略的 Q 函数，而不依赖于当前策略。

因此其更新函数中的 Q 函数 Q(s<sub>t+1</sub>, a') 并不通过 `choose_action` 函数获取状态 s<sub>t+1</sub> 对应的动作 a<sub>t+1</sub> 计算而得，而是直接通过 `np.max(self.q_table[next_state]` 获取最优策略的 Q 函数，对应公式
$$
Q(s_t,a_t) \leftarrow Q(s_t,a_t)+\alpha[r_{t+1}+ \gamma \mathop\max_{a'}Q(s_{t+1},a')-Q(s_t,a_t)]
$$
但是在实际执行过程中，其依旧使用 ϵ-贪心策略，通过 `choose_action` 函数来选择状态 s<sub>t+1</sub> 对应的动作



#### Main

```python
if __name__ == '__main__':
    # Initialize the environment
    env = CliffWalkingEnv()

    # Initialize the agents
    sarsa_agent = SarsaAgent(env, alpha=0.1, gamma=1.0, epsilon=0.1)
    q_learning_agent = QLearningAgent(env, alpha=0.1, gamma=1.0, epsilon=0.1)

    # Train the agents
    num_episodes = 500
    rewards_sarsa, paths_sarsa = sarsa_agent.learn(num_episodes)
    rewards_q_learning, paths_q_learning = q_learning_agent.learn(num_episodes)

    # Visualize the final paths
    visualize_path(env, paths_sarsa[-1], "SARSA Final Path")
    visualize_path(env, paths_q_learning[-1], "Q-learning Final Path")

    # Plot the rewards
    plot_rewards(rewards_sarsa, rewards_q_learning)
```

本次实验中的参数设置为：

- 调整学习率 `α` = 0.1
- 折扣因子 `γ` = 1.0
- 探索率 `ϵ` = 0.1
- 训练迭代轮次 `num_episodes` = 500

最后实验将动态展示在 SARSA 算法和 Q-Learning 算法下智能体在最后一回合中的路径，以及**奖励曲线**，用于比较 SARSA 和 Q-learning 的每回合奖励，分析算法性能



## 四、实验结果

### 路径展示

#### SARSA

<img src="/Users/Caleb/Documents/Senior Course/Reinforcement Learning/CliffWalk/Result/SARSA Final Path.png" alt="SARSA Final Path" style="zoom:72%;" />

#### Q-Learning

<img src="/Users/Caleb/Documents/Senior Course/Reinforcement Learning/CliffWalk/Result/Q-Learning Final Path.png" alt="Q-Learning Final Path" style="zoom:72%;" />

### 奖励曲线

![Rewards](/Users/Caleb/Documents/Senior Course/Reinforcement Learning/CliffWalk/Result/Rewards.png)

### 算法对比

由奖励曲线可得：

- **Q-Learning：**奖励曲线在训练初期可能波动较大，但随着训练回合的增加，奖励逐渐趋于稳定，并且最终达到较高的奖励值。

- **SARSA：**奖励曲线在训练初期可能比 Q-Learning更稳定，但最终奖励值可能略低于 Q-Learning。

综上所述，Q-Learning在训练过程中表现出更强的探索性，可能在早期阶段有较大的波动，但最终能够获得更高的奖励。SARSA 则表现出更稳定的学习过程，但最终奖励可能略低于 Q-learning。这表明 Q-learning可能在长期学习中具有更强的优化能力，而 SARSA 则在稳定性上表现更好。

由两种算法各自的最终路径也可得，Q-Learning 算法的路径选择相较于 SARSA 算法的路径选择更优。这是因为

- **Q- Learning** 是一种 **off-policy** 算法，它在学习过程中更注重探索，能够更快地找到最优策略。从奖励曲线可以看出，Q-learning在训练后期能够获得更高的奖励，表明它在长期学习中具有更好的性能。
- **SARSA** 是一种 **on-policy** 算法，它在学习过程中更注重当前策略的优化，因此在训练过程中表现出更高的稳定性。然而，由于它更依赖于当前策略，可能在探索性上不如 Q-learning，导致最终奖励略低。



由分析也可给出两种算法各自的适用场景：

- **Q-Learning：**适用于需要较强探索性的场景，尤其是在环境动态变化较大或需要找到全局最优解的情况下。
- **SARSA：**适用于需要较高稳定性的场景，尤其是在环境相对稳定或需要避免高风险动作的情况下。

