<h1 align='center'>Deep Q-Network</h1>

<h2 align='center'>（2024学年秋季学期）</h2>

**课程名称：强化学习与博弈论                   			                       批改人：**

| 实验      | 基于DQN算法训练CartPole-v1 | 专业（方向） | 计算机科学与技术 |
| --------- | -------------------------- | ------------ | ---------------- |
| **学号**  | **21307244**               | **姓名**     | **钟宇**         |
| **Email** | **tue2dayzz@gmail.com**    | **完成日期** | **2024.1.3**     |

[toc]

## 一、实验内容

本次实验要求使用一种深度强化学习算法解决一个应用案例

DQN（Deep Q-Network）作为⼀种结合了深度学习和强化学习的算法，已经⼴泛应⽤于控制问题和决策任务，因此本次实验也将使用 DQN 来训练 Gymnasium 环境中的经典控制问题 `CartPole-v1`。

`CartPole-v1` 是一个典型的强化学习基准问题，目标是控制一个小车使其上的杆子保持直立不倒。通过本次实验，旨在掌握深度强化学习的基本原理，并实现一个能够稳定解决 `CartPole-v1` 问题的智能体。



## 二、实验环境

**环境名称**：`CartPole-v1`

**环境描述**：

- 状态空间：4 维连续空间，包括小车位置、小车速度、杆子角度和杆子角速度。
- 动作空间：2 维离散空间，包括向左推小车和向右推小车。
- 奖励机制：每保持杆子直立一步，奖励 +1；当杆子倾斜超过一定角度或小车移动超出边界时，回合结束。
- 目标：最大化累积奖励，通常达到 500 分即可认为问题解决。



**实验工具**

- **编程语言**：Python
- **深度学习框架**：PyTorch
- **环境库**：Gymnasium
- **可视化工具**：Matplotlib



## 三、实验过程

### 实验方法

本次实验采用 DQN 算法来实现经典深度强化学习中的控制问题 `CartPole-v1`



### 算法简述

DQN（Deep Q-Network）是一种将深度神经网络与 Q-Learning 相结合的强化学习算法。它通过神经网络来近似 Q 值函数，从而解决了**传统 Q-Learning 在高维状态空间中难以处理的问题**。DQN 的核心思想是利用深度学习的强大拟合能力，直接从原始输入（如状态）中学习最优策略



#### Q-Learning 与 Q 值函数

Q-Learning 是一种基于值函数的强化学习算法，其目标是学习一个 Q 值函数 Q(s,a)，表示在状态 s 下执行动作 a 后，未来累积奖励的期望值。Q 值函数的更新公式为：
$$
Q(s,a) \leftarrow Q(s,a)+\alpha[r+ \gamma \mathop\max_{a'}Q(s',a')-Q(s,a)]
$$
其中：

- α 是学习率。
- γ 是折扣因子，用于平衡当前奖励和未来奖励的重要性。
- r 是执行动作 a 后获得的即时奖励。
- s′ 是执行动作 a 后的下一状态。
- max<sub>⁡a′</sub>Q(s′,a′) 是在下一状态 s′ 下选择最优动作的 Q 值。

Q-Learning 的目标是通过不断更新 Q 值函数，最终找到最优策略 π*，使得在每个状态下选择的动作能够最大化累积奖励。



#### DQN 核心思想

在传统 Q-Learning 中，Q 值函数通常以表格形式存储，适用于状态空间和动作空间较小的问题。然而，当状态空间或动作空间较大时，表格形式的 Q 值函数会变得难以处理。DQN 通过引入深度神经网络来近似 Q 值函数，从而解决了这一问题。

DQN的核心思想包括以下两点：

1. **用神经网络近似 Q 值函数**：

   - 使用一个深度神经网络 Q(s,a;θ) 来近似 Q 值函数，其中 θ 是神经网络的参数。
   - 神经网络的输入是状态 s，输出是每个动作 a 对应的 Q 值。

2. **通过梯度下降优化 Q 值函数**：

   - 定义损失函数为Q值函数的均方误差（MSE）：
     $$
     L(\theta)=E \left[\left(r+ \gamma \mathop\max_{⁡a'}Q(s',a';\theta^-)−Q(s,a;\theta)\right)^2 \right]
     $$

   - 其中，θ^-^ 是目标网络的参数，用于计算目标 Q 值 $r+ \gamma \mathop\max_{⁡a'}Q(s',a';\theta^-)$

   - 通过梯度下降法更新神经网络的参数 θ ，以最小化损失函数。



#### DQN 关键技术

为了稳定训练过程并提高算法的性能，DQN 引入了以下关键技术：

1. **经验回放（Experience Replay）**：
   - 将智能体与环境交互的经验（状态、动作、奖励、下一状态、是否终止）存储在一个经验回放缓冲区中。
   - 在训练时，从缓冲区中随机采样一个批次的经验进行学习。
   - 经验回放的作用：
     - 打破数据之间的相关性，提高训练的稳定性。
     - 提高数据的利用率，避免重复采样。
2. **目标网络（Target Network）**：
   - 使用一个独立的神经网络（目标网络）来计算目标 Q 值 $r+ \gamma \mathop\max_{⁡a'}Q(s',a';\theta^-)$。
   - 目标网络的参数 θ^−^ 定期从主网络（评估网络）复制而来。
   - 目标网络的作用：
     - 稳定目标 Q 值的计算，避免训练过程中的振荡。
     - 提高算法的收敛性。
3. **探索与利用（Exploration vs. Exploitation）**：
   - 使用 ϵ-贪婪策略（ϵ-greedy policy）来平衡探索与利用。
   - 在每一步中，以概率 ϵ 随机选择动作（探索），以概率 1−ϵ 选择当前 Q 值最大的动作（利用）。
   - ϵ 的值通常从初始值（如 1.0）**逐步衰减**到一个较小的值（如 0.05），以便在训练初期更多地探索，后期更多地利用。



#### DQN 训练流程

1. 初始化评估网络 Q(s,a;θ) 和目标网络 Q(s,a;θ^−^)，并设置 θ^−^=θ。

2. 初始化经验回放缓冲区。

3. 对于每个 episode：

   - 初始化状态 s。

   - 对于每一步：

     - 使用 ϵ-贪婪策略选择动作 a。

     - 执行动作 a，观察奖励 r 和下一状态 s′。

     - 将经验 (s,a,r,s′,done) 存储到经验回放缓冲区中。

     - 从缓冲区中随机采样一个批次的经验。

     - 计算目标Q值：
       $$
       y = r+ \gamma \mathop\max_{⁡a'}Q(s',a';\theta^-)
       $$

     - 计算损失函数：
       $$
       L(\theta) = \frac{1}{N} \sum^N_{i=1}(y_i-Q(s_i,a_i;\theta))^2
       $$

     - 使用梯度下降法更新评估网络的参数 θ。

     - 定期更新目标网络的参数 θ^−^。

4. 重复上述过程，直到智能体达到预期的性能。



### 代码实现

#### Q-Network 架构

```python
class QNet(nn.Module):
    """Q-network for DQN."""
    def __init__(self, input_size, hidden_size, output_size):
        super(QNet, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.fc3 = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        return self.fc3(x)
```

`QNet` 类定义一个简单的全连接神经网络，用于近似 Q 函数。

Q-Network 由三个全连接层构成：

- `fc1`：输入层到隐藏层。
- `fc2`：隐藏层到隐藏层。
- `fc3`：隐藏层到输出层。

其中

- 输入大小 `input_size` 为环境中状态空间的大小 `state_size`
- 输出大小 `output_size` 为环境中动作空间的大小 `action_size`



#### 经验回放缓冲区

```python
class ReplayBuffer:
    """Experience replay buffer for DQN."""
    def __init__(self, capacity):
        self.buffer = deque(maxlen=capacity)

    def __len__(self):
        return len(self.buffer)

    def push(self, *transition):
        self.buffer.append(transition)

    def sample(self, batch_size):
        return random.sample(self.buffer, batch_size)

    def clear(self):
        self.buffer.clear()
```

`ReplayBuffer` 类是 DQN 的经验回放缓冲区，用于存储智能体的经验（状态、动作、奖励、下一个状态、终止标志），同时也支持网络可以随机采样一批经验用于训练

经验回放缓冲区采用了双端队列（`deque`）的数据结构，最大长度为 `capacity`。如果 buffer 内的经验超过了最大长度，则旧的经验会自动从 buffer 内弹出并替换为新的经验



#### DQN Agent

```python
class DQNAgent:
    """DQN agent."""
    def __init__(self, state_size, action_size, hidden_size, device, args):
        self.state_size = state_size
        self.action_size = action_size
        self.device = device
        self.args = args
        self.learn_step = 0

        # Networks
        self.eval_net = QNet(state_size, hidden_size, action_size).to(device)
        self.target_net = QNet(state_size, hidden_size, action_size).to(device)
        self.target_net.load_state_dict(self.eval_net.state_dict())

        # Optimizer and loss function
        self.optimizer = optim.Adam(self.eval_net.parameters(), lr=args.lr)
        self.loss_fn = nn.MSELoss()

        # Replay buffer
        self.buffer = ReplayBuffer(args.capacity)

        # Epsilon for epsilon-greedy
        self.eps = args.eps

    def choose_action(self, state):
        """Choose an action using epsilon-greedy policy."""
        if random.random() <= self.eps:
            return random.randint(0, self.action_size - 1)
        else:
            with torch.no_grad():
                state = torch.FloatTensor(state).unsqueeze(0).to(self.device)
                q_values = self.eval_net(state)
                return torch.argmax(q_values).item()

    def store_transition(self, *transition):
        """Store a transition in the replay buffer."""
        self.buffer.push(*transition)

    def update_target_net(self):
        """Update the target network."""
        self.target_net.load_state_dict(self.eval_net.state_dict())
```

`DQNAgent` 类实现了 DQN 算法的核心逻辑，包括动作选择、经验存储、网络更新等

首先该智能体初始化评估网络（`eval_net`）和目标网络（`target_net`）分别用于计算估计 Q 值以及目标 Q 值，接着初始化优化器（Adam）、损失函数（MSE）以及先前创建的经验回放缓冲区 `buffer`，最后设置了初始 ε 值。

- `choose_action` 行动选择采用了 **ε -greedy 策略**，即

  - 以概率 ε 选择随机行动（探索） 
  - 以 1-ε 的概率选择当前策略（ 利用） 

  如果随机采样小于等于 ε 探索率，就通过 `np.random.randint` 函数在动作空间 `self.env.action_space.n` 中随机选择一个行动,否则，就选择当前状态下预测的 q 值最大的行动  

  - **禁用梯度计算， 提高效率**：使用 `eval_net`（评估网络）来计算 Q 值，并根据 Q 值选择动作。这个过程是推理阶段的一部分，不需要计算梯度 
  - 通过 `torch.unsqueeze` 函数来调整输入状态 `state` 的形状以适应神经网络的输入

- **`store_transition` 方法**用于将一条经验存储到缓冲区中
- **`update_target_net` 方法**用于在学习一定步数后，将目标网络的参数更新为评估网络的参数



#### DQN Learn

```python
def learn(self):
    """Train the agent using a batch of experiences."""
    # Sample a batch of transitions
    batch = self.buffer.sample(self.args.batch_size)
    states, actions, rewards, next_states, dones = zip(*batch)

    # Convert to tensors and move to device
    states = torch.FloatTensor(np.array(states)).to(self.device)
    actions = torch.LongTensor(actions).to(self.device)
    rewards = torch.FloatTensor(rewards).to(self.device)
    next_states = torch.FloatTensor(np.array(next_states)).to(self.device)
    dones = torch.FloatTensor(dones).to(self.device)

    # Compute q_eval with eval_net and q_next with target_net
    q_eval = self.eval_net(states).gather(1, actions.unsqueeze(1)).squeeze(1)
    q_next = self.target_net(next_states).detach().max(1)[0]
    q_target = rewards + self.args.gamma * (1 - dones) * q_next

    # Compute loss and update the network
    loss = self.loss_fn(q_eval, q_target)
    self.optimizer.zero_grad()
    loss.backward()
    self.optimizer.step()

    # Update epsilon
    self.eps = max(self.args.eps_min, self.eps * self.args.eps_decay)

    # Update target network
    if self.learn_step % self.args.update_target == 0:
        self.update_target_net()
    self.learn_step += 1
```

`learn` 方法实现了 DQN 网络的主要学习、更新逻辑

##### 计算 q_eval

`q_eval` 使用评估网络 `eval_net` 计算当前状态 `states` 下所有动作的 Q 值，输出形状为 `(batch_size, action_size)`，表示每个状态下所有动作的 Q 值

- **`gather(1, actions.unsqueeze(1))`** 用于从 `eval_net` 的输出中选择与 `actions` 对应的 Q 值。
- 最后输出是一个形状为 `(batch_size,)` 的张量，表示**每个状态下选择的动作的 Q 值**



##### 计算 q_next

`q_next` 使用目标网络 `target_net` 计算下一个状态（`next_states`）的所有动作中最大的 Q 值

- **`detach()`**：

  - 从计算图中分离 `target_net` 的输出，避免梯度传播到目标网络。

  - 这是因为目标网络的参数是在一定学习步数后由评估网络的参数更新得到的，而**不需要通过反向传播进行梯度更新**

- **`max(1)[0]`**：

  - 沿着第 1 维度（动作维度）计算最大值。
  - 返回一个形状为 `(batch_size,)` 的张量，表示每个下一个状态下所有动作中的最大 Q 值。



##### 计算 q_target

根据 Q-learning 算法的更新规则，结合即时奖励、折扣因子和下一个状态的最大 Q 值估计来计算目标 Q 值

$$
q\_target = rewards + args.gamma * (1 - dones) * q\_next
$$

- 如果 episode 未结束，目标 Q 值包括当前奖励和未来奖励的折扣值。
- 如果 episode 结束，目标 Q 值仅包括当前奖励

- `(1 - dones)` 是终止标志的补码
  - 如果 `dones = 1.0`（episode 结束），则 `(1 - dones) = 0.0`，表示没有未来奖励。
  - 如果 `dones = 0.0`（episode 未结束），则 `(1 - dones) = 1.0`，表示考虑未来奖励



#####  ε 和目标网络的更新

初始时 ε 为 1，表示完全探索，随着训练过程 ε 按照衰减率（`eps_decay`）逐渐降低，以增加智能体的利⽤能⼒，但由于仍然要在后期增加探索的随机性，因此也需要确保 ε 不低于最小值（`eps_min`）

目标网络的参数 θ^−^ 定期从评估网络复制而来，评估网络在学习的过程中会维护一个参数 `self.learn_step` 用于记录更新的步数，每隔一定步数（`update_target`），就将目标网络的参数更新为评估网络的参数



#### Train

```python
def train(env, agent, args):
    """Train the DQN agent."""
    scores = []
    for episode in range(args.n_episodes):
        state = env.reset()[0]
        episode_reward = 0
        done = False
        step_cnt = 0

        while not done and step_cnt < 500:
            step_cnt += 1
            action = agent.choose_action(state)
            next_state, reward, done, _, _ = env.step(action)
            agent.store_transition(state, action, reward, next_state, done)
            episode_reward += reward
            state = next_state

            # Train the agent if enough data is available
            if len(agent.buffer) >= args.batch_size:
                agent.learn()

        scores.append(episode_reward)
        print(f"Episode: {episode + 1}, Reward: {episode_reward}, Epsilon: {agent.eps:.4f}")

    return scores
```

`train` 函数为 DQN 的训练函数，一共训练 `n_episodes` 个轮次，`scores` 用于存储每一个轮次的总奖励值

在每一个轮次中，智能体与环境交互，选择动作并存储经验，同时函数也进行了网络更新前的判断，由于采用的是经验回放的技术，需要从缓冲区中随机采样一个批次的经验进行学习，因此只有当 `buffer` 中有多于一个批次数量大小的经验才可以进行学习

由于 `CartPole-v1` 通常达到 500 分即可认为问题解决，因此每一个轮次中也维护一个变量 `step_cnt`，当智能体获得 500 分的奖励时，该轮次也自动结束



## 四、实验结果

### 奖励曲线

DQN 在 CartPole-v1 环境中训练的奖励（Reward）曲线

- 横轴是训练的 Episode
- 纵轴是每个 Episode 的 Reward

![DQN Rewards](/Users/Caleb/Documents/Senior Course/Reinforcement Learning/DeepLearning/DQN Rewards.jpg)

**分析：**

- **早期学习阶段**

  在前 100 个 Episode 左右，Reward 逐渐上升。这说明 DQN 模型在不断探索环境，学习如何更好地控制 CartPole。曲线的增长虽然有些波动，但总体上呈现上升趋势，这是正常的，尤其是在 Q-learning 方法中，探索（Exploration）阶段的随机性会导致这样的现象。

- **中期阶段**

  从 Episode 100 到大约 200，Reward 达到了接近 500 的值（环境的最大分数）。这表明模型开始学到了一个较优的策略。

  但是可以观察到较多的波动：有时 Reward 很高（接近 500），有时却很低。这可能有以下几个原因：

  1. 探索与利用权衡问题：在训练过程中，DQN 仍然会以一定概率进行探索，而探索会导致策略暂时变差。
  2. 模型不稳定：由于 DQN 训练可能会面临过大的目标值波动（即 Q-value overestimation），可能导致策略不稳定。
  3. 经验回放池的问题：如果经验回放池中存储了太多过时或低质量的数据，也可能影响训练效果。

- 后期阶段

  从 Episode 200 开始，Reward 仍然有明显的波动。虽然模型能够多次达到接近最大分数的表现，但仍然频繁出现较低分数的 Episode（甚至接近 0）。



### 改进方向

1. **优化探索策略**

   当前的 Epsilon-Greedy 探索策略可能导致过多的随机性。可以尝试使用 Decaying Epsilon 或替代方法（如 Boltzmann Exploration）。或者引入更高级的策略，如 Noisy Nets。

2. **目标网络更新**

   更改目标网络（Target Network）更新频率

3. **优化 DQN 算法**

   可以尝试 Double DQN、Dueling DQN 或者 Prioritized Experience Replay，从而减小 Q-value 的高估问题，提高性能。
   
