[toc]

# Task——Deep Reinforcement Learning
实现深度强化学习方法:
- Policy Gradient
- Actor-Critic

环境：[月球着陆器](https://gym.openai.com/envs/LunarLander-v2/)

# Baseline
## Simple
定义优势函数(Advantage function)为执行完action之后直到结束每一步的reward累加，即：
$$
A_1=R_1=r_1+r_2+....+r_T,\\
A_2=R_2=r_2+r_3+...+r_T,\\
...\\
A_T=R_T=r_T
$$
其中，$R$为动作状态值函数，$r_i$为执行完动作$a_i$得到的reward
```python
for episode in range(EPISODE_PER_BATCH):

        state = env.reset()
        total_reward, total_step = 0, 0
        seq_rewards = []
        while True:

            action, log_prob = agent.sample(state) # at, log(at|st)
            next_state, reward, done, _ = env.step(action)

            log_probs.append(log_prob) # [log(a1|s1), log(a2|s2), ...., log(at|st)]
            # seq_rewards.append(reward)
            state = next_state
            total_reward += reward
            total_step += 1
            rewards.append(reward) # change here
  
            if done:
                final_rewards.append(reward)
                total_rewards.append(total_reward)

                break
```
## Medium Baseline—Policy Gradient
第二个版本的cumulated reward，把离a1比较近的的reward给比较大的权重，比较远的给比较小的权重，如下：
$$
A_1=R_1=r_1+\gamma r_2+....+\gamma ^{T-1} r_T,\\
A_2=R_2=r_2+\gamma r_3+...+\gamma ^{T-2} r_T,\\
...\\
A_T=R_T=r_T
$$
![在这里插入图片描述](https://i-blog.csdnimg.cn/direct/7c528f3ac50c40dfa7a6b8588db077ad.png)
```python
# Take a state input and generate a probability distribution of an action through a series of Fully Connected Layers
class PolicyGradientNetwork(nn.Module):

    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(8, 16)
        self.fc2 = nn.Linear(16, 16)
        self.fc3 = nn.Linear(16, 4)

    def forward(self, state):
        hid = torch.tanh(self.fc1(state))
        hid = torch.tanh(hid)
        return F.softmax(self.fc3(hid), dim=-1)
```



```python
      while True:
            action, log_prob = agent.sample(state) # at, log(at|st)
            next_state, reward, done, _ = env.step(action)

            log_probs.append(log_prob) # [log(a1|s1), log(a2|s2), ...., log(at|st)]
            seq_rewards.append(reward) # r1, r2, ...., rt
            state = next_state
            total_reward += reward
            total_step += 1 # total_step in each episode is different
            # rewards.append(reward) # change here
            # ! IMPORTANT !
            # Current reward implementation: immediate reward,  given action_list : a1, a2, a3 ......
            #                                                         rewards :     r1, r2 ,r3 ......
            # medium：change "rewards" to accumulative decaying reward, given action_list : a1,                           a2,                           a3, ......
            #                                                           rewards :           r1+0.99*r2+0.99^2*r3+......, r2+0.99*r3+0.99^2*r4+...... ,  r3+0.99*r4+0.99^2*r5+ ......
    
            if done: # done is return by environment, true means current episode is done
                final_rewards.append(reward) # final step reward
                total_rewards.append(total_reward) # total reward of this episode
                # calculate accumulative decaying reward
                discounted_rewards = []
                R = 0
                for r in reversed(seq_rewards):
                  R = r + rate * R
                  discounted_rewards.insert(0, R)

                rewards.extend(discounted_rewards)
                break
```
## Strong Baseline——Actor-Critic
![在这里插入图片描述](https://i-blog.csdnimg.cn/direct/d43f2d55d24043959cf397502058e4bd.png)在 **Actor-Critic** 算法中，**Actor** 和 **Critic** 的损失函数主要基于策略梯度方法（用于更新 Actor 网络）以及价值函数（用于更新 Critic 网络）。这两部分的损失分别由策略的 **Advantage** 估计和状态价值的误差构成。

Actor 网络的目标是通过 **策略梯度（Policy Gradient）** 方法，最大化预期的累计奖励 $ \mathbb{E} [R] $。为此，损失函数通常为负的 log 概率乘以 **Advantage**（优势函数），该优势函数描述了当前策略执行动作的好坏程度。
$$L_{\text{Actor}} = -\mathbb{E}_{\pi} [\log(\pi(a|s)) \cdot A(s, a)]$$

其中：

- $\log(\pi(a|s))$ ：是状态 \( s \) 下选择动作 \( a \) 的 log 概率。
- $ A(s, a) $：是 **Advantage**，代表实际收益与 Critic 估计的差距，定义为：$A(s, a) = r + \gamma V(s)_{t+1} - V(s)_t$
  其中：
  - $r$ ：当前动作的即时奖励。
  - $\gamma $：折扣因子，用于考虑未来奖励的权重。
  - $V(s)_{t+1}$：下一个状态的价值估计。
  - $V(s)_t $：当前状态的价值估计。

因此，Actor 损失函数的整体公式为：
$$L_{\text{Actor}} =-\sum{ ^T _{i=1}} \log(\pi(a|s)) \cdot A(s,a)= -\sum { ^T _{i=1}}\log(\pi(a|s)) \cdot (r + \gamma V(s') - V(s))$$

**2. Critic 损失公式**

Critic 网络的目标是尽可能精确地估计状态的价值 \($ V(s) $\)，所以我们使用 **价值误差** 作为 Critic 损失。常用的损失函数是 **均方误差（Mean Squared Error, MSE）** 或者 **平滑 L1 损失**。

Critic 的损失函数可以写为：
$$L_{\text{Critic}} = \mathbb{E} \left[ \left( V(s)_t - \left( r + \gamma V(s)_{t+1} \right) \right)^2 \right]$$

也就是说，Critic 通过最小化 \( $V(s)_t$\) 和 \( $r + \gamma V(s)_{t+1}$ \) 之间的误差，来提高状态价值的估计。

在使用 **平滑 L1 损失** 的情况下，公式为：$$L_{\text{Critic}} = \text{smooth\_l1\_loss}(V(s)_t, r + \gamma V(s)_{t+1})$$.**平滑 L1 损失** 比均方误差对异常值更具鲁棒性。


```python
from torch.optim.lr_scheduler import StepLR
class ActorCritic(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc = nn.Sequential(
            nn.Linear(8, 16),
            nn.Tanh(),
            nn.Linear(16, 16),
            nn.Tanh()
        )

        self.actor = nn.Linear(16, 4)
        self.critic = nn.Linear(16, 1)

        self.values = []
        self.optimizer = optim.SGD(self.parameters(), lr=0.001)

    def forward(self, state):
        hid = self.fc(state)
        self.values.append(self.critic(hid).squeeze(-1))
        return F.softmax(self.actor(hid), dim=-1)

    def learn(self, log_probs, rewards):
        values = torch.stack(self.values)
        loss = (-log_probs * (rewards - values.detach())).sum() + F.smooth_l1_loss(values, rewards)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        self.values = []

    def sample(self, state):
        action_prob = self(torch.FloatTensor(state))
        action_dist = Categorical(action_prob)
        action = action_dist.sample()
        log_prob = action_dist.log_prob(action)
        return action.item(), log_prob
        

```

## Boss Baseline—Mask
Mask 蒙版（Mask） 和 Rate（折扣因子） 解释**

- Mask

`mask` 是一个 **蒙版向量**，用于过滤掉无效的或不需要考虑的状态值或奖励。这通常在处理 **序列数据** 或者 **部分状态无效的任务** 时很有用。例如，在某些环境中，某些时间步的奖励可能不可用，或这些时间步不需要计入学习。

通过 mask，我们可以有选择性地忽略某些状态或动作：
$$A(s, a) = r + \gamma \cdot \text{mask} \cdot V(s)_{t+1} - V(s)+t$$

- Rate (折扣因子 \( $\gamma$ \))

`rate` 也就是折扣因子 \( $\gamma$ \)，用于对未来奖励进行折现。它的作用是 **平衡即时奖励与长期奖励**。折扣因子 \( $\gamma$ \) 的取值范围通常在 \( $[0, 1] $\)，当 \( $\gamma = 0$ \) 时，表示完全只关注即时奖励；当 \($ \gamma \to 1 $\) 时，表示对长期奖励的重视程度增加。

因此，完整的 **Advantage** 函数（带有蒙版和折扣因子的形式）是：
$$A(s, a) = r + \gamma \cdot \text{mask} \cdot V(s)_{t+1} - V(s)_t$$

**完整的损失函数公式**()

1. Actor 损失公式：
   $$L_{\text{Actor}} = -\sum{ ^T _{i=1}} \log(\pi(a|s)) \cdot \left( r + \gamma \cdot \text{mask} \cdot V(s)_{t+1} - V(s)_t \right)$$

2. Critic 损失公式：
   $$L_{\text{Critic}} = \text{smooth\_l1\_loss}\left( V(s)_t, r + \gamma \cdot \text{mask} \cdot V(s)_{t+1} \right)$$



```python
from torch.optim.lr_scheduler import StepLR
class ActorCritic(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc = nn.Sequential(
            nn.Linear(8, 16),
            nn.Tanh(),
            nn.Linear(16, 16),
            nn.Tanh()
        )
        
        self.actor = nn.Linear(16, 4)
        self.critic = nn.Linear(16, 1)
        
        self.values = []
        self.optimizer = optim.SGD(self.parameters(), lr=0.001)
        self.scheduler = optim.lr_scheduler.CyclicLR(self.optimizer, 
                                                     base_lr=2e-4, max_lr=2e-3, 
                                                     step_size_up=10, mode='triangular2')
        
    def forward(self, state):
        hid = self.fc(state)
        self.values.append(self.critic(hid).squeeze(-1))
        return F.softmax(self.actor(hid), dim=-1)
    
    def learn(self, log_probs, rewards, mask, rate):
        values = torch.stack(self.values)
        advantage = rewards + rate* mask * torch.cat([values[1:], torch.zeros(1)]) - values
        loss = (-log_probs * (advantage.detach())).sum() + \
               F.smooth_l1_loss(advantage, torch.zeros(len(advantage)))
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        self.scheduler.step()
        
        self.values = []
        
    def sample(self, state):
        action_prob = self(torch.FloatTensor(state))
        action_dist = Categorical(action_prob)
        action = action_dist.sample()
        log_prob = action_dist.log_prob(action)
        return action.item(), log_prob

```



```python
while True:

            action, log_prob = agent.sample(state) # at, log(at|st)
            next_state, reward, done, _ = env.step(action)

            log_probs.append(log_prob) # [log(a1|s1), log(a2|s2), ...., log(at|st)]
            seq_rewards.append(reward)
            state = next_state
            total_reward += reward
            total_step += 1
            if done:
                final_rewards.append(reward)
                total_rewards.append(total_reward)
                rewards += seq_rewards
                mask += [1]*len(seq_rewards)
                mask[-1] = 0
                break
```