import gym
from gym import spaces
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import pandas_datareader.data as web
from collections import deque
import random

class TradingEnv(gym.Env):
    def __init__(self, start_date, end_date, tc=0.05/100, ticker='^DJI'):
        self.start = start_date
        self.end = end_date
        self.tc = tc
        self.ticker = ticker

        self.action_space = spaces.Discrete(3)  # 0: sell, 1: hold, 2: buy
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(7,), dtype=np.float32)

        self.data_df = self.load_and_prepare_data()
        self.curr_index = 0
        self.data_len = self.data_df.shape[0]
        self.position = 0

    def step(self, action):
        done = False
        current_price = self.data_df.iloc[self.curr_index]['Close']
        next_price = self.data_df.iloc[self.curr_index + 1]['Close']

        # Calculate reward
        stock_return = (next_price - current_price) / current_price
        change_in_position = action - 1  # -1: sell, 0: hold, 1: buy
        cost = abs(change_in_position) * self.tc
        reward = change_in_position * stock_return - cost

        # Update position
        self.position += change_in_position

        if self.curr_index == self.data_len - 2:
            done = True
        self.curr_index += 1

        obs = self.get_state()
        info = {'date': self.data_df.index[self.curr_index], 'return': stock_return, 'position': self.position}

        return obs, reward, done, info

    def reset(self):
        self.curr_index = 0
        self.position = 0
        return self.get_state()

    def get_state(self):
        df = self.data_df
        i = self.curr_index
        state = [
            df.iloc[i]['Close'] / df.iloc[i-1]['Close'] - 1,  # Daily return
            df.iloc[i]['Volume'] / df.iloc[i-5:i]['Volume'].mean() - 1,  # Volume
            df.iloc[i]['Close'] / df.iloc[i-5:i]['Close'].mean() - 1,  # 5-day MA
            df.iloc[i]['Close'] / df.iloc[i-20:i]['Close'].mean() - 1,  # 20-day MA
            df.iloc[i]['RSI'],
            df.iloc[i]['MACD'],
            self.position
        ]
        return np.array(state)

    def load_and_prepare_data(self):
        df = web.DataReader(self.ticker, 'stooq', self.start, self.end)
        df = df.sort_index()
        df['RSI'] = self.compute_rsi(df['Close'])
        df['MACD'] = self.compute_macd(df['Close'])
        return df.dropna()

    @staticmethod
    def compute_rsi(prices, period=14):
        delta = prices.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
        rs = gain / loss
        return 100 - (100 / (1 + rs))

    @staticmethod
    def compute_macd(prices, fast=12, slow=26, signal=9):
        ema_fast = prices.ewm(span=fast, adjust=False).mean()
        ema_slow = prices.ewm(span=slow, adjust=False).mean()
        macd = ema_fast - ema_slow
        signal_line = macd.ewm(span=signal, adjust=False).mean()
        return macd - signal_line

    def get_stock_cumulative_return(self):
        return (self.data_df['Close'] / self.data_df['Close'].iloc[0]).to_dict()

class DoubleQLearningAgent:
    def __init__(self, env, alpha=0.1, gamma=0.99, epsilon_start=1.0, epsilon_end=0.01, epsilon_decay=0.995):
        self.env = env
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = epsilon_start
        self.epsilon_end = epsilon_end
        self.epsilon_decay = epsilon_decay
        self.q_table1 = {}
        self.q_table2 = {}
        self.experience_replay = deque(maxlen=1000)

    def get_q_value(self, state, action, q_table):
        return q_table.get((state, action), 0.0)

    def choose_action(self, state):
        if np.random.uniform() < self.epsilon:
            return self.env.action_space.sample()
        else:
            q_values = [self.get_q_value(state, a, self.q_table1) + self.get_q_value(state, a, self.q_table2) for a in range(self.env.action_space.n)]
            return np.argmax(q_values)

    def update_q_tables(self, state, action, reward, next_state):
        self.experience_replay.append((state, action, reward, next_state))
        if len(self.experience_replay) >= 64:
            batch = random.sample(self.experience_replay, 64)
            for state, action, reward, next_state in batch:
                if np.random.random() < 0.5:
                    self.update_q_table(state, action, reward, next_state, self.q_table1, self.q_table2)
                else:
                    self.update_q_table(state, action, reward, next_state, self.q_table2, self.q_table1)

    def update_q_table(self, state, action, reward, next_state, q_table_update, q_table_select):
        best_next_action = max(range(self.env.action_space.n), key=lambda a: q_table_select.get((next_state, a), 0))
        td_target = reward + self.gamma * self.get_q_value(next_state, best_next_action, q_table_update)
        current_q = self.get_q_value(state, action, q_table_update)
        q_table_update[(state, action)] = current_q + self.alpha * (td_target - current_q)

    def decay_epsilon(self):
        self.epsilon = max(self.epsilon_end, self.epsilon * self.epsilon_decay)

def train_agent(env, agent, num_episodes=1000):
    episode_rewards = []
    for episode in range(num_episodes):
        state = tuple(env.reset())
        done = False
        total_reward = 0
        while not done:
            action = agent.choose_action(state)
            next_state, reward, done, _ = env.step(action)
            next_state = tuple(next_state)
            agent.update_q_tables(state, action, reward, next_state)
            state = next_state
            total_reward += reward
        agent.decay_epsilon()
        episode_rewards.append(total_reward)
        if episode % 100 == 0:
            print(f"Episode {episode}, Total Reward: {total_reward}, Epsilon: {agent.epsilon:.2f}")
    return episode_rewards

def test_agent(env, agent):
    state = tuple(env.reset())
    done = False
    total_reward = 0
    actions_taken = []
    while not done:
        action = agent.choose_action(state)
        next_state, reward, done, info = env.step(action)
        state = tuple(next_state)
        total_reward += reward
        actions_taken.append((info['date'], action, info['position'], info['return']))
    stock_cumulative_return = env.get_stock_cumulative_return()
    return total_reward, actions_taken, stock_cumulative_return

def calculate_sharpe_ratio(returns, risk_free_rate=0.01):
    excess_returns = returns - risk_free_rate / 252  # Daily risk-free rate
    return np.mean(excess_returns) / np.std(excess_returns) * np.sqrt(252)

def calculate_sortino_ratio(returns, risk_free_rate=0.01):
    excess_returns = returns - risk_free_rate / 252  # Daily risk-free rate
    downside_returns = np.minimum(excess_returns, 0)
    return np.mean(excess_returns) / np.std(downside_returns) * np.sqrt(252)

def calculate_max_drawdown(cumulative_returns):
    peak = cumulative_returns.expanding(min_periods=1).max()
    drawdown = (cumulative_returns / peak) - 1
    return drawdown.min()

if __name__ == '__main__':
    env = TradingEnv(start_date='2010-01-01', end_date='2024-01-01')
    agent = DoubleQLearningAgent(env)

    # Training
    episode_rewards = train_agent(env, agent)

    # Plot training results
    plt.figure(figsize=(10, 5))
    plt.plot(episode_rewards)
    plt.title('Training Progress')
    plt.xlabel('Episode')
    plt.ylabel('Total Reward')
    plt.show()

    # Testing
    test_reward, actions, stock_cumulative_return = test_agent(env, agent)
    print(f"Test Reward: {test_reward}")

    # Plot test results
    df = pd.DataFrame(actions, columns=['Date', 'Action', 'Position', 'Return'])
    df['Date'] = pd.to_datetime(df['Date'])
    df.set_index('Date', inplace=True)
    df['AgentCumulativeReturn'] = (1 + df['Return']).cumprod()

    plt.figure(figsize=(12, 6))
    plt.plot(df.index, df['AgentCumulativeReturn'], label='Agent Cumulative Return')
    plt.plot(df.index, [stock_cumulative_return[date] for date in df.index], label='Stock Cumulative Return')
    buy_points = df[df['Action'] == 2].index
    sell_points = df[df['Action'] == 0].index
    plt.scatter(buy_points, df.loc[buy_points, 'AgentCumulativeReturn'], color='green', marker='^', label='Buy')
    plt.scatter(sell_points, df.loc[sell_points, 'AgentCumulativeReturn'], color='red', marker='v', label='Sell')
    plt.title('Agent Performance vs Stock Performance')
    plt.xlabel('Date')
    plt.ylabel('Cumulative Return')
    plt.legend()
    plt.show()

    # Calculate risk-adjusted metrics
    agent_returns = df['Return'].values
    stock_returns = np.diff([stock_cumulative_return[date] for date in df.index]) / np.array([stock_cumulative_return[date] for date in df.index][:-1])

    agent_sharpe = calculate_sharpe_ratio(agent_returns)
    stock_sharpe = calculate_sharpe_ratio(stock_returns)

    agent_sortino = calculate_sortino_ratio(agent_returns)
    stock_sortino = calculate_sortino_ratio(stock_returns)

    agent_max_drawdown = calculate_max_drawdown(df['AgentCumulativeReturn'])
    stock_max_drawdown = calculate_max_drawdown(pd.Series(stock_cumulative_return).sort_index())

    print(f"Agent Sharpe Ratio: {agent_sharpe:.4f}")
    print(f"Stock Sharpe Ratio: {stock_sharpe:.4f}")
    print(f"Agent Sortino Ratio: {agent_sortino:.4f}")
    print(f"Stock Sortino Ratio: {stock_sortino:.4f}")
    print(f"Agent Max Drawdown: {agent_max_drawdown:.4f}")
    print(f"Stock Max Drawdown: {stock_max_drawdown:.4f}")
  
