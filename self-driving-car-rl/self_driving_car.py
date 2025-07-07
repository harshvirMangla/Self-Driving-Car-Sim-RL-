import numpy as np
import pygame
import random
import math
from collections import deque
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import matplotlib.pyplot as plt
import json

pygame.init()

WIN_W = 1200
WIN_H = 800
CAR_W = 20
CAR_H = 10
ROAD_W = 100
FPS = 60

WHITE = (255, 255, 255)
BLACK = (0, 0, 0)
RED = (255, 0, 0)
GREEN = (0, 255, 0)
BLUE = (0, 0, 255)
YELLOW = (255, 255, 0)
GRAY = (128, 128, 128)
DARK_GRAY = (64, 64, 64)

class Car:
    def __init__(self, x, y):
        self.x = x
        self.y = y
        self.angle = 0
        self.speed = 0
        self.max_spd = 8
        self.accel = 0.5
        self.brake = 0.3
        self.turn_spd = 5
        self.width = CAR_W
        self.height = CAR_H
        self.sensors = [1.0] * 5
        self.sensor_len = 150
        self.alive = True
        self.dist_traveled = 0
        self.last_x = x
        self.last_y = y
        self.crash_count = 0
        self.update_sensors()

    def update(self, action):
        if not self.alive:
            return
          
        if action == 1:
            self.angle -= self.turn_spd
        elif action == 2:
            self.angle += self.turn_spd
        elif action == 3:
            self.speed = min(self.speed + self.accel, self.max_spd)
        elif action == 4:
            self.speed = max(self.speed - self.brake, 0)
          
        self.speed *= 0.98
        self.x += math.cos(math.radians(self.angle)) * self.speed
        self.y += math.sin(math.radians(self.angle)) * self.speed
        dist = math.sqrt((self.x - self.last_x) ** 2 + (self.y - self.last_y) ** 2)
        self.dist_traveled += dist
        self.last_x, self.last_y = self.x, self.y
        self.x = max(0, min(WIN_W - self.width, self.x))
        self.y = max(0, min(WIN_H - self.height, self.y))
        self.update_sensors()

    def update_sensors(self):
        sensor_angles = [-90, -45, 0, 45, 90]
        self.sensors = []
      
        for angle in sensor_angles:
            rad = math.radians(self.angle + angle)
            end_x = self.x + math.cos(rad) * self.sensor_len
            end_y = self.y + math.sin(rad) * self.sensor_len
            distance = self.sensor_len
            if end_x <= 0 or end_x >= WIN_W:
                if math.cos(rad) != 0:
                    distance = min(distance, abs(self.x - (0 if end_x <= 0 else WIN_W)) / abs(math.cos(rad)))
            if end_y <= 0 or end_y >= WIN_H:
                if math.sin(rad) != 0:
                    distance = min(distance, abs(self.y - (0 if end_y <= 0 else WIN_H)) / abs(math.sin(rad)))
            self.sensors.append(distance / self.sensor_len)
          
        while len(self.sensors) < 5:
            self.sensors.append(1.0)
        self.sensors = self.sensors[:5]

    def get_state(self, targ_x, targ_y):
        dx = targ_x - self.x
        dy = targ_y - self.y
        dist_to_targ = math.sqrt(dx ** 2 + dy ** 2)
        angle_to_targ = math.atan2(dy, dx)
        rel_angle = angle_to_targ - math.radians(self.angle)
        rel_angle = math.atan2(math.sin(rel_angle), math.cos(rel_angle))
      
        state = [
            self.x / WIN_W,
            self.y / WIN_H,
            math.sin(math.radians(self.angle)),
            math.cos(math.radians(self.angle)),
            self.speed / self.max_spd,
            dist_to_targ / 1000,
            math.sin(rel_angle),
            math.cos(rel_angle),
        ]
      
        if len(self.sensors) != 5:
            self.sensors = [1.0] * 5
        state.extend(self.sensors)
        return np.array(state, dtype=np.float32)

    def check_collision(self):
        margin = 30
      
        if (self.x < margin or self.x > WIN_W - margin or
                self.y < margin or self.y > WIN_H - margin):
            self.alive = False
            self.crash_count += 1
            return True
                  
        return False

    def draw(self, screen):
        if not self.alive:
            return
          
        for i, angle in enumerate([-90, -45, 0, 45, 90]):
            rad = math.radians(self.angle + angle)
            end_x = self.x + math.cos(rad) * self.sensors[i] * self.sensor_len
            end_y = self.y + math.sin(rad) * self.sensors[i] * self.sensor_len
            pygame.draw.line(screen, YELLOW, (self.x, self.y), (end_x, end_y), 1)
          
        car_pts = [
            (-self.width // 2, -self.height // 2),
            (self.width // 2, -self.height // 2),
            (self.width // 2, self.height // 2),
            (-self.width // 2, self.height // 2)
        ]
      
        rotated_pts = []
        for px, py in car_pts:
            rad = math.radians(self.angle)
            rx = px * math.cos(rad) - py * math.sin(rad)
            ry = px * math.sin(rad) + py * math.cos(rad)
            rotated_pts.append((self.x + rx, self.y + ry))
        pygame.draw.polygon(screen, RED if self.alive else GRAY, rotated_pts)

class DQNAgent:
    def __init__(self, state_sz, action_sz):
        self.state_sz = state_sz
        self.action_sz = action_sz
        self.memory = deque(maxlen=10000)
        self.epsilon = 1
        self.eps_min = 0.01
        self.eps_decay = 0.995
        self.lr = 0.001
        self.gamma = 0.95
        self.batch_sz = 64
        self.q_net = self.build_model()
        self.targ_net = self.build_model()
        self.update_targ_net()

    def build_model(self):
        model = keras.Sequential([
            layers.Input(shape=(self.state_sz,)),
            layers.Dense(128, activation='relu'),
            layers.Dense(128, activation='relu'),
            layers.Dense(64, activation='relu'),
            layers.Dense(self.action_sz, activation='linear')
        ])
        model.compile(optimizer=keras.optimizers.Adam(learning_rate=self.lr),
                      loss='mse')
        return model

    def update_targ_net(self):
        self.targ_net.set_weights(self.q_net.get_weights())

    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    def act(self, state):
        if np.random.random() <= self.epsilon:
            return random.randrange(self.action_sz)
          
        q_vals = self.q_net.predict(state.reshape(1, -1), verbose=0)
        return np.argmax(q_vals[0])

    def replay(self):
        if len(self.memory) < self.batch_sz:
            return
          
        batch = random.sample(self.memory, self.batch_sz)
        states = np.array([e[0] for e in batch])
        actions = np.array([e[1] for e in batch])
        rewards = np.array([e[2] for e in batch])
        next_states = np.array([e[3] for e in batch])
        dones = np.array([e[4] for e in batch])
        curr_qs = self.q_net.predict(states, verbose=0)
        next_qs = self.targ_net.predict(next_states, verbose=0)
        targets = curr_qs.copy()
        max_next_qs = np.max(next_qs, axis=1)
        targets[range(self.batch_sz), actions] = rewards + (1 - dones) * self.gamma * max_next_qs
        self.q_net.fit(states, targets, epochs=1, verbose=0)
      
        if self.epsilon > self.eps_min:
            self.epsilon *= self.eps_decay

class Environment:
    def __init__(self):
        self.car = Car(100, 400)
        self.targ_x = WIN_W - 100
        self.targ_y = 400
        self.randomize_targ()
        self.ep_reward = 0
        self.steps = 0
        self.max_steps = 500
        self.got_100 = True
        self.got_200 = True

    def randomize_targ(self):
        self.targ_x = random.randint(100, WIN_W - 100)
        self.targ_y = random.randint(100, WIN_H - 100)

    def reset(self):
        while True:
            x = random.randint(100, WIN_W - 200)
            y = random.randint(100, WIN_H - 100)
            dist = math.hypot(self.targ_x - x, self.targ_y - y)
            if dist > 300:
                break
              
        self.car = Car(x, y)
        self.randomize_targ()
        self.ep_reward = 0
        self.steps = 0
        self.got_200 = False
        self.got_100 = False
        return self.car.get_state(self.targ_x, self.targ_y)

    def step(self, action):
        self.steps += 1
        prev_dist = math.sqrt((self.car.x - self.targ_x) ** 2 + (self.car.y - self.targ_y) ** 2)
        prev_spd = self.car.speed
        self.car.update(action)
      
        reward = self.calc_reward(prev_dist, prev_spd)
        done = (not self.car.alive or
                self.steps >= self.max_steps or
                self.reached_targ())
      
        self.ep_reward += reward
        return self.car.get_state(self.targ_x, self.targ_y), reward, done

    def calc_reward(self, prev_dist, prev_spd):
        reward = 0
      
        curr_dist = math.sqrt((self.car.x - self.targ_x) ** 2 + (self.car.y - self.targ_y) ** 2)
        dist_diff = prev_dist - curr_dist
        reward += dist_diff * 0.1
      
        min_sensor = min(self.car.sensors)
        if min_sensor < 0.2:
            reward -= 10 * (0.2 - min_sensor)
          
        if 2 < self.car.speed < 6:
            reward += 2
        elif self.car.speed > 8:
            reward -= 2
        else:
            reward -= 2
          
        if self.car.check_collision():
            reward -= 50
        if self.reached_targ():
            reward += 200
          
        rel_angle = math.atan2(self.targ_y - self.car.y, self.targ_x - self.car.x) - math.radians(self.car.angle)
        rel_angle = math.atan2(math.sin(rel_angle), math.cos(rel_angle))
        angle_bonus = (1 - abs(rel_angle) / math.pi) * 10
        reward += angle_bonus
      
        if curr_dist < 200 and not self.got_200:
            reward += 20
            self.got_200 = True
        if curr_dist < 100 and not self.got_100:
            reward += 30
            self.got_100 = True

      # time penalty (this worked better)
        reward -= 0.1
        return reward

    def reached_targ(self):
        dist = math.sqrt((self.car.x - self.targ_x) ** 2 + (self.car.y - self.targ_y) ** 2)
        return dist < 50

    def render(self, screen):
        screen.fill(WHITE)
        pygame.draw.rect(screen, DARK_GRAY, (0, 0, WIN_W, 50))
        pygame.draw.rect(screen, DARK_GRAY, (0, WIN_H - 50, WIN_W, 50))
        pygame.draw.rect(screen, DARK_GRAY, (0, 0, 50, WIN_H))
        pygame.draw.rect(screen, DARK_GRAY, (WIN_W - 50, 0, 50, WIN_H))
        pygame.draw.circle(screen, GREEN, (int(self.targ_x), int(self.targ_y)), 25)
        self.car.draw(screen)
        font = pygame.font.Font(None, 36)
      
        info_txt = [
            f"Episode Reward: {self.ep_reward:.1f}",
            f"Distance: {math.sqrt((self.car.x - self.targ_x) ** 2 + (self.car.y - self.targ_y) ** 2):.1f}",
            f"Speed: {self.car.speed:.1f}",
            f"Steps: {self.steps}"
        ]
      
        for i, txt in enumerate(info_txt):
            rendered_txt = font.render(txt, True, BLACK)
            screen.blit(rendered_txt, (10, 10 + i * 40))

def main():
    screen = pygame.display.set_mode((WIN_W, WIN_H))
    pygame.display.set_caption("Self-Driving Car RL Sim")
    clock = pygame.time.Clock()
    env = Environment()
    init_state = env.reset()
    state_sz = len(init_state)
    action_sz = 5
  
    print(f"State size: {state_sz}")
    print(f"Action size: {action_sz}")
    print("State components: [x, y, sin(angle), cos(angle), speed, distance_to_target, sin(relative_angle), cos(relative_angle), sensor1, sensor2, sensor3, sensor4, sensor5]")
  
    agent = DQNAgent(state_sz, action_sz)
    episodes = 1000
    ep_rewards = []
    ep_lengths = []
    print("Starting training...")
    print("Controls: Close window to stop training")
    print("Actions: 0=straight, 1=left, 2=right, 3=accelerate, 4=brake")
  
    for ep in range(episodes):
        state = env.reset()
        total_reward = 0
      
        while True:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    pygame.quit()
                    return
            action = agent.act(state)
            next_state, reward, done = env.step(action)
            agent.remember(state, action, reward, next_state, done)
            state = next_state
            total_reward += reward
            if ep % 5 == 0:
                env.render(screen)
                font = pygame.font.Font(None, 24)
                ep_txt = font.render(f"Episode: {ep}/{episodes}", True, BLACK)
                eps_txt = font.render(f"Epsilon: {agent.epsilon:.3f}", True, BLACK)
                screen.blit(ep_txt, (10, WIN_H - 60))
                screen.blit(eps_txt, (10, WIN_H - 40))
                pygame.display.flip()
                clock.tick(FPS)
            if done:
                break
              
        if len(agent.memory) > agent.batch_sz:
            agent.replay()
          
        if ep % 10 == 0:
            agent.update_targ_net()
          
        ep_rewards.append(total_reward)
        ep_lengths.append(env.steps)
      
        if ep % 100 == 0:
            avg_reward = np.mean(ep_rewards[-100:])
            avg_len = np.mean(ep_lengths[-100:])
            print(f"Episode {ep}, Avg Reward: {avg_reward:.2f}, Avg Length: {avg_len:.2f}, Epsilon: {agent.epsilon:.3f}")
          
    print("Training completed!")
    agent.q_net.save("self_driving_car_model.h5")
    print("Model saved as 'self_driving_car_model.h5'")
  
    stats = {
        'episode_rewards': ep_rewards,
        'episode_lengths': ep_lengths
    }
  
    with open('training_stats.json', 'w') as f:
        json.dump(stats, f)
      
    plt.figure(figsize=(12, 4))
    plt.subplot(1, 2, 1)
    plt.plot(ep_rewards)
    plt.title('Episode Rewards')
    plt.xlabel('Episode')
    plt.ylabel('Reward')
    plt.subplot(1, 2, 2)
    plt.plot(ep_lengths)
    plt.title('Episode Lengths')
    plt.xlabel('Episode')
    plt.ylabel('Steps')
    plt.tight_layout()
    plt.savefig('training_results.png')
    plt.show()
    print("\nRunning demo with trained agent...")
    agent.epsilon = 0
  
    for demo_ep in range(5):
        state = env.reset()
        total_reward = 0
      
        while True:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    pygame.quit()
                    return
                  
            action = agent.act(state)
            next_state, reward, done = env.step(action)
            state = next_state
            total_reward += reward
            env.render(screen)
            font = pygame.font.Font(None, 24)
            demo_txt = font.render(f"DEMO - Episode: {demo_ep + 1}/5", True, BLACK)
            reward_txt = font.render(f"Total Reward: {total_reward:.1f}", True, BLACK)
            screen.blit(demo_txt, (10, WIN_H - 80))
            screen.blit(reward_txt, (10, WIN_H - 60))
          
            if env.reached_targ():
                success_txt = font.render("TARGET REACHED!", True, GREEN)
                screen.blit(success_txt, (WIN_W // 2 - 100, WIN_H // 2))
              
            pygame.display.flip()
            clock.tick(FPS)
          
            if done:
                pygame.time.wait(2000)
                break
        print(f"Demo {demo_ep + 1}: Total Reward = {total_reward:.2f}")
    pygame.quit()

if __name__ == "__main__":
    main()
