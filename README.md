# Self-Driving Car RL Simulation

Hey there! This is a fun little project I built to mess around with reinforcement learning using a DQN (Deep Q-Network) to train a car to navigate toward a target while avoiding walls. It’s written in Python, uses Pygame for the visuals, and TensorFlow for the neural network. The car learns to steer, accelerate, and brake based on sensor inputs and its position relative to a randomly placed target.

## Features

- A car with 5 sensors to detect walls (like a simplified LIDAR).
- DQN agent that learns to drive toward a green target circle.
- Visualizes training with Pygame and saves plots of rewards and episode lengths.
- Saves the trained model and stats for later use.
- Includes a demo mode to watch the trained car in action.

## Prerequisites

You’ll need Python 3.8+ and the following libraries:
- `numpy`
- `pygame`
- `tensorflow`
- `matplotlib`

Install them with:
```bash
pip install -r requirements.txt
```

## Setup

1. Clone this repo:
   ```bash
   git clone https://github.com/yourusername/self-driving-car-rl.git
   cd self-driving-car-rl
   ```

2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

3. Run the simulation:
   ```bash
   python self_driving_car.py
   ```

## How It Works

- The car starts at a random position and tries to reach a green target.
- It uses 5 sensors to detect walls and avoid collisions.
- The DQN learns by trial and error, balancing exploration (random moves) and exploitation (using the trained model).
- Training runs for 1000 episodes, with progress printed every 100 episodes.
- After training, it saves:
  - `self_driving_car_model.h5`: The trained neural network.
  - `training_stats.json`: Episode rewards and lengths.
  - `training_results.png`: Plots of training performance.
- A demo runs 5 episodes to show off the trained car.

## Controls

- Close the Pygame window to stop training or the demo.
- Actions the car can take: go straight, turn left, turn right, accelerate, brake.

## Output Files

- `self_driving_car_model.h5`: The trained DQN model.
- `training_stats.json`: JSON file with episode rewards and lengths.
- `training_results.png`: Plot of training performance.

## Notes

- The car might take a while to learn (1000 episodes can take some time depending on your hardware).
- The simulation window (1200x800) shows the car, sensors, target, and some stats during training.
- If the car crashes into walls (gray borders), it gets a penalty and the episode ends.

## Contributing

Got ideas to make this better? Feel free to open an issue or submit a pull request on GitHub!

## License

This project is licensed under the MIT License. See the `LICENSE` file for details.
