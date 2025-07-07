# Self-Driving Car RL Simulation 🚗🎯

Hey there! Welcome to my self-driving car project—a Python-based reinforcement learning (RL) simulation where a car learns to chase a green target while avoiding walls, powered by a Deep Q-Network (DQN). I built this using Pygame for the visuals and TensorFlow for the RL magic. The best part? After ~250 episodes, the car learns to hit the target almost every time! Check out the demo video on Google Drive to see it in action:

[Watch the Demo Video](https://drive.google.com/drive/folders/1FSgIrCrJrTQtGxWXs4pCFWSo4zQhnNBw?usp=drive_link)

*Pro tip: If the video link doesn’t work, ensure it’s set to “Anyone with the link” in Google Drive. Alternatively, look for `self_driving_car_demo.mp4` in the shared folder.*

## Why This Project Rocks

This simulation is my take on teaching a car to drive itself using RL. The car uses five sensors to “see” walls, learns to steer and control speed, and gets really good at reaching the target by episode ~250, as shown in the demo video. It’s a fun way to explore DQN-based learning, and the Pygame visuals make it satisfying to watch!

## Features

- **Smart Car**: Five sensors detect walls (like a mini-LIDAR system).
- **DQN Brain**: Learns optimal actions (go straight, turn, accelerate, brake) via a Deep Q-Network.
- **Real-Time Visuals**: Pygame shows the car, sensors, target, and live stats (reward, distance, speed).
- **Training Insights**: Saves performance plots (`training_results.png`) and stats (`training_stats.json`).
- **Demo Mode**: Post-training, the car shows off its skills in 5 demo episodes.
- **Fast Learner**: Reliably hits the target after ~250 episodes, as seen in the video!

## Prerequisites

You’ll need:
- Python 3.8+
- Libraries: `numpy`, `pygame`, `tensorflow`, `matplotlib`

Install them with:
```bash
pip install -r requirements.txt
```

## Setup

1. Clone the repo:
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

- **Setup**: The car starts at a random position, with a green target placed at least 300 pixels away.
- **Training**: Runs for 1000 episodes, with the DQN balancing random exploration and learned actions. Progress is printed every 100 episodes.
- **Performance**: By ~250 episodes, the car consistently reaches the target (check the video!).
- **Output Files**:
  - `self_driving_car_model.h5`: Trained DQN model.
  - `training_stats.json`: Episode rewards and lengths.
  - `training_results.png`: Plots of training performance.
- **Demo**: After training, 5 demo episodes show the car navigating to the target.

## Controls

- Close the Pygame window to stop training or demo.
- Actions: Go straight (0), turn left (1), turn right (2), accelerate (3), brake (4).

## Demo Video

The video at [Google Drive](https://drive.google.com/drive/folders/1FSgIrCrJrTQtGxWXs4pCFWSo4zQhnNBw?usp=drive_link) shows the car in action after training. By ~250 episodes, it’s dodging walls and hitting the target like a pro. Watch it to see the RL learning process pay off!

*Note: If you have a direct video link (e.g., `https://drive.google.com/file/d/VIDEO_ID/view?usp=sharing`), let me know, and I can update this README for a cleaner link.*

## Output Files

- `self_driving_car_model.h5`: The trained neural network.
- `training_stats.json`: Training metrics in JSON format.
- `training_results.png`: Graphs of rewards and episode lengths.

## Tips for Running

- Training might take a few minutes (1000 episodes, depending on your machine).
- The Pygame window (1200x800) shows the car (red if alive, gray if crashed), yellow sensors, and the green target.
- Crashes into gray walls end the episode with a penalty.
- Want to tweak things? Check out the `Environment` class for reward logic or `DQNAgent` for network settings in `self_driving_car.py`.

## Contributing

Got ideas to make the car smarter or add obstacles? Open an issue or send a pull request—I’d love to collaborate!

## License

Licensed under the MIT License. See `LICENSE` for details.

## Acknowledgments

- Big thanks to Pygame for making visuals easy and TensorFlow for powering the DQN.
- Built this to learn RL and have some fun—hope you enjoy it too!

*Happy watching, and let the car drive!*
