# Self-Driving Car RL Simulation

This project implements a reinforcement learning (RL) simulation where a car navigates to a randomly placed target while avoiding walls, using a Deep Q-Network (DQN). Built with Python, Pygame for visualization, and TensorFlow for the RL model, the car reliably reaches the target after approximately 250 training episodes, as demonstrated in the video below.

[View Demo Video](https://drive.google.com/drive/folders/1FSgIrCrJrTQtGxWXs4pCFWSo4zQhnNBw?usp=drive_link)


## Overview

The simulation features a car with five sensors to detect walls, trained over 1000 episodes to optimize actions (go straight, turn left/right, accelerate, brake). Pygame displays the car, sensors, target, and real-time metrics. Training results are saved as a model, statistics, and performance plots.

## Features

- Five sensors for wall detection.
- DQN-based learning for navigation.
- Real-time Pygame visualization (1200x800 window).
- Outputs: trained model (`self_driving_car_model.h5`), stats (`training_stats.json`), and plots (`training_results.png`).
- Demo mode showcasing the trained model.
- Target reached consistently after ~250 episodes.

## Prerequisites

- Python 3.8+
- Dependencies: `numpy`, `pygame`, `tensorflow`, `matplotlib`

Install dependencies:
```bash
pip install -r requirements.txt
```

## Installation

1. Clone the repository:
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

## Usage

- **Training**: The car trains for 1000 episodes, with progress logged every 100 episodes. The car learns to reach the target by ~250 episodes.
- **Demo**: Post-training, five demo episodes display the car’s performance.
- **Controls**: Close the Pygame window to stop training or demo.
- **Outputs**:
  - `self_driving_car_model.h5`: Trained DQN model.
  - `training_stats.json`: Episode rewards and lengths.
  - `training_results.png`: Performance plots.

## Demo Video

The video at [Google Drive](https://drive.google.com/drive/folders/1FSgIrCrJrTQtGxWXs4pCFWSo4zQhnNBw?usp=drive_link) shows the car reaching the target after ~250 episodes, demonstrating effective learning.

## Notes

- Training duration varies by hardware (expect a few minutes for 1000 episodes).
- The car (red if active, gray if crashed) navigates a 1200x800 window with gray wall borders.
- Modify `Environment` or `DQNAgent` classes in `self_driving_car.py` to adjust rewards or model architecture.

## Contributing

Contributions are welcome. Please submit issues or pull requests to enhance the simulation.

## License

MIT License. See `LICENSE` for details.

## Acknowledgments

Developed using Pygame for visualization and TensorFlow for reinforcement learning.
