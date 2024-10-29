# PokerPPO
# PokerRL-PPO-Agent

![PokerRL-PPO-Agent](https://github.com/YourUsername/PokerRL-PPO-Agent/blob/main/assets/poker_image.png)

## Table of Contents

- [Overview](#overview)
- [Features](#features)
- [Prerequisites](#prerequisites)
- [Installation](#installation)
- [Usage](#usage)
  - [Training the PPO Agent](#training-the-ppo-agent)
  - [Evaluating the Agent](#evaluating-the-agent)
- [Configuration](#configuration)
- [Project Structure](#project-structure)
- [Results](#results)
- [Contributing](#contributing)
- [License](#license)
- [Acknowledgements](#acknowledgements)
- [Contact](#contact)

## Overview

**PokerRL-PPO-Agent** is a reinforcement learning project that leverages the Proximal Policy Optimization (PPO) algorithm to train an agent capable of playing poker efficiently. Built upon the [PokerRL](https://github.com/EricSteinberger/PokerRL) environment, this project demonstrates how advanced RL techniques can be applied to strategic card games, offering insights into agent behavior, strategy development, and performance evaluation.

## Features

- **Proximal Policy Optimization (PPO)**: Utilizes the PPO algorithm from Stable Baselines3 for stable and efficient training.
- **Parallel Environments**: Supports multi-processing to accelerate training using multiple environments simultaneously.
- **Model Checkpointing**: Periodically saves model checkpoints to prevent data loss and facilitate analysis.
- **Evaluation Callbacks**: Continuously evaluates the agent's performance against predefined benchmarks during training.
- **Reproducibility**: Ensures consistent results by setting seeds across all random number generators.
- **GPU Support**: Automatically leverages GPU resources if available for faster computations.

## Prerequisites

Before getting started, ensure you have the following installed on your system:

- **Python 3.7 or later**: [Download Python](https://www.python.org/downloads/)
- **Git**: [Download Git](https://git-scm.com/downloads)
- **Virtual Environment (Recommended)**: To manage dependencies without affecting system-wide packages.

## Installation

1. **Clone the Repository**

   ```bash
   git clone https://github.com/YourUsername/PokerRL-PPO-Agent.git
   cd PokerRL-PPO-Agent
   ```

2. **Create and Activate a Virtual Environment**

   It's recommended to use a virtual environment to manage dependencies:

   ```bash
   # Create a virtual environment named 'venv'
   python3 -m venv venv

   # Activate the virtual environment
   # On Unix or MacOS
   source venv/bin/activate

   # On Windows
   venv\Scripts\activate
   ```

3. **Install Dependencies**

   Install the required Python packages using `requirements.txt`:

   ```bash
   pip install -r requirements.txt
   ```

   If `requirements.txt` is not provided, manually install the necessary packages:

   ```bash
   pip install gym
   pip install stable-baselines3
   pip install torch  # Ensure PyTorch is installed; adjust based on your system and CUDA availability
   ```

4. **Install PokerRL**

   Clone and install the PokerRL package:

   ```bash
   git clone https://github.com/EricSteinberger/PokerRL.git
   cd PokerRL
   pip install -e .
   cd ..
   ```

5. **Verify Installation**

   To ensure that everything is set up correctly, run the following Python script:

   ```python
   import gym
   import poker_rl  # Replace with the actual module name if different

   env = gym.make('PokerRL-v0')  # Replace with the correct environment ID if different
   obs = env.reset()
   print(obs)
   env.close()
   ```

   **Note:** Replace `'PokerRL-v0'` with the actual environment ID as defined in the `PokerRL` package.

## Usage

### Training the PPO Agent

1. **Configure Training Parameters**

   You can adjust training parameters such as total timesteps, learning rate, number of environments, etc., in the `train_poker_agent.py` script.

2. **Run the Training Script**

   Execute the training script to start training the PPO agent:

   ```bash
   python train_poker_agent.py
   ```

   This script will:

   - Initialize multiple parallel environments.
   - Train the PPO agent for the specified number of timesteps.
   - Save model checkpoints periodically.
   - Evaluate the agent's performance during training.
   - Save the final trained model.

3. **Monitor Training Progress**

   - **TensorBoard**: If enabled, monitor training metrics in real-time.

     ```bash
     tensorboard --logdir=./tensorboard/
     ```

     Open the provided URL in your browser to visualize metrics like reward, loss, etc.

   - **Saved Models**: Check the `models/` directory for saved checkpoints and the best-performing model.

### Evaluating the Agent

After training, evaluate the performance of the trained agent using the `test_agent` function included in the training script.

1. **Run the Evaluation**

   The `test_agent` function is automatically called at the end of the training script. It will:

   - Load the trained model.
   - Run the agent for a specified number of episodes.
   - Print the total reward for each episode.

2. **Sample Output**

   ```
   Episode 1: Total Reward = 150
   Episode 2: Total Reward = 200
   Episode 3: Total Reward = 180
   Episode 4: Total Reward = 220
   Episode 5: Total Reward = 190
   ```

   *Note:* Actual rewards will depend on the specifics of the `PokerRL` environment and the training process.

## Configuration

You can customize various aspects of the training and evaluation process by modifying the `train_poker_agent.py` script:

- **Environment ID**: Ensure the correct environment ID is used (`ENV_ID = 'PokerRL-v0'`).
- **Training Parameters**: Adjust `TOTAL_TIMESTEPS`, `LEARNING_RATE`, `NUM_ENVS`, etc., to suit your computational resources and desired training duration.
- **Callbacks**: Configure checkpointing frequency, evaluation intervals, and reward thresholds as needed.
- **Device Selection**: The script automatically selects GPU if available; you can override this by setting the `device` parameter in the PPO model.

## Project Structure

```
PokerRL-PPO-Agent/
├── assets/
│   └── poker_image.png
├── models/
│   ├── best_model/
│   └── ppo_poker_final.zip
├── logs/
│   └── evaluation/
├── PokerRL/  # Cloned PokerRL repository
├── train_poker_agent.py
├── requirements.txt
├── README.md
└── LICENSE
```

- **assets/**: Contains images or other media used in the README.
- **models/**: Stores trained model checkpoints and the final model.
- **logs/**: Contains evaluation logs.
- **PokerRL/**: The PokerRL environment cloned and installed.
- **train_poker_agent.py**: Main script for training and evaluating the PPO agent.
- **requirements.txt**: Lists all Python dependencies.
- **README.md**: This documentation file.
- **LICENSE**: Project licensing information.

## Results

After training, the PPO agent demonstrates proficiency in playing poker within the `PokerRL` environment. Evaluation metrics indicate consistent performance improvements over time, with the agent achieving substantial rewards across multiple episodes.

**Sample Evaluation Results:**

```
Episode 1: Total Reward = 150
Episode 2: Total Reward = 200
Episode 3: Total Reward = 180
Episode 4: Total Reward = 220
Episode 5: Total Reward = 190
```

*Note:* Results may vary based on hyperparameter configurations and the complexity of the environment.

## Contributing

Contributions are welcome! Whether it's reporting bugs, suggesting enhancements, or submitting pull requests, your input is valuable.

1. **Fork the Repository**

2. **Create a New Branch**

   ```bash
   git checkout -b feature/YourFeature
   ```

3. **Commit Your Changes**

   ```bash
   git commit -m "Add Your Feature"
   ```

4. **Push to Your Fork**

   ```bash
   git push origin feature/YourFeature
   ```

5. **Open a Pull Request**

   Provide a clear description of your changes and the motivation behind them.

## License

This project is licensed under the [MIT License](LICENSE).

## Acknowledgements

- **[PokerRL](https://github.com/EricSteinberger/PokerRL)**: The foundational environment for training the poker agent.
- **[Stable Baselines3](https://github.com/DLR-RM/stable-baselines3)**: Provides the PPO implementation and other RL algorithms.
- **[OpenAI Gym](https://gym.openai.com/)**: Facilitates the creation and interaction with reinforcement learning environments.
- **[PyTorch](https://pytorch.org/)**: The deep learning framework used for model computations.

## Contact

For any inquiries or support, please contact:

- **Your Name**
- **Email**: your.email@example.com
- **LinkedIn**: [Your LinkedIn Profile](https://www.linkedin.com/in/yourprofile/)
- **GitHub**: [Your GitHub Profile](https://github.com/YourUsername)

---

Happy Training! 🎉
