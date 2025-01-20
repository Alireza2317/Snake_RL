# Reinforcement Learning Snake Game 🐍

This project is an implementation of a **Reinforcement Learning (RL) agent** to play the classic Snake game. The RL agent should learn to navigate, avoid obstacles, and collect food by maximizing rewards in a grid world.

## Features
- **Customizable RL Agent**: Train the agent using a neural network for decision-making.
- **Interactive Game Environment**: Playable GUI and non-GUI versions of the Snake game.
- **Live Progress Visualization**: Monitor training progress with real-time plots.

## Installation

### Step 1: Clone the Repository
```bash
git clone https://github.com/Alireza2317/Snake_RL
cd rl-snake-game
```
### Step 2: Create virtual environment (optional but recommended)
```bash
python -m venv env
source env/bin/activate  # On Windows: env\Scripts\activate
```
### Step 3: Install dependencies
```bash
pip install -r requirements.txt
```

### Step 4: Run and train the agent, and watch it play
```bash
python app.py
```
The code will first train the agent with the default configs and then play the game visually.
Feel free to change the `configs` dictionary to change the outcome.

## Project structure
```
.
├── app.py                 # Main entry point for training and playing the game
├── snake.py               # Snake game logic and environment
├── agent.py               # Reinforcement Learning agent implementation
```

Happy coding :)