# Eutopia Pacman Contest - Winning Agent

## Overview

This repository contains the implementation of the winning agents for the **Eutopia Pacman Contest**, held in December 2024. The competition featured teams from multiple universities, competing in a sophisticated "Capture the Flag" variant of Pacman. I am proud to share that I placed **1st in the competition**.

## About the Contest

The Eutopia Pacman Contest is a team-based competition where agents, written in Python, navigate a divided Pacman map to:

- Defend their side from enemy Pacmen.
- Steal food from the opponent’s side.
- Leverage power capsules to turn the tide by scaring enemy ghosts.

Participants needed to demonstrate a deep understanding of algorithms, AI techniques, and efficient decision-making under computational constraints. For more details on the contest rules and framework, refer to the official [documentation](https://github.com/aig-upf/pacman-contest).

## My Winning Strategy

### Agents Overview

My team comprised two custom-designed agents:

1. **Offensive  Agent**

   - Focused on infiltrating enemy territory to collect food and power capsules efficiently.
   - Leveraged strategic pathfinding, power capsule utilization, and food prioritization.

2. **Defensive  Agent**

   - Patrolled our side to prevent enemy Pacmen from stealing food.
   - Dynamically adjusted patrol points based on food and capsule locations.
   - Prioritized chasing visible Pacmen and predicting potential infiltrations.

### Key Features

- **Dynamic Decision-Making**: Both agents used a combination of heuristic-based evaluations and forward simulations to predict and counter opponent actions.
- **Power Capsule Exploitation**: Offensively, the agents planned routes to maximize food collection after consuming a power capsule. Defensively, they ensured minimal exposure to enemy Pacmen.
- **Patrolling Strategy**: The defensive agent dynamically cycled through patrol points based on the distribution of food and capsules, adapting to the evolving game state.
- **Simulation-Based Evaluation**: The offensive agent utilized forward simulations to predict long-term outcomes of specific actions, balancing immediate rewards with future gains.

## Results

### Tournament Highlights

- **Final Rank**: 1st place

## Acknowledgments

This project builds upon the Pacman AI framework developed at UC Berkeley and extended by RMIT and UPF. I am grateful to the organizers and participants of the Eutopia Pacman Contest for creating such an intellectually stimulating competition.

---

Feel free to explore the code and use it as inspiration for your own AI projects. Contributions and feedback are welcome!

**Potential improvement**:

- Parametrise all the important constants used in my code and build a Genetic Algorithm that finds a more optimal solution through a tournament based selection and breeding.

