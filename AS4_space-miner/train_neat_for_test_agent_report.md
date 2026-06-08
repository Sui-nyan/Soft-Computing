# Report: Fitness Design Experiment for Space Miner

## Overview

This experiment trained a NEAT neural-network agent to play the Space Miner game. The agent controls a ship that must collect minerals, avoid asteroids, manage fuel, and survive as long as possible. The main training file is `train_neat_for_test_agent.py`, with experiment constants defined in `train_neat_for_test_agent_config.py`.

The goal of the experiment was not only to produce a working agent, but also to understand how different fitness-function designs affect learning. Several reward and penalty terms were tested, including mineral rewards, mineral progress rewards, fuel rewards, asteroid danger penalties, idle penalties, wasted mining penalties, and collision penalties.

The most important result was that the highest scoring agent came from a relatively simple fitness function. Adding more detailed rewards and penalties helped explain behavior, but after a certain point it made the training result worse.

## Training Setup

The agent is evolved with NEAT. Each genome becomes a feed-forward neural network that receives the game state and outputs actions for the ship. The ship can turn, thrust, and mine. During evaluation, each genome plays one episode of Space Miner, and the resulting episode statistics are converted into a fitness score.

The base score used in the experiment is:

```python
test_score = (alive_time / 4) + (ship.minerals * 100)
```

This score directly rewards the two most important objectives:

- staying alive
- collecting minerals

Mineral collection is weighted much more strongly than survival. This is important because an agent that only survives without mining is not actually solving the task.

## Final Fitness Function

The best-performing version kept the fitness function simple:

```python
fitness = test_score * TEST_SCORE_WEIGHT
fitness += mineral_progress * MINERAL_PROGRESS_WEIGHT
fitness += mineral_approach * MINERAL_APPROACH_WEIGHT
fitness -= asteroid_danger * ASTEROID_DANGER_WEIGHT
```

This means the active fitness function used four main ideas:

- reward the real game score
- reward long-term progress toward minerals
- reward frame-by-frame movement toward minerals
- penalize dangerous closeness to asteroids

Other possible terms were implemented or tested, but were disabled in the final version:

- mineral heading alignment
- mineral velocity alignment
- fuel efficiency reward
- remaining fuel reward
- idle penalty
- wasted mining penalty
- explicit asteroid collision penalty
- out-of-fuel penalty

These terms seemed reasonable individually, but combining too many of them made the optimization problem harder for NEAT.

## Observation 1: Same Configuration Can Train Differently

One important observation was that training runs were not perfectly repeatable, even when the same configuration was used. This is expected for NEAT because evolution depends on random initialization, mutation, crossover, species formation, and the order in which good structures are discovered.

As a result, two runs with the same parameters can produce different agents. One run might quickly discover a useful mineral-seeking behavior, while another run might get stuck in a weaker strategy such as drifting, turning inefficiently, or surviving without collecting many minerals.

Because of this, a single training run is not enough to judge whether a fitness design is good. The score trend, best genome behavior, and repeated runs all need to be considered. A configuration that occasionally produces a good agent is less reliable than one that consistently guides the population toward useful behavior.

## Observation 2: More Rewards and Penalties Can Make Performance Worse

At first, it seemed natural to add more rewards and penalties for every behavior we wanted:

- reward fuel efficiency
- reward remaining fuel
- reward alignment with minerals
- penalize idling
- penalize wasted mining
- penalize asteroid collisions
- penalize running out of fuel

However, adding more shaping terms did not always improve performance. At a certain point, the agent started optimizing the shaping terms instead of the real task. For example, strong fuel rewards can make the agent too conservative, while idle or wasted-action penalties can discourage exploration during early learning. Collision and danger penalties can also make the agent overly cautious in a crowded map.

This made the fitness landscape noisier. Instead of receiving a clear signal that minerals are the main goal, the agent had to balance many smaller signals that sometimes conflicted with each other. The result was lower performance, even though the fitness function looked more complete.

## Observation 3: A Simple Fitness Function Scored Highest

The highest scoring setup was the simpler fitness function. It focused on the real objective, then added only enough shaping to help the agent reach minerals and avoid obvious asteroid danger.

This worked better because the learning signal was easier to interpret:

- collecting minerals gives a large reward
- surviving gives useful but smaller reward
- moving toward minerals gives partial credit before mining succeeds
- getting too close to asteroids is discouraged

The simple function did not try to describe every possible good behavior. Instead, it rewarded the outcome that mattered and used only a few supporting terms. This made it easier for NEAT to discover agents that actually play the game well.

## Observation 4: Weight Adjustment Helps Set Starting Behavior

Although the final fitness function was simple, adjusting weights was still useful. The weights changed the early behavior that the agent learned first.

For example, increasing mineral approach or mineral progress made early agents more likely to move toward minerals, even before they learned to mine reliably. Increasing asteroid danger penalties made the agents more cautious around asteroids. Increasing survival or fuel-related rewards made agents more conservative.

This shows that weights are useful for shaping the starting direction of learning. They can help the population avoid completely random or passive behavior in early generations. However, if the weights become too strong or too numerous, they can pull the agent away from the main objective.

## Discussion

The experiment showed that fitness design is a balance between guidance and over-control. A sparse reward such as mineral collection can be too difficult at the start, because early agents may rarely collect minerals. Some shaping is helpful because it gives partial credit for moving in the right direction.

However, too much shaping can become harmful. When many rewards and penalties are active at the same time, the agent may learn behavior that satisfies the fitness formula without achieving the real goal. This is especially important in neuroevolution, where the algorithm does not understand the task directly. It only follows the numeric fitness value.

The best approach was to keep the main objective dominant and add only a small number of shaping terms that directly support it.

## Conclusion

The experiment found that simpler fitness design produced the best Space Miner agent. Training remained stochastic, so repeated runs with the same configuration could still produce different results. Adding rewards and penalties helped explore behavior, but too many terms reduced performance by making the learning signal less clear.

The final lesson is that reward shaping should be used carefully. Weight adjustment is helpful for guiding the initial behavior of the agent, but the fitness function should stay focused on the actual task: collect minerals, survive, and avoid asteroids.
