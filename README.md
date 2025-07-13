Inverted Pendulum is a classic reinforcement learning problem which we have solved using DQN algorithm.

LOGIC BEHIND INVERTED PENDULUM PROBLEM

State (Observation): [cart position, cart velocity, pole angle,
pole angular velocity]
Action Space: 0 = push left, 1 = push right
Reward: +1 per time step the pole stays upright (CartPole).
Termination: 1)Pole falls too far (angle threshold crossed)
2)Cart moves off-screen
 3)Max steps reached
Physics Logic: The system uses Newton’s laws to update
motion. Pole tends to fall due to gravity, and the agent must learn
how to balance it by moving the cart left or right.
RL Loop: 1)Observe current state
 2)Take action (force)
 3)Receive next state + reward
 4)Repeat to learn a policy that balances the pole
