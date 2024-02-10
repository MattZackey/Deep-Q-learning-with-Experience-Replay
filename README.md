# Q-learning with Cartpole

The following is an example of applying Q-learning to the Cart Pole environment from OpenAI gymnasium. The theory in this project is from the paper: [Playing Atari with Deep Reinforcement Learning](https://arxiv.org/pdf/1312.5602.pdf). Note that the only difference in this project is that a [soft target update](https://arxiv.org/pdf/1509.02971.pdf?source=post_page---------------------------) was used to update the target network. The soft target update was found to improve learning speed. 

# Training results

The following figure shows the amount of time the agent can balance the pole. Note that the environment is truncated at 500 time steps.  

![Results](https://github.com/MattZackey/Q-learning-with-Cartpole/blob/main/Training%20results.png?raw=true)

