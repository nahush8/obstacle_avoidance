"""
Once a model is learned, use this to play it.
"""

import gameEngine
import numpy as np
from nn import neural_net
import matplotlib.pyplot as plt

NUM_SENSORS = 3
plt.ion()

def play(model):
    iteration = 0
    sum_of_reward_per_epoch = 0
    game_state = gameEngine.GameState()

    # Do nothing to get initial.
    _, state = game_state.frame_step((2))

    # Move.
    while True:
        car_distance += 1

        # Choose action.
        action = (np.argmax(model.predict(state, batch_size=1)))

        # Take action.
        curr_reward, state = game_state.frame_step(action)
        iteration += 1
        sum_of_reward_per_epoch += curr_reward
        plt.scatter(epoch,sum_of_reward_per_epoch)
        if iteration % 200 == 0:
            with open(timestr + '_deepQ', 'a') as fp:
                fp.write(str(sum_of_reward_per_epoch) + '\n')
                fp.flush()
            fp.close()
        sum_of_reward_per_epoch = 0
        epoch += 1
        plt.pause(0.05)

if __name__ == "__main__":
    saved_model = 'saved-models/164-150-100-50000-25000.h5'
    model = neural_net(NUM_SENSORS, [164, 150], saved_model)
    play(model)