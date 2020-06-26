import gym
import random
import numpy as np
from IPython.display import clear_output
from time import sleep

#   We want to build a self-driving taxi that can pick up passengers at one of a set of fixed locations,
#  drop them off at another location and get there in the quickest amount of time while avoiding obstacles.

random.seed(0)

streets = gym.make("Taxi-v3").env
streets.render()

# R, G, B, and Y are pickup or dropoff locations.
# The BLUE letter indicates where we need to pick someone up from.
# The MAGENTA letter indicates where that passenger wants to go to.
# The solid lines represent walls that the taxi cannot cross.
# The filled rectangle represents the taxi itself - it's yellow when empty, and green when carrying a passenger.

# "streets", is a 5x5 grid. The state of this world at any time can be defined by:
# Where the taxi is (one of 5x5 = 25 locations)
# What the current destination is (4 possibilities)
# Where the passenger is (5 possibilities: at one of the destinations, or inside the taxi)
# So there are a total of 25 x 4 x 5 = 500 possible states that describe our world.

# For each state, there are six possible actions:
# Move South, East, North, or West
# Pickup a passenger
# Drop off a passenger

# Q-Learning will take place using the following rewards and penalties at each state:
# A successfull drop-off yields +20 points
# Every time step taken while driving a passenger yields a -1 point penalty
# Picking up or dropping off at an illegal location yields a -10 point penalty
# Moving across a wall isn't allowed.

# Let's define an initial state, with the taxi at location (2, 3), the passenger at pickup location 2, and the
# destination at location 0:
initial_state = streets.encode(2, 3, 2, 0)
streets.s = initial_state
streets.render()

print(streets.P[initial_state])

# each row corresponds to a potential action at this state: move South, North, East, or West, pickup, or dropoff. The
# four values in each row are the probability assigned to that action, the next state that results from that action,
# the reward for that action, and whether that action indicates a successful dropoff took place.
# So for example, moving North from this state would put us into state number 368, incur a penalty of -1 for taking up
# time, and does not result in a successful dropoff.

#  First we need to train our model. We'll train over 10,000 simulated taxi runs.
#  For each run, we'll step through time, with a 10% chance at each step of making a random, exploratory step instead of
#  using the learned Q values to guide our actions.

q_table = np.zeros([streets.observation_space.n, streets.action_space.n])
# streets.observation_space.n gives us the state we are currently in out of 500 states
# streets.action_space.n gives us the points for every action initially assigned to 0
print(q_table)

learning_rate = 0.1
discount_factor = 0.6
exploration = 0.1  # probability to explore a random action
epochs = 10000

for taxi_run in range(epochs):
    state = streets.reset()
    # randomize taxi location, dropoff and pickup location
    done = False

    while not done:
        random_value = random.uniform(0, 1)
        if random_value < exploration:
            action = streets.action_space.sample()  # Explore a random action
        else:
            action = np.argmax(q_table[state])  # Use the action with the highest q-value
            # table consists of 6 values corresponding to each action on that state

        next_state, reward, done, info = streets.step(action)
        # next state gives state number out of 500 states
        # reward gives points gained/lost on next action
        # done gives false until dropoff successful
        # info gives probability as 1 in all cases

        # assigning value to every action based on next moves by using a formula
        prev_q = q_table[state, action]
        next_max_q = np.max(q_table[next_state])
        new_q = (1 - learning_rate) * prev_q + learning_rate * (reward + discount_factor * next_max_q)
        q_table[state, action] = new_q

        state = next_state

print(q_table[initial_state])
# these are values for initial actions
# most positive action is done in this case "go West"(index - 3)

# for visualization
for tripnum in range(1, 11):
    state = streets.reset()
    # new case every iteration

    done = False
    trip_length = 0

    while not done and trip_length < 25:  # trip length cannot be more than 25 or it will be wrong
        action = np.argmax(q_table[state])
        print(action)
        # action gives us number of element of array with highest value
        # for example 1 means north, 0 means south etc.
        next_state, reward, done, info = streets.step(action)  # values for next step
        clear_output(wait=True)
        print("Trip number " + str(tripnum) + " Step " + str(trip_length))
        print(streets.render(mode='ansi'))
        sleep(.5)
        state = next_state
        trip_length += 1

    sleep(2)