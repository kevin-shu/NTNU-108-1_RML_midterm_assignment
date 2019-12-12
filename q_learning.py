# Q learning
import numpy as np
import random
import sys
from datetime import datetime

import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib import rcParams

from env import Env

env = Env(18,12,50,0,1)
total_episode = 100000
total_test = 10
alpha = 0.7 # learning step size
gamma = 0.8 # reward discount rate
epsilon = 0.5 # Exploring rate
# lamb = 0.7 # the lambda weighting factor

def save_frames_as_gif(frames,file_name="animation.gif"):
    """
    Displays a list of frames as a gif, with controls
    """
    plt.figure(figsize=(frames[0].shape[1] / 72.0, frames[0].shape[0] / 72.0), dpi = 72)
    patch = plt.imshow(frames[0])
    plt.axis('off')

    def animate(i):
        patch.set_data(frames[i])

    anim = animation.FuncAnimation(plt.gcf(), animate, frames = len(frames), interval=50)
    anim.save(file_name, writer='imagemagick', fps=10)

def get_file_name():
    file_name = (   
                    str(env.court_width-env.outter_margin*2)+"_"+
                    str(env.court_height-env.outter_margin*2)+"_"+
                    str(env.opponent_number)+"_"
                )

    if env.opponent_can_move==1:
        file_name = file_name+"opponents-can-move_"
    else:   
        file_name = file_name+"opponents-can-not-move_"
        
    if env.drunk:
        file_name = file_name+"drunk_"
    else:   
        file_name = file_name+"sober_"

    file_name = file_name+"a"+str(alpha)+"_"
    file_name = file_name+"g"+str(gamma)+"_"
    file_name = file_name+"e"+str(epsilon)+"_"
                    
    file_name = file_name+"trained-"+str(total_episode)+"-episodes"
    
    return file_name

trained_model_path = None
if len(sys.argv)>1:
    trained_model_path = sys.argv[0]

dimension = [env.court_width, env.court_height, env.court_width, env.court_height]
if env.opponent_can_move == 1:
    for i in range(env.opponent_number):
        dimension.append(env.court_width)
        dimension.append(env.court_height)
dimension.append( len(env.action_space) )

if trained_model_path:
    dict_data = np.load(trained_model_path)
    Q = dict_data['arr_0'] # extract the first array

    if Q.shape != tuple(dimension):
        raise NameError("This model don't match")

else:
    print("Start training...")
    Q = np.zeros( dimension )
    # eligibility = np.zeros( dimension )

    def pick_action(s, epsilon): # s = (x,y,holding_ball)
        max_value = -999999

        is_exploring = random.random() < epsilon

        if is_exploring: # if exploring, randomly pick a action
            return random.randint( 0, len(env.action_space)-1 )
        else: # if not, pick a best action base on Q
            return np.argmax(Q[s])

    progress = 0

    for episode in range(total_episode):

        s = env.reset()

        for i in range(60):
            a = pick_action(s,epsilon)
            next_state, reward, over = env.step(a) #make a mov from s using a and get the new state s and the reward r    

            Q[s][a] = (1-alpha)*Q[s][a] + alpha*( reward + gamma * Q[next_state].max() )
            s = next_state

            if over:
                break

        if episode*100 / total_episode > progress:
            print(progress, "%")
            progress = progress+1

    np.savez_compressed('Q_'+get_file_name()+'.npz', Q)


for test in range(total_test):
    s = env.reset()
    frames = []
    # print("== Test", test, ":")
    for i in range(60):
        # env.render()
        frames.append( env.render(mode = 'rgb_array') )
        a = pick_action(s,0)
        next_state, reward, over = env.step(a) #make a mov from s using a and get the new state s and the reward r    

        Q[s][a] = (1-alpha)*Q[s][a] + alpha*( reward + gamma * Q[next_state].max() )
        s = next_state
        if over:
            break

    file_name = get_file_name()+"_test-"+str(test)+".gif"

    save_frames_as_gif(frames, file_name)

