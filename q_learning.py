# Q learning
import numpy as np
import math, random
import sys, os
from datetime import datetime

import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib import rcParams

from env import Env

# ====== YOU CAN MODIFY THE PARAMETERS HERE ====== #
env = Env(9,6,5,0,1)
total_episode = 20000
total_test = 20
will_render = True
alpha = 0.2 # learning step size
gamma = 0.8 # reward discount rate
epsilon = 0.7 # Exploring rate
max_steps = 100 # Max steps in every episode
# ====== ********************************* ====== #


folder_name = datetime.now().strftime("%Y%m%d%H%M%S")
os.mkdir( folder_name, 0o755 )
this_dir = os.path.dirname(os.path.abspath(__file__)) #<-- absolute dir the script is in
print(this_dir)
# os.chdir(path)

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
            # return np.argmax(Q[s])
            _max = np.max(Q[s])
            possible_actions = []
            for a in range(len(env.action_space)):
                if Q[s][a]==_max:
                    possible_actions.append(a)
            return possible_actions[random.randint( 0, len(possible_actions)-1 )]

    progress = 0

    for episode in range(total_episode):

        s = env.reset()

        for i in range(max_steps):
            a = pick_action(s,epsilon)
            next_state, reward, over = env.step(a) #make a mov from s using a and get the new state s and the reward r    

            Q[s][a] = (1-alpha)*Q[s][a] + alpha*( reward + gamma * Q[next_state].max() )
            s = next_state

            if over:
                break

        if episode*100 / total_episode > progress:
            print(progress, "%")
            progress = progress+1

    file_name = 'trained_model.npz'
    abs_file_path = os.path.join(this_dir, folder_name, file_name)
    np.savez_compressed(abs_file_path, Q)

test_result = 0 # Succesful times

file_name = "testing_log.txt"
abs_file_path = os.path.join(this_dir, folder_name, file_name)
fo = open(abs_file_path, "w+")

for test in range(total_test):
    s = env.reset()
    frames = []
    # print("== Test", test, ":")
    fo.write( "== Test "+str(test)+":\n" )

    for i in range(max_steps):
        if will_render:
            frames.append( env.render(mode = 'rgb_array') )
        a = pick_action(s,0)
        next_state, reward, over = env.step(a) #make a mov from s using a and get the new state s and the reward r    

        fo.write( str(s)+" "+str(a)+"\n" )

        # Q[s][a] = (1-alpha)*Q[s][a] + alpha*( reward + gamma * Q[next_state].max() )
        s = next_state
        if over:
            if reward >= 10:
                test_result += 1
            break

    if will_render:
        file_name = "test-"+str(test)+".gif"
        abs_file_path = os.path.join(this_dir, folder_name, file_name)
        save_frames_as_gif(frames, abs_file_path)

    fo.write( "\n" )

fo.close()

file_name = "report.txt"
abs_file_path = os.path.join(this_dir, folder_name, file_name)
fo = open(abs_file_path, "w+")

fo.write( "# Environment:\n" )
fo.write( "Width: "+str(env.court_width-env.outter_margin*2)+"\n" )
fo.write( "Height: "+str(env.court_height-env.outter_margin*2)+"\n" )
fo.write( "Opponents: "+str(env.opponent_number)+"\n" )
fo.write( "Opponentt moving: "+str(env.opponent_can_move)+"\n" )
fo.write( "Drunk: "+str(env.drunk)+"\n" )
fo.write( "\n" )

fo.write( "# Training setting:\n" )
fo.write( "Alpha: "+str(alpha)+"\n" )
fo.write( "Gamma: "+str(gamma)+"\n" )
fo.write( "Epsilon: "+str(epsilon)+"\n" )
fo.write( "Max steps: "+str(max_steps)+"\n" )
fo.write( "Training episodes: "+str(total_episode)+"\n" )
fo.write( "\n" )

fo.write( "# Test result:\n" )
fo.write( "Test episodes: "+str(total_test)+"\n" )
fo.write( "Success ratio: "+str(test_result*100/total_test)+"%"+"\n")

fo.close()

print("Test over. "+str(test_result)+" success out of "+str(total_test)+"("+str(test_result*100/total_test)+"%)")

