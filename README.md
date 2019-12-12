
# How to run the program
In the terminal, switch to the folder of the project, and then:
``` 
# Install required modules: 
pip install -r requirements.txt

# Run Q-learning:
python q_learning.py
```
After that, this program will generate se

If you want to load the trained model, assign the trained numpy array compressed file as follow:
```
python q_learning.py 36_24_250_trained-1000000-times_drunk_opponents-can-not-move_q-matrix.npz
```

You should notice that, to load the trained model , the numpy version should be lower than 1.16.2. 
```
pip install numpy==1.16.2
```

# Features:
1. Can save trained model into file and load it later.
2. If you choose to render the result, It would also generate the gif of test episodes.
3. The programe will also generate the report file to record the setting and test result.

# About the basketball game enviroment

Env( court_width, court_height, opponent_number=0, opponent_can_move=0, drunk_mode=0 )

Parameters:

* court_width: Integer, The width of court
* court_width: Integer, The height of court
* opponent_number: Integer, The number of opponent in this scenario
* opponent_can_move: Boolean, if it's True, opponent will be randomly distributed in every episode.
* drunk: Boolean, if it's True, then there's chance that the player will move two cells instead of one.

We should notice that, because we need to give the leaving-playing-field penalty, so I added the outter court margin, so the actual size will be: (court_width+margin*2, court_height+margin*2)

We could use the env as follow:
```
import Env from env
env = Env(36,24,250,0,1) # 36*24 court
```

### Warning:
When you set the `opponent_can_move` parameter as `True`, the training model will be HUGE. 
Because we need to record the opponent state (x,y) in the Q matrix, every opponent will take 2 dimension, and each dimension will be the length of width and height.
For example, when we have a 9*6 court (the world will be 11*8), 2 opponent. The Q matrix's shape will be:
```
11(player's x), 8(player's y), 11(ball's x), 8(ball's y), 11(opponent_1's x) * 8(opponent_1's y) * 11(opponent_1's x) * 8(opponent_1's y) * 5(action space)
```

## References:
* http://incompleteideas.net/book/first/ebook/node78.html
* https://towardsdatascience.com/reinforcement-learning-temporal-difference-sarsa-q-learning-expected-sarsa-on-python-9fecfda7467e
* https://zhuanlan.zhihu.com/p/26985029
* https://www.zhihu.com/question/26408259
* https://blog.techbridge.cc/2017/11/04/openai-gym-intro-and-q-learning/