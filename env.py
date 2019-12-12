import random
import math
import gym
from gym.envs.classic_control import rendering
import time

class Env():
    def __init__(self,court_width, court_height, opponent_number=0, opponent_can_move=False, drunk=False):

        self.outter_margin = 1
        self.court_width = court_width + self.outter_margin*2 # 周遭一圈是界外
        self.court_height = court_height + self.outter_margin*2 # 周遭一圈是界外
        
        self.opponent_number = opponent_number
        self.opponent_can_move = opponent_can_move
        self.drunk = drunk

        self.basket_position = [self.court_width-self.outter_margin-1, math.ceil(self.court_height/2)]
        self.ball_position = [self.outter_margin, math.ceil(self.court_height/2)]
        self.player_position = [0,0]
        self.opponents = []
        self.action_space = ["r","l","u","d","s"]

        self.viewer = None

        self.reset(True)

    def reset(self, first_time=False):
        
        self.ball_position = [self.outter_margin, math.ceil(self.court_height/2)]

        self.player_position[0] = random.randint(self.outter_margin,self.court_width-self.outter_margin-1)
        self.player_position[1] = random.randint(self.outter_margin,self.court_height-self.outter_margin-1)
        
        if first_time or self.opponent_can_move :
            self.opponents = []
            while len(self.opponents) < self.opponent_number:
                e = [ random.randint(1,self.court_width-2), random.randint(1,self.court_height-2) ]
                if e!=self.player_position and e!=self.basket_position and e!=self.ball_position:
                    self.opponents.append(e)

        return self._get_observation()
        
    def step(self, action):
        reward = self._get_reward()
        episode_over = self._determine_over()
        self._take_action(action)
        # self.status = self.env.step()
        ob = self._get_observation()
        return ob, reward, episode_over

    def _take_action(self, action):
        is_holding_ball = self.ball_position == self.player_position

        action = self.action_space[action]
        
        if action == "s":
            if is_holding_ball:
                d = (self.player_position[0]-self.basket_position[0], self.player_position[1]-self.basket_position[1])
                distance = math.sqrt( d[0]**2 + d[1]**2 )
                # If distance < 4, then the aggent could shoot
                if distance < 4:
                    if distance < 1:
                        if random.random() < 0.9:
                            self.ball_position = self.basket_position
                    elif distance < 3:
                        if random.random() < 0.66:
                            self.ball_position = self.basket_position
                    else:
                        if random.random() < 0.1:
                            self.ball_position = self.basket_position
                    # 如果沒有投籃成功，跑到定點
                    if self.ball_position != self.basket_position:
                        self.ball_position = [
                                                self.outter_margin + math.floor( 0.8*(self.court_width-self.outter_margin*2) ), 
                                                self.outter_margin + math.ceil( 0.5*(self.court_height-self.outter_margin*2) )
                                            ]
        else:
            if self.drunk:
                random_number = random.random()
                if random_number<0.6:
                    self._forward(action)
                elif random_number<0.6+0.3:
                    self._forward(action)
                    self._forward(action)
            else:
                self._forward(action)
            

    def _forward(self,action):
        is_holding_ball = self.ball_position == self.player_position

        player_origin_position = [self.player_position[0],self.player_position[1]]
        if action == "u":
            self.player_position[1]+=1        
        elif action == "d":
            self.player_position[1]-=1
        elif action == "r":
            self.player_position[0]+=1
        elif action == "l":
            self.player_position[0]-=1

        if (self.player_position[1] >= self.court_height or
            self.player_position[1] < 0 or
            self.player_position[0] >= self.court_width or
            self.player_position[0] < 0 or
            self.player_position == self.basket_position ):

            self.player_position[0] = player_origin_position[0]
            self.player_position[1] = player_origin_position[1]

        if is_holding_ball:
            self.ball_position = self.player_position

    def _get_reward(self):
        if self.player_position in self.opponents:
            return -5
        if ( self.player_position[0]<=0 or self.player_position[0]>=self.court_width-1 or 
             self.player_position[1]<=0 or self.player_position[1]>=self.court_height-1     ):
            return -100
        if self.ball_position == self.basket_position:
            d = (self.player_position[0]-self.ball_position[0], self.player_position[1]-self.ball_position[1])
            distance = math.sqrt( d[0]**2 + d[1]**2 )
            if distance < 3:
                return 10
            elif distance < 4:
                return 30
            else:
                return 0
        else:
            return 0

    def _get_observation(self):
        ob = (self.player_position[0], self.player_position[1], self.ball_position[0], self.ball_position[1])
        if self.opponent_can_move:
            for op in self.opponents:
                ob = ob + (op[0],op[1])
        return ob

    def _determine_over(self):
        reward = self._get_reward()
        if reward<0 or reward>=10:
            return True
        else:
            return False

    def render(self, mode='human'):

        scale = 20
        dot_radius = 8

        if self.viewer is None:
            screen_width = self.court_width * scale
            screen_height = self.court_height * scale

            self.viewer = rendering.Viewer(screen_width, screen_height)

            # Outter court
            outter_court = rendering.FilledPolygon(
                [
                    (0, 0), 
                    (0, self.court_height*scale), 
                    (self.court_width*scale,    self.court_height * scale), 
                    (self.court_width*scale,    0)
                ]
            )
            outter_court.set_color(0.5,0.5,0.5)
            outter_court_trans = rendering.Transform(translation=(0,0))
            outter_court.add_attr(outter_court_trans)
            self.viewer.add_geom(outter_court)

            # Inner court
            inner_width = (self.court_width-self.outter_margin*2)*scale
            inner_height = (self.court_height-self.outter_margin*2)*scale
            inner_court = rendering.FilledPolygon([
                (   0, 0), 
                (   0, inner_height), 
                (   inner_width, inner_height), 
                (   inner_width, 0)
            ])
            inner_court.set_color(1,1,1)
            inner_court_trans = rendering.Transform(translation=(self.outter_margin*scale,self.outter_margin*scale))
            inner_court.add_attr(inner_court_trans)
            self.viewer.add_geom(inner_court)

            # Basket
            self.basket_geom = self.viewer.draw_circle(dot_radius)
            self.basket_trans = rendering.Transform()
            self.basket_geom.add_attr(self.basket_trans)
            self.viewer.add_geom(self.basket_geom)
            self.basket_geom.set_color(135/255,206/255,250/255)

            # Opponents
            self.opponent_trans = []
            for o in self.opponents:
                o_geom = self.viewer.draw_circle(dot_radius)
                o_trans = rendering.Transform()
                o_geom.add_attr(o_trans)
                o_geom.set_color(220/255,20/255,60/255)
                self.viewer.add_geom(o_geom)
                self.opponent_trans.append(o_trans)

            # Player
            self.player_geom = self.viewer.draw_circle(dot_radius)
            self.player_trans = rendering.Transform()
            self.player_geom.add_attr(self.player_trans)
            self.viewer.add_geom(self.player_geom)

            # Ball
            self.ball_geom = self.viewer.draw_circle(dot_radius)
            self.ball_trans = rendering.Transform()
            self.ball_geom.add_attr(self.ball_trans)
            self.viewer.add_geom(self.ball_geom)
            self.ball_geom.set_color(255/255,140/255,0)

        self.player_trans.set_translation(self.player_position[0]*scale+dot_radius-2, self.player_position[1]*scale+dot_radius)
        self.ball_trans.set_translation(self.ball_position[0]*scale+dot_radius+2, self.ball_position[1]*scale+dot_radius)
        self.basket_trans.set_translation(self.basket_position[0]*scale+dot_radius, self.basket_position[1]*scale+dot_radius)
        for i in range(len(self.opponents)):
            self.opponent_trans[i].set_translation( self.opponents[i][0]*scale+dot_radius, self.opponents[i][1]*scale+dot_radius )

        time.sleep(0.1)

        return self.viewer.render(return_rgb_array = mode=='rgb_array')
