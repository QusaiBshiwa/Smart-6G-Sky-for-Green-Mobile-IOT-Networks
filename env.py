import sys
import numpy as np
import time
import math
import copy
from gym import spaces
from  gym . utils  import  seeding

# Take v in [0,1], theta in [-1,1] as the action
if sys.version_info.major == 2:
    import  tkinter  as  tk
else:
    import  tkinter  as  tk
'''
The map size is 400*400
Number of dronesN_UAV=1
The number of service points is 100, all service points are randomly distributed within a limited area, and service points have different data collection characteristics
Fixed map
'''
WIDTH  =  10   # width of the map
HEIGHT  =  10   # height of the map
UNIT  =  40   # The size of each square (pixel value) 400m*400m map
LDA  = [ 4. , 8. , 15. , 20. ]   # Suppose there are 4 types of sensors, that is, there are 4 different Poisson parameters, and the sensor data generation obeys Poisson distribution
max_LDA  =  max ( LDA )
C  =  5000   # The capacity of the sensor is assumed to be 5000
P_u  =  pow ( 10 , - 5 )   # The transmit power of the sensor is 0.01mW, -20dbm
P_d  =  10   # UAV downlink transmit power 10W, 40dBm
H  =  10.   # The drone has a fixed hovering height of 10m
R_d  =  30.   # The charging coverage of the drone can receive 0.1mW at 10m and 0.01mW at 30m
N_S_  =  100   # number of devices
V  =  20   # The maximum speed of the drone is 20m/s
b_S_  =  np . random . randint ( 0 , 500 , N_S_ )   # Initialize the current data buffer size of the sensor

# nonlinear energy receiver model
Mj = 9.079 * pow(10, -6)
aj = 47083
bj = 2.9 * pow(10, -6)
Oj = 1 / (1 + math.exp(aj * bj))

np.random.seed(1)


# Define the drone class
class UAV(tk.Tk, object):
    def __init__(self, R_dc=10., R_eh=30.):
        super(UAV, self).__init__()
        # POI location
        self . N_POI  =  N_S_   # number of sensors
        self . dis  =  np . zeros ( self . N_POI )   # distance squared
        self . elevation  =  np . zeros ( self . N_POI )   # 仰角
        self . pro  =  np . zeros ( self . N_POI )   # line-of-sight probability
        self . h  =  np . zeros ( self . N_POI )   # channel gain
        self.N_UAV = 1
        self . max_speed  =  V   # The maximum speed of the drone is 20m/s
        self . H  =  10.   # The flying height of the drone is 10m
        self.X_min = 0
        self.Y_min = 0
        self.X_max = (WIDTH) * UNIT
        self . Y_max  = ( HEIGHT ) *  UNIT   # map bounds
        self . R_dc  =  R_dc   # Horizontal coverage distance 10m
        self . R_eh  =  R_eh   # Horizontal coverage distance 30m
        self . sdc  =  math . sqrt ( pow ( self . R_dc , 2 ) +  pow ( self . H , 2 ))   # max DC service distance
        self . seh  =  math . sqrt ( pow ( self . R_eh , 2 ) +  pow ( self . H , 2 ))   # maximum EH service distance
        self . noise  =  pow ( 10 , - 12 )   # noise power is -90dbm
        self.AutoUAV = []
        self.Aim = []
        self . N_AIM  =  1   # select the number of users to serve
        self.FX = 0.
        self.SoPcenter = np.random.randint(10, 390, size=[self.N_POI, 2])
        # Take v in [0,1], theta in [-1,1] as the action
        self.action_space = spaces.Box(low=np.array([0., -1.]), high=np.array([1., 1.]),
                                       dtype=np.float32)
        self . state_dim  =  6   # The state space is the relative position of the highest priority user position and the drone, the drone position, whether it hit the wall, the number of data overflows
        self.state = np.zeros(self.state_dim)
        self . xy  =  np . zeros (( self . N_UAV , 2 ))   # UAV position

        # Assuming there are 4 types of sensors, that is, there are 4 different Poisson parameters, randomly assign Poisson parameters to sensors
        CoLDA = np.random.randint(0, len(LDA), self.N_POI)
        self . lda  = [ LDA [ CoLDA [ i ]] for  i  in  range ( self . N_POI )]   # Specify data growth rate for sensors
        self . b_S  =  np . random . randint ( 0. , 500. , self . N_POI ). astype ( np . float32 )   # Initialize the current sensor data buffer size
        self.Fully_buffer = C
        self . N_Data_overflow  =  0   # data overflow count
        self.Q = np.array(
            [ self . lda [ i ] *  self . b_S [ i ] /  self . Fully_buffer  for  i  in  range ( self . N_POI )])   # Data collection priority
        self.idx_target = np.argmax(self.Q)
        self.updata = self.b_S[self.idx_target] / self.Fully_buffer
        '''
        # Specify an area where the environment has an impact on the sensor
        for i in range(self.N_POI):
            if all(self.SoPcenter[i] >= [120, 120]) and all(self.SoPcenter[i] <= [280, 280]):
                self.lda[i] += 3.
        '''

        self.title('MAP')
        self . geometry ( '{0}x{1}' . format ( WIDTH  *  UNIT , HEIGHT  *  UNIT ))   # Tkinter geometry
        self.build_maze()

    # create map
    def build_maze(self):
        # Create a canvas Canvas. White background, width and height.
        self.canvas = tk.Canvas(self, bg='white', width=WIDTH * UNIT, height=HEIGHT * UNIT)

        '''
        # mark the special area
        for c in range(120, 280, UNIT * 4 - 1):
            x0, y0, x1, y1 = c, 120, c, 280
            self.canvas.create_line(x0, y0, x1, y1)
        for r in range(120, 280, UNIT * 4 - 1):
            x0, y0, x1, y1 = 120, r, 280, r
            self.canvas.create_line(x0, y0, x1, y1)
        '''
        # create user
        for  i  in  range ( self . N_POI ):
            # Create an ellipse, specifying the starting position. fill color
            if  self . lda [ i ] ==  LDA [ 0 ]:
                self.canvas.create_oval(
                    self.SoPcenter[i][0] - 5, self.SoPcenter[i][1] - 5,
                    self.SoPcenter[i][0] + 5, self.SoPcenter[i][1] + 5,
                    fill='pink')
            elif self.lda[i] == LDA[1]:
                self.canvas.create_oval(
                    self.SoPcenter[i][0] - 5, self.SoPcenter[i][1] - 5,
                    self.SoPcenter[i][0] + 5, self.SoPcenter[i][1] + 5,
                    fill='blue')
            elif self.lda[i] == LDA[2]:
                self.canvas.create_oval(
                    self.SoPcenter[i][0] - 5, self.SoPcenter[i][1] - 5,
                    self.SoPcenter[i][0] + 5, self.SoPcenter[i][1] + 5,
                    fill='green')
            elif self.lda[i] == LDA[3]:
                self.canvas.create_oval(
                    self.SoPcenter[i][0] - 5, self.SoPcenter[i][1] - 5,
                    self.SoPcenter[i][0] + 5, self.SoPcenter[i][1] + 5,
                    fill='red')

        # create drone
        self.xy = np.random.randint(100., 300., size=[self.N_UAV, 2])

        for  i  in  range ( self . N_UAV ):
            L_UAV = self.canvas.create_oval(
                self.xy[i][0] - R_d, self.xy[i][1] - R_d,
                self.xy[i][0] + R_d, self.xy[i][1] + R_d,
                fill='yellow')
            self.AutoUAV.append(L_UAV)

        # user selection
        pxy = self.SoPcenter[np.argmax(self.Q)]
        L_AIM = self.canvas.create_rectangle(
            pxy[0] - 10, pxy[1] - 10,
            pxy[0] + 10, pxy[1] + 10,
            fill='red')
        self.Aim.append(L_AIM)

        self.canvas.pack()

    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    # reset, initialize the drone's position randomly
    def reset(self):
        self.render()
        for  i  in  range ( self . N_UAV ):
            self.canvas.delete(self.AutoUAV[i])
        self.AutoUAV = []

        for  i  in  range ( len ( self . Aim )):
            self.canvas.delete(self.Aim[i])

        # Randomly initialize the drone position
        self.xy = np.random.randint(100, 300, size=[self.N_UAV, 2]).astype(float)
        for  i  in  range ( self . N_UAV ):
            L_UAV = self.canvas.create_oval(
                self.xy[i][0] - R_d, self.xy[i][1] - R_d,
                self.xy[i][0] + R_d, self.xy[i][1] + R_d,
                fill='yellow')
            self.AutoUAV.append(L_UAV)
        self.FX = 0.

        self . b_S  =  np . random . randint ( 0 , 500 , self . N_POI )   # Initialize the current sensor data buffer size
        self.b_S = np.asarray(self.b_S, dtype=np.float32)
        self . N_Data_overflow  =  0   # data overflow count
        self . Q  =  np . array ([ self . lda [ i ] *  self . b_S [ i ] /  self . Fully_buffer  for  i  in  range ( self . N_POI )])   # data collection priority

        # Initialize state space values
        self.idx_target = np.argmax(self.Q)
        self.updata = self.b_S[self.idx_target] / self.Fully_buffer
        self . pxy  =  self . SoPcenter [ self . idx_target ]   # Initially select the one with the highest priority

        L_AIM = self.canvas.create_rectangle(
            self.pxy[0] - 10, self.pxy[1] - 10,
            self.pxy[0] + 10, self.pxy[1] + 10,
            fill='red')
        self.Aim.append(L_AIM)

        self.state = np.concatenate(((self.pxy - self.xy[0]).flatten() / 400., self.xy.flatten() / 400., [0., 0.]))

        return self.state

    # Pass in current state and input action output next state and reward
    def step_move(self, action, above=False):
        if above == True:
            detX = action[:self.N_UAV] * self.max_speed
            detY = action[self.N_UAV:] * self.max_speed
        else:
            detX = action[0] * self.max_speed * math.cos(action[1] * math.pi)
            detY = action[0] * self.max_speed * math.sin(action[1] * math.pi)
        state_ = np.zeros(self.state_dim)
        xy_  =  copy .deepcopy ( self .xy )   # position update
        Flag  =  False   # Whether the drone is flying or not
        for  i  in  range ( self . N_UAV ):   # UAV position update
            xy_[i][0] += detX
            xy_[i][1] += detY
            # When the updated position of the drone is out of the map range
            if xy_[i][0] >= self.X_min and xy_[i][0] <= self.X_max:
                if xy_[i][1] >= self.Y_min and xy_[i][1] <= self.Y_max:
                    self.FX = 0.
                    Flag = True
                else:
                    xy_[i][0] -= detX
                    xy_[i][1] -= detY
                    self.FX += 1.
            else:
                xy_[i][0] -= detX
                xy_[i][1] -= detY
                self.FX += 1.
        if Flag:
            # flight energy
            V = math.sqrt(pow(detX, 2) + pow(detY, 2))
            ec = 79.86 * (1 + 0.000208 * pow(V, 2)) + 88.63 * math.sqrt(
                math.sqrt(1 + pow(V, 4) / 1055.0673312400002) - pow(V, 2) / 32.4818) + 0.009242625 * pow(V, 3)
        else:
            ec  =  168.49   # hover energy

        for  i  in  range ( self . N_UAV ):
            self.canvas.move(self.AutoUAV[i], xy_[i][0] - self.xy[i][0], xy_[i][1] - self.xy[i][1])

        self.xy = xy_
        # After the location of the drone is updated, determine the status of the service point receiving the service
        self . N_Data_overflow  =  0   # Record the number of users with data overflow per slot
        self . b_S  += [ np . random . poisson ( self . lda [ i ]) for  i  in  range ( self . N_POI )]   # sensor data buffer update
        for  i  in  range ( self . N_POI ):   # Data overflow handling
            if self.b_S[i] >= self.Fully_buffer:
                self . N_Data_overflow  +=  1   # data overflow user count
                self.b_S[i] = self.Fully_buffer
        self.updata = self.b_S[self.idx_target] / self.Fully_buffer

        # state space normalization
        state_ [: 2 ] = ( self . pxy  -  xy_ ). flatten () /  400.   # Update the relative position of the user and the drone
        state_ [ 2 : 4 ] =  xy_ . flatten () /  400.   # drone absolute position
        state_ [ 4 ] =  self . FX  /  400.   # Drone crossings/total steps
        state_ [ 5 ] =  self . N_Data_overflow  /  self . N_POI   # Proportion of data overflow users

        Done = False

        # Definition of reward - get to the destination as soon as possible / don't hit the wall / reduce energy consumption
        reward = -(abs(state_[0]) + abs(state_[1])) * 100 - self.FX * 10 - self.N_Data_overflow * 5
        self . Q_dis ()   # Get the channel gain of all users and drones
        ehu  =  0   # charge to cover the user
        data_rate  =  0   # data rate
        eh  =  0   # total charge
        if (above == False and self.dis[self.idx_target] <= self.sdc) or (
                above == True and abs(state_[0]) <= 0.002 and abs(state_[1]) <= 0.002):
            Done = True
            reward += 500
            # Only collect data for target users
            data_rate = math.log(1 + P_u * self.h[self.idx_target] / self.noise, 2)  # 2.397~4.615
            self.b_S[self.idx_target] = 0
            for  i  in  range ( self . N_POI ):
                if self.dis[i] <= self.seh and i != self.idx_target:
                    ehu  +=  1
                    eh  +=  self . Non_linear_EH ( P_d  *  self . h [ i ])   # input is 10-4W~10-5W, output is 0.6751969599046135~7.418403066937866
        # print(sum_rate,ehu,eh)
        self.state = state_

        return  state_ , reward , Done , data_rate , ehu , eh , ec   # state value, reward, whether to reach the goal, total data rate, users covered, energy collection, UAV energy consumption

    def step_hover(self, hover_time):
        # The drone doesn't move, so s[:5] doesn't change
        self . N_Data_overflow  =  0   # Record the number of users with data overflow per slot
        self . b_S  += [ np . random . poisson ( self . lda [ i ]) *  hover_time  for  i  in  range ( self . N_POI )]   # Sensor data buffer update
        for  i  in  range ( self . N_POI ):   # Data overflow handling
            if self.b_S[i] >= self.Fully_buffer:
                self . N_Data_overflow  +=  1   # data overflow user count
                self.b_S[i] = self.Fully_buffer
        self.updata = self.b_S[self.idx_target] / self.Fully_buffer
        self . state [ 5 ] =  self . N_Data_overflow  /  self . N_POI   # Proportion of data overflow users

    # After each time the drone updates its position, calculate the distance and elevation angle between the drone and all users, as well as the path gain
    def Q_dis(self):
        for  i  in  range ( self . N_POI ):
            self.dis[i] = math.sqrt(
                pow(self.SoPcenter[i][0] - self.xy[0][0], 2) + pow(self.SoPcenter[i][1] - self.xy[0][1], 2) + pow(
                    self . H , 2 ))   # original distance
            self.elevation[i] = 180 / math.pi * np.arcsin(self.H / self.dis[i])  # 仰角
            self . pro [ i ] =  1  / ( 1  +  10  *  math . exp ( - 0.6  * ( self . elevation [ i ] -  10 )))   # line-of-sight probability
            self.h[i] = (self.pro[i] + (1 - self.pro[i]) * 0.2) * pow(self.dis[i], -2.3) * pow(10,
                                                                                               - 30  /  10 )   # The reference distance gain is -30db

    # Input is 10-4W~10-5W, output is 0~9.079muW
    def Non_linear_EH(self, Pr):
        if  Pr  ==  0 :
            return 0
        P_prac = Mj / (1 + math.exp(-aj * (Pr - bj)))
        Peh  = ( P_prac  -  Mj  *  Oj ) / ( 1  -  Oj )   # in W
        return Peh * pow(10, 6)

    # Input is 10-4W~10-5W, output is 0~9.079muW
    def linear_EH(self, Pr):
        if  Pr  ==  0 :
            return 0
        return Pr * pow(10, 6) * 0.2

    # Reselect target user
    def CHOOSE_AIM(self):
        for  i  in  range ( len ( self . Aim )):
            self.canvas.delete(self.Aim[i])

        # Reselect target user
        self . Q  =  np . array ([ self . lda [ i ] *  self . b_S [ i ] /  C  for  i  in  range ( self . N_POI )])   # data collection priority
        self.idx_target = np.argmax(self.Q)
        self.updata = self.b_S[self.idx_target] / self.Fully_buffer
        self.pxy = self.SoPcenter[self.idx_target]
        L_AIM = self.canvas.create_rectangle(
            self.pxy[0] - 10, self.pxy[1] - 10,
            self.pxy[0] + 10, self.pxy[1] + 10,
            fill='red')
        self.Aim.append(L_AIM)

        self.state[:2] = (self.pxy - self.xy[0]).flatten() / 400.
        self.render()
        return self.state

    # Call Tkinter's update method, 0.01 seconds to take a step.
    def render(self, t=0.01):
        time.sleep(t)
        self.update()

    def sample_action(self):
        v = np.random.rand()
        theta = -1 + 2 * np.random.rand()
        return [v, theta]


def update():
    for  t  in  range ( 10 ):
        env.reset()
        while True:
            env.render()
            paras = env.sample_action()
            s, r, done, sum_rate, cover_u, eh, ec = env.step_move(paras)
            if done:
                break


if __name__ == '__main__':
    env = UAV()
    env.after(10, update)
    env.mainloop()