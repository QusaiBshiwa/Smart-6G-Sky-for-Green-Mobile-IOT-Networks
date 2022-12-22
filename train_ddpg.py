import  os
import time
from  env  import  UAV
from ddpg import AGENT
import datetime
import  matplotlib . pyplot  as  plt
import numpy as np
import argparse
import tensorflow as tf

parser = argparse.ArgumentParser(description='Train the DDPG model.')
parser.add_argument('--is_train', type=int, default=1, metavar='train(1) or eval(0)',
                    help='train model of evaluate the trained model')
# TRAINING
parser.add_argument('--gamma', type=float, default=0.9, metavar='discount rate',
                    help='The discount rate of long-term returns')
parser.add_argument('--mem_size', type=int, default=8000, metavar='memorize size',
                    help='max size of the replay memory')
parser.add_argument('--batch_size', type=int, default=64, metavar='batch size',
                    help='batch size')
parser.add_argument('--lr_actor', type=float, default=0.001, metavar='learning rate of actor',
                    help='learning rate of actor network')
parser.add_argument('--lr_critic', type=float, default=0.001, metavar='learning rate of critic',
                    help='learning rate of critic network')
parser.add_argument('--replace_tau', type=float, default=0.001, metavar='replace_tau',
                    help='soft replace_tau')
parser.add_argument('--episode_num', type=int, default=1601, metavar='episode number',
                    help='number of episodes for training')
parser.add_argument('--Num_episode_plot', type=int, default=10, metavar='plot freq',
                    help='frequent of episodes to plot')
parser.add_argument('--save_model_freq', type=int, default=100, metavar='save freq',
                    help='frequent to save network parameters')
parser.add_argument('--model', type=str, default='P_moddpg', metavar='save path',
                    help='the save path of the train model')
parser.add_argument('--R_dc', type=float, default=10., metavar='R_DC',
                    help='the radius of data collection')
parser.add_argument('--R_eh', type=float, default=30., metavar='R_EH',
                    help='the radius of energy harvesting')
parser.add_argument('--w_dc', type=float, default=100., metavar='W_DC',
                    help='the weight of data collection')
parser.add_argument('--w_eh', type=float, default=100., metavar='W_EH',
                    help='the weight of energy harvesting')
parser.add_argument('--w_ec', type=float, default=5., metavar='W_EC',
                    help='the weight of energy consumption')

args = parser.parse_args()

#####################  set the save path  ####################
model_path = '/{}/{}/'.format(args.model, 'models')
path = os.getcwd() + model_path
if not os.path.exists(path):
    os.makedirs(path)
logs_path = '/{}/{}/'.format(args.model, 'logs')
path = os.getcwd() + logs_path
if not os.path.exists(path):
    os.makedirs(path)
figs_path = '/{}/{}/'.format(args.model, 'figs')
path = os.getcwd() + figs_path
if not os.path.exists(path):
    os.makedirs(path)

# Set the font format of the horizontal and vertical coordinates of the drawing
font1 = {'family': 'Times New Roman',
         'weight': 'normal',
         'size': 16,
         }
font2 = {'family': 'Times New Roman',
         'weight': 'normal',
         'size': 14,
         }
Q = 10.
V_ME = 10.
V_max = 20.0
EC_min = 126.
EC_hov  =  168.49   # Hover energy consumption
EC_grd = 178.

def train():
    var = 2.  # control exploration
    for ep in range(args.episode_num):
        s = env.reset()
        ep_reward = 0
        idu  =  0   # service user count
        N_DO  =  0   # Average data overflow users per EPISODES
        DQ  =  0   # IoT user average queue cache percentage
        FX  =  0   # Drone flying out of bounds count
        sum_rate  =  0   # sum data rate
        ehu  =  0   # Total number of EH users
        Eh = 0  # total harvested energy
        Ec = 0  # flying energy consumption
        TD_error  =  0   # critic network training error
        A_loss  =  0   # actor network Q value
        Ht  =  0   # record hover time
        Ft  =  0   # record flight time
        update_network  =  0   # record network training times
        while True:
            LDA = [4., 8., 15., 20.]
            time_steps+=1
            pxy = env.SoPcenter[np.argmax(env.Q)]
            L_AIM = env.canvas.create_rectangle(
                            pxy[0] - 10, pxy[1] - 10,
                            pxy[0] + 10, pxy[1] + 10,
                                fill='red')
            env.Aim.append(L_AIM)

            # mobility of some users
            if time_steps % 10 == 0:  #300
                p = np.random.randint(0, env.N_POI, 2) 

                for j in p:
                    env.SoPcenter[j][0] = env.SoPcenter[j][0] - np.random.randint(-4, 4)
                    env.SoPcenter[j][1] = env.SoPcenter[j][1] - np.random.randint(-8, 8)
                   
                  
                    

                    if env.SoPcenter[j][0] <= env.X_min :
                        env.SoPcenter[j][0] = env.X_min
                        env.SoPcenter[j][0] = env.SoPcenter[j][0] - np.random.randint(-2, 2)

                    elif env.SoPcenter[j][0] >= env.X_max :
                        env.SoPcenter[j][0] = env.X_max
                        env.SoPcenter[j][0] = env.SoPcenter[j][0] - np.random.randint(-2, 2)
                    
                    elif env.SoPcenter[j][1] <= env.Y_min  :
                        env.SoPcenter[j][1] = env.Y_min
                        env.SoPcenter[j][1] = env.SoPcenter[j][1] - np.random.randint(-2, 2)
                    
                    elif env.SoPcenter[j][1] >= env.Y_max :
                        env.SoPcenter[j][1] = env.Y_max
                        env.SoPcenter[j][1] = env.SoPcenter[j][1] - np.random.randint(-2, 2)
                    
            
                    # print(env.SoPcenter[j])
                        # env.SoPcenter[j][1] = env.SoPcenter[j][1] - np.random.randint(-1, 1)

                    # print(env.SoPcenter[10])

                for  i  in  range (env. N_POI ):
                # Create an ellipse, specifying the starting position. fill color
                 if env.lda[i] == LDA[0]:
                    env.canvas.create_oval(
                        env.SoPcenter[i][0] - 5, env.SoPcenter[i][1] - 5,
                        env.SoPcenter[i][0] + 5, env.SoPcenter[i][1] + 5,
                        fill='pink')
                 elif env.lda[i] == LDA[1]:
                    env.canvas.create_oval(
                        env.SoPcenter[i][0] - 5, env.SoPcenter[i][1] - 5,
                        env.SoPcenter[i][0] + 5, env.SoPcenter[i][1] + 5,
                        fill='blue')
                 elif env.lda[i] == LDA[2]:
                    env.canvas.create_oval(
                        env.SoPcenter[i][0] - 5, env.SoPcenter[i][1] - 5,
                        env.SoPcenter[i][0] + 5, env.SoPcenter[i][1] + 5,
                        fill='green')
                 elif env.lda[i] == LDA[3]:
                    env.canvas.create_oval(
                        env.SoPcenter[i][0] - 5, env.SoPcenter[i][1] - 5,
                        env.SoPcenter[i][0] + 5, env.SoPcenter[i][1] + 5,
                        fill='red')

            pxy = env.SoPcenter[np.argmax(env.Q)]
            L_AIM = env.canvas.create_rectangle(
            pxy[0] - 10, pxy[1] - 10,
            pxy[0] + 10, pxy[1] + 10,
            fill='red')
            env.Aim.append(L_AIM)
            # print(env.SoPcenter[0])
            # env.render()
            ft  =  1   # flight time
            act = agent.choose_action(s)
            act = np.clip(np.random.normal(act, var), a_bound.low,
                          a_bound.high)  # add randomness to action selection for exploration
            s_, r, done, dr, cu, eh, ec = env.step_move(act)
            Ft += ft
            Ec  +=  ec
            N_DO += env.N_Data_overflow
            DQ += sum(env.b_S / env.Fully_buffer)
            FX += env.FX

            r += (args.w_dc * dr + args.w_eh * (cu + eh) - args.w_ec * ec)
            ep_reward += r
            agent.store_transition(s, act, r, s_)
            if agent.pointer > args.mem_size:
                # decay the action randomness
                var  =  max ([ var  *  .9999 , 0.1 ])
                td_error, a_loss = agent.learn()
                update_network += 1
                TD_error += td_error
                A_loss += a_loss

            if  done :   # target device falls within drone DC range
                idu  +=  1   # service user count
                ht  =  Q  *  env . updata  /  dr   # Calculate hover time
                sum_rate += dr
                ehu  +=  cu
                Eh  +=  eh  *  ht
                Ht += ht
                env.step_hover(ht)
                N_DO += env.N_Data_overflow
                DQ += sum(env.b_S / env.Fully_buffer)
                s = env.CHOOSE_AIM()
            else:
                s = s_

            if Ht+Ft >= 600:
                if update_network != 0:
                    TD_error /= update_network
                    A_loss /= update_network
                N_DO  /= ( Ht + Ft )   # Average number of overflow data users
                DQ /= (Ht+Ft)
                DQ  /=  env . N_POI   # Average user data buffer size
                FX /= Ft
                Ec  /=  Ft   # Average flight energy consumption
                if  idu :
                    aEh  =  Eh  /  idu
                else:
                    aEh  =  0

                # Real-time output training data
                print(
                    'Ep:%i |TD_error:%i |A_loss:%i |ep_r:%i |L_data:%.2f |sum rate:%.2f |idu:%i |ehu:%i |energy_har:%.2f |avg eh:%.2f |ec:%.2f |N_D:%i |FX:%.1f ' % (
                        ep, TD_error, A_loss, ep_reward, DQ, sum_rate, idu, ehu, Eh, aEh, Ec, N_DO, FX))
                '''
                # write relevant data to the document
                # Fig 1:Average data rate(sum_rate/idu)/Total harvested energy(Ec)/Average flying energy consumption(Ec)
                # Fig 2:Total number of DC devices(idu)/Average energy harvesting rate(Ec/Ht)/Average number of EH devices(ehu/idu)
                '''
                write_str = '%i|%.3f|%.3f|%.3f|%.3f|%.3f|%.3f|%.3f|%.3f|%.3f|%i|%i\n' % (
                    ep, ep_reward, TD_error, A_loss, sum_rate, Eh, aEh, Ec, Ht, Ft, ehu, idu)
                file.write(write_str)
                file.flush()

                # Store the relevant data in the queue for drawing
                plot_x.append(ep)
                plot_TD_error.append(TD_error)
                plot_A_loss.append(A_loss)
                plot_R . append ( ep_reward )   # cumulative reward

                plot_N_DO . append ( N_DO )   # data overflow user count
                plot_DQ . append ( DQ )   # Average user data buffer size
                plot_sr . append ( sum_rate )   # round total hover throughput
                plot_Eh . append ( Eh )   # total energy collected for the round
                plot_ehu . append ( ehu )   # round total charging users
                plot_idu . append ( idu )   # round total collected data user
                plot_Ec . append ( Ec )   # average energy consumption per step per round
                plot_HT . append ( Ht )   # hover time
                plot_FT . append ( Ft )   # flight time
                break

        if ep % args.save_model_freq == 0 and ep != 0:
            agent.save_ckpt(model_path, ep)


if __name__ == '__main__':

    # Initialize the environment
    env = UAV(args.R_dc, args.R_eh)

    # reproducible, set a random seed, in order to be able to reproduce
    env.seed(1)
    np.random.seed(1)
    tf.random.set_seed(1)

    # Define state space, action space, range of motion range
    s_dim = env.state_dim
    a_num  =  2
    a_dim = env.action_space.shape[0]
    a_bound = env.action_space
    '''
    print('s_dim',s_dim)
    print('a_dim',a_dim)
    print('r_dim',r_dim)
    '''
    # Use agent algorithm
    agent = AGENT(args, a_num, a_dim, s_dim, a_bound, True)

    #End Training part:



    #Printing
    t1 = time.time()
    plot_x = []
    plot_R  = []   # Cumulative reward
    plot_N_DO  = []   # data overflow user count
    plot_DQ  = []   # Average user data buffer size
    plot_sr  = []   # Total hover throughput per round
    plot_Eh  = []   # Total energy collected per round
    plot_Ec  = []   # Average energy consumption per step
    plot_ehu  = []   # round total charging users
    plot_idu  = []   # round total collection of data users
    plot_HT  = []   # Hover time
    plot_FT  = []   # flight time
    plot_TD_error = []
    plot_A_loss = []

    file = open(os.path.join('.{}{}'.format(logs_path, 'log.txt')), 'w+')
    train()
    file.close
    #######################################1, Accumulated reward##### ##########################################################################################################################################################################################################################################################################################################################################################################################################################################################################################################################################################################################################
    # draw
    fig = plt.figure(figsize=(8, 8))
    ax = fig.add_subplot(1, 1, 1)  # R
    ax.tick_params(labelsize=16)
    ax.grid(linestyle='-.')

    ax.plot(plot_x, plot_R)
    ax.set_xlabel('Number of training episodes', font1)
    ax.set_ylabel('Accumulated reward', font1)

    label1 = ax.get_xticklabels() + ax.get_yticklabels()
    [label.set_fontname('Times New Roman') for label in label1]
    fig.tight_layout()

    plt.savefig('.{}{}'.format(figs_path, 'Accumulated_reward.jpg'))
    #######################################1、loss##############################################
    # draw
    fig = plt.figure(figsize=(16, 8))

    ax1 = fig.add_subplot(1, 2, 1)
    ax1.tick_params(labelsize=12)
    ax1 . grid ( linestyle = '-.' )
    ax2 = fig.add_subplot(1, 2, 2)
    ax2.tick_params(labelsize=12)
    ax2 . grid ( linestyle = '-.' )

    ax1.plot(plot_x, plot_A_loss)
    ax1.set_xlabel('Number of training episodes', font1)
    ax1.set_ylabel('loss of Actor', font1)
    label1 = ax1.get_xticklabels() + ax1.get_yticklabels()
    [label.set_fontname('Times New Roman') for label in label1]

    ax2.plot(plot_x, plot_TD_error)
    ax2.set_xlabel('Number of training episodes', font1)
    ax2.set_ylabel('td_error of critic', font1)
    label2 = ax2.get_xticklabels() + ax2.get_yticklabels()
    [label.set_fontname('Times New Roman') for label in label2]

    plt.subplots_adjust(wspace=0.3)
    plt.savefig('.{}{}'.format(figs_path, 'loss.jpg'))
    ##################Take 10 drawings############################ ######
    avg_ehu  = [ plot_ehu [ i ] /  plot_idu [ i ] if  plot_idu [ i ] !=  0  else  0  for  i  in  range ( len ( plot_ehu ))]   # Average number of charging users per hover
    avg_eh  = [ plot_Eh [ i ] /  plot_idu [ i ] if  plot_idu [ i ] !=  0  else  0  for  i  in  range ( len ( plot_Eh ))]   # Average charge per hover
    eh_rate  = [ plot_Eh [ i ] /  plot_HT [ i ] if  plot_idu [ i ] !=  0  else  0  for  i  in  range ( len ( plot_Eh ))]   # Average energy collection rate
    plot_r  = [ plot_sr [ i ] /  plot_idu [ i ] if  plot_idu [ i ] !=  0  else  0  for  i  in  range ( len ( plot_sr ))]   # Average data rate per hover
    x = 0
    plot_sr1  =  0   # The amount of data collected in rounds
    plot_r1  =  0   # average data rate per hover
    plot_Eh1  =  0   # total energy collected
    plot_Eh2  =  0   # average harvested energy
    plot_Eh3  =  0   # average energy harvesting rate
    plot_Ec1  =  0   # Average energy consumption
    plot_idu1  =  0   # Number of users uploading data
    plot_ehu1  =  0   # Total number of charging users
    plot_ehu2  =  0   # Average number of charging users per hover
    plot_DQ1  =  0   # Average user data buffer size
    plot_N_DO1  =  0   # data overflow user count

    plot_x_avg = []
    plot_sr_avg  = []   # The amount of data collected in rounds
    plot_r_avg  = []   # Average data rate per hover
    plot_Eh_avg  = []   # total energy collected
    plot_avg_Eh_avg  = []   # Average collected energy
    plot_avg_Eh_rate  = []   # Average energy collection rate
    plot_Ec_avg  = []   # Average energy consumption
    plot_idu_avg  = []   # Number of users uploading data
    plot_ehu_avg  = []   # Total number of charging users
    plot_avg_ehu_avg  = []   # Average number of charging users per hover
    plot_DQ_avg  = []   # Average user data buffer size
    plot_N_DO_avg  = []   # Data overflow user count

    for  i  in  range ( 1 , len ( plot_x )):
        x += i
        plot_sr1  +=  plot_sr [ i ]   # The amount of data collected in rounds
        plot_r1  +=  plot_r [ i ]   # Average data rate per hover
        plot_Eh1  +=  plot_Eh [ i ]   # total energy collected
        plot_Eh2  +=  avg_eh [ i ]   # Average collected energy
        plot_Eh3  +=  eh_rate [ i ]   # Average energy harvesting rate
        plot_Ec1  +=  plot_Ec [ i ]   # Average energy consumption
        plot_idu1  +=  plot_idu [ i ]   # Number of users uploading data
        plot_ehu1  +=  plot_ehu [ i ]   # Total number of charging users
        plot_ehu2  +=  avg_ehu [ i ]   # Average number of charging users per hover
        plot_DQ1  +=  plot_DQ [ i ]   # Average user data buffer size
        plot_N_DO1  +=  plot_N_DO [ i ]   # data overflow user count
        if i % args.Num_episode_plot == 0 and i != 0:
            plot_x_avg.append(x / args.Num_episode_plot)
            plot_sr_avg . append ( plot_sr1  /  args . Num_episode_plot )   # The amount of data collected in rounds
            plot_r_avg . append ( plot_r1  /  args . Num_episode_plot )   # amount of data collected in rounds
            plot_Eh_avg . append ( plot_Eh1  /  args . Num_episode_plot )   # total energy collected
            plot_avg_Eh_avg . append ( plot_Eh2  /  args . Num_episode_plot )   # Average collected energy
            plot_avg_Eh_rate . append ( plot_Eh3  /  args . Num_episode_plot )   # Average collected energy
            plot_Ec_avg . append ( plot_Ec1  /  args . Num_episode_plot )   # Average energy consumption
            plot_idu_avg . append ( plot_idu1  /  args . Num_episode_plot )   # upload data users
            plot_ehu_avg . append ( plot_ehu1  /  args . Num_episode_plot )   # total number of charging users
            plot_avg_ehu_avg . append ( plot_ehu2  /  args . Num_episode_plot )   # Average number of users charging per hover
            plot_DQ_avg . append ( plot_DQ1  /  args . Num_episode_plot )   # Average user data buffer size
            plot_N_DO_avg . append ( plot_N_DO1  /  args . Num_episode_plot )   # data overflow user count

            x = 0
            plot_sr1  =  0   # The amount of data collected in rounds
            plot_r1  =  0   # average data rate per hover
            plot_Eh1  =  0   # total energy collected
            plot_Eh2  =  0   # average harvested energy
            plot_Eh3  =  0   # average energy harvesting rate
            plot_Ec1  =  0   # Average energy consumption
            plot_idu1  =  0   # Number of users uploading data
            plot_ehu1  =  0   # Total number of charging users
            plot_ehu2  =  0   # Average number of charging users per hover
            plot_DQ1  =  0   # Average user data buffer size
            plot_N_DO1  =  0   # data overflow user count

        #####################################################################
        '''
        # draw
        1. Accumulated reward, 2. Sum rate of collected data in rounds 3. Average data rate per hover
        4. Total harvested energy in rounds 5. Average harvested energy per hover
        6. The average flight energy consumption per step of the round. 7. The number of ID users who upload data
        8. The number of EH users 9. Average number of EH users
        10. System average data level Average data buffer length 11. Number of users with data overflow N_d
        '''
    # Fig 1:Average data rate(sum_rate/idu)/Total harvested energy(Ec)/Average flying energy consumption(Ec/Ft)
    # Fig 2:Total number of DC devices(idu)/Average energy harvesting rate(Ec/Ht)/Average number of EH devices(ehu/idu)
    ############################################main_result_1########################
    p1  =  plt . figure ( figsize = ( 24 , 8 ))   # The first subfigure and determine the canvas size

    ax1 = p1.add_subplot(1, 3, 1)
    ax1.tick_params(labelsize=12)
    ax1 . grid ( linestyle = '-.' )
    ax2 = p1.add_subplot(1, 3, 2)
    ax2.tick_params(labelsize=12)
    ax2 . grid ( linestyle = '-.' )
    ax3 = p1.add_subplot(1, 3, 3)
    ax3.tick_params(labelsize=12)
    ax3.grid(linestyle='-.')

    ax1.plot(plot_x_avg, plot_sr_avg, marker='*', markersize='10', linewidth='2')
    ax1.set_xlabel('Number of training episodes', font1)
    ax1.set_ylabel('sum data-rate (bits/Hz)', font1)

    label1 = ax1.get_xticklabels() + ax1.get_yticklabels()
    [label.set_fontname('Times New Roman') for label in label1]

    label1 = ax1.get_xticklabels() + ax1.get_yticklabels()
    [label.set_fontname('Times New Roman') for label in label1]

    ax2.plot(plot_x_avg, plot_Eh_avg, marker='*', markersize='10', linewidth='2')
    ax2.set_xlabel('Number of training episodes', font1)
    ax2.set_ylabel(r'Total harvested energy ($\mu$W)', font1)

    label2 = ax2.get_xticklabels() + ax2.get_yticklabels()
    [label.set_fontname('Times New Roman') for label in label2]

    ax3.plot(plot_x_avg, plot_Ec_avg, marker='*', markersize='10', linewidth='2')
    ax3.set_xlabel('Number of training episodes', font1)
    ax3.set_ylabel('Average flying energy consumption (W)', font1)

    label3 = ax3.get_xticklabels() + ax3.get_yticklabels()
    [label.set_fontname('Times New Roman') for label in label3]

    plt.subplots_adjust(wspace=0.3)
    plt.savefig('.{}{}'.format(figs_path, 'FIG_3.jpg'))
    ####################################################################################################################
    # Fig 2:Total number of DC devices(idu)/Average energy harvesting rate(Ec/Ht)/Average number of EH devices(ehu/idu)
    p1  =  plt . figure ( figsize = ( 22 , 18 ))   # The first subfigure and determine the canvas size

    ax1 = p1.add_subplot(2, 2, 1)
    ax1.tick_params(labelsize=12)
    ax1 . grid ( linestyle = '-.' )
    ax2 = p1.add_subplot(2, 2, 2)
    ax2.tick_params(labelsize=12)
    ax2 . grid ( linestyle = '-.' )
    ax3 = p1.add_subplot(2, 2, 3)
    ax3.tick_params(labelsize=12)
    ax3.grid(linestyle='-.')
    ax4 = p1.add_subplot(2, 2, 4)
    ax4.tick_params(labelsize=12)
    ax4 . grid ( linestyle = '-.' )

    ax1.plot(plot_x_avg, plot_idu_avg, marker='*', markersize='10', linewidth='2')
    ax1.set_xlabel('Number of training episodes', font1)
    ax1.set_ylabel('Total number of DC devices', font1)

    label1 = ax1.get_xticklabels() + ax1.get_yticklabels()
    [label.set_fontname('Times New Roman') for label in label1]

    ax2.plot(plot_x_avg, plot_r_avg, marker='*', markersize='10', linewidth='2')
    ax2.set_xlabel('Number of training episodes', font1)
    ax2.set_ylabel('Average data-rate (bits/Hz)', font1)
    [label.set_fontname('Times New Roman') for label in label2]

    ax3.plot(plot_x_avg, plot_avg_ehu_avg, marker='*', markersize='10', linewidth='2')
    ax3.set_xlabel('Number of training episodes', font1)
    ax3.set_ylabel('Average number of EH devices', font1)

    label3 = ax3.get_xticklabels() + ax3.get_yticklabels()
    [label.set_fontname('Times New Roman') for label in label3]

    ax4.plot(plot_x_avg, plot_avg_Eh_rate, marker='*', markersize='10', linewidth='2')
    ax4.set_xlabel('Number of training episodes', font1)
    ax4.set_ylabel(r'Average energy harvesting rate ($\mu$W/s)', font1)

    label4 = ax4.get_xticklabels() + ax4.get_yticklabels()
    [label.set_fontname('Times New Roman') for label in label4]

    plt.subplots_adjust(wspace=0.3)
    plt.savefig('.{}{}'.format(figs_path, 'FIG_4.jpg'))
    ############################################data_rate/harvested energy########################
    ####################################fly energy consumption/total number of EH user########
    p1  =  plt . figure ( figsize = ( 28 , 14 ))   # The first subfigure and determine the canvas size

    ax1 = p1.add_subplot(2, 4, 1)
    ax1.tick_params(labelsize=12)
    ax1 . grid ( linestyle = '-.' )
    ax2 = p1.add_subplot(2, 4, 2)
    ax2.tick_params(labelsize=12)
    ax2 . grid ( linestyle = '-.' )
    ax3 = p1.add_subplot(2, 4, 3)
    ax3.tick_params(labelsize=12)
    ax3.grid(linestyle='-.')
    ax4 = p1.add_subplot(2, 4, 4)
    ax4.tick_params(labelsize=12)
    ax4 . grid ( linestyle = '-.' )
    ax5 = p1.add_subplot(2, 4, 5)
    ax5.tick_params(labelsize=12)
    ax5 . grid ( linestyle = '-.' )
    ax6 = p1.add_subplot(2, 4, 6)
    ax6.tick_params(labelsize=12)
    ax6 . grid ( linestyle = '-.' )
    ax7 = p1.add_subplot(2, 4, 7)
    ax7.tick_params(labelsize=12)
    ax7 . grid ( linestyle = '-.' )
    ax8 = p1.add_subplot(2, 4, 8)
    ax8.tick_params(labelsize=12)
    ax8.grid(linestyle='-.')

    ax1.plot(plot_x_avg, plot_r_avg, marker='*', markersize='10', linewidth='2')
    ax1.set_xlabel('Number of training episodes', font1)
    ax1.set_ylabel('data rate (bits/Hz)', font1)

    label1 = ax1.get_xticklabels() + ax1.get_yticklabels()
    [label.set_fontname('Times New Roman') for label in label1]

    ax2.plot(plot_x_avg, plot_Eh_avg, marker='*', markersize='10', linewidth='2')
    ax2.set_xlabel('Number of training episodes', font1)
    ax2.set_ylabel(r'Harvested energy ($\mu$W)', font1)

    label2 = ax2.get_xticklabels() + ax2.get_yticklabels()
    [label.set_fontname('Times New Roman') for label in label2]

    ax3.plot(plot_x_avg, plot_Ec_avg, marker='*', markersize='10', linewidth='2')
    ax3.set_xlabel('Number of training episodes', font1)
    ax3.set_ylabel('Average fly energy consumption (W)', font1)

    label3 = ax3.get_xticklabels() + ax3.get_yticklabels()
    [label.set_fontname('Times New Roman') for label in label3]
    # ax3.legend(prop=font2)

    ax4.plot(plot_x_avg, plot_ehu_avg, marker='*', markersize='10', linewidth='2')
    ax4.set_xlabel('Number of training episodes', font1)
    ax4.set_ylabel('Total number of EH user', font1)

    label4 = ax4.get_xticklabels() + ax4.get_yticklabels()
    [label.set_fontname('Times New Roman') for label in label4]
    # ax4.legend(prop=font2)

    ax5.plot(plot_x_avg, plot_sr_avg, marker='*', markersize='10', linewidth='2')
    ax5.set_xlabel('Number of training episodes', font1)
    ax5.set_ylabel('sum rate (bits/Hz)', font1)

    label5 = ax5.get_xticklabels() + ax5.get_yticklabels()
    [label.set_fontname('Times New Roman') for label in label5]
    # ax5.legend(prop=font2)

    ax6.plot(plot_x_avg, plot_idu_avg, marker='*', markersize='10', linewidth='2')
    ax6.set_xlabel('Number of training episodes', font1)
    ax6.set_ylabel('Total number of ID user', font1)

    label6 = ax6.get_xticklabels() + ax6.get_yticklabels()
    [label.set_fontname('Times New Roman') for label in label6]
    # ax6.legend(prop=font2)

    ax7.plot(plot_x_avg, plot_avg_Eh_avg, marker='*', markersize='10', linewidth='2')
    ax7.set_xlabel('Number of training episodes', font1)
    ax7.set_ylabel(r'Average harvested energy ($\mu$W)', font1)

    label7 = ax7.get_xticklabels() + ax7.get_yticklabels()
    [label.set_fontname('Times New Roman') for label in label7]
    # ax7.legend(prop=font2)

    ax8.plot(plot_x_avg, plot_avg_ehu_avg, marker='*', markersize='10', linewidth='2')
    ax8.set_xlabel('Number of training episodes', font1)
    ax8.set_ylabel('Average number of EH user', font1)

    label8 = ax8.get_xticklabels() + ax8.get_yticklabels()
    [label.set_fontname('Times New Roman') for label in label8]
    # ax8.legend(prop=font2)

    plt.subplots_adjust(wspace=0.3)
    plt.savefig('.{}{}'.format(figs_path, 'sum_up.jpg'))
    plt . clf ()
    ##############################################10. System average data level Average data buffer length############################################## #######
    fig = plt.figure(figsize=(16, 8))

    ax1 = fig.add_subplot(1, 2, 1)
    ax1.tick_params(labelsize=12)
    ax1 . grid ( linestyle = '-.' )
    ax2 = fig.add_subplot(1, 2, 2)
    ax2.tick_params(labelsize=12)
    ax2 . grid ( linestyle = '-.' )

    ax1.plot(plot_x_avg, plot_DQ_avg, marker='*', markersize='10', linewidth='2')
    ax1.set_xlabel('Number of training episodes', font1)
    ax1.set_ylabel('Average data buffer length (%)', font1)

    label1 = ax1.get_xticklabels() + ax1.get_yticklabels()
    [label.set_fontname('Times New Roman') for label in label1]

    ax2.plot(plot_x_avg, plot_N_DO_avg, marker='*', markersize='10', linewidth='2')
    ax2.set_xlabel('Number of training episodes', font1)
    ax2.set_ylabel(r'$N_d^{AVG}$', font1)

    label2 = ax2.get_xticklabels() + ax2.get_yticklabels()
    [label.set_fontname('Times New Roman') for label in label2]

    plt.savefig('.{}{}'.format(figs_path, 'system_performance.jpg'))
    plt.show
    ###################################################################################
    now_time = datetime.datetime.now()
    date = now_time.strftime('%Y-%m-%d %H_%M_%S')
    print('Running time: ', time.time() - t1)


