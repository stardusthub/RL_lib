import matplotlib.pyplot as plt
import numpy as np

'''以下3条语句用来显示中文
'''
'''
from pylab import * 
plt.rcParams['font.sans-serif']=['SimHei']
plt.rcParams['axes.unicode_minus']=False
'''
MMM = 1


def Proc(a):
    Nx = len(a)
    print(Nx, max(a), min(a))
    running_avg =   np.empty(int(Nx / MMM))
    for t in range(int(Nx / MMM)):
        running_avg[t] = np.array(a[t * MMM:(t + 1) * MMM]).sum()
    return running_avg


def Proc1(a):
    Nx = len(a)
    print(Nx)
    running_avg = np.empty(int(Nx))
    for t in range(int(Nx)):
        running_avg[t] = np.array(a[max(0, t - 20):t + 1]).mean()
    return running_avg


if __name__ == '__main__':
    # a=np.loadtxt('E:/agent0_head0.txt')
    # b=np.loadtxt('E:/1head_agen'         t0_head1.txt')

    c1 = np.loadtxt('episode_reward.txt')

    '''
    c1=np.loadtxt('E:/r_room_1.txt')
    c2=np.loadtxt('E:/r_room_2.txt')
    c3=np.loadtxt('E:/r_room_3.txt')
    c4=np.loadtxt('E:/r_room_4.txt')
    '''

    # c1=np.loadtxt('E:/Paper相关/Paper35-JIOT-DRL/Paper_Materials/Paper_code/Figure_code/plot_temperature---fig.5/Total_reward_Convergence13.txt')
    # d=np.loadtxt('E:/data_Comfort.txt')
    # e=np.loadtxt('E:/004_reward.txt')
    # a1=Proc(a)
    # a2=Proc(b)
    a1 = Proc(c1)
    a11 = Proc1(a1)

    # a3_11=Proc(c1)
    # a3_21=Proc1(a3_11)
    # a4=Proc(d)
    # a5=Proc(e)
    # plt.plot(a1,label='agent0_head0')
    # plt.plot(a2,label='1head_agent0_head1')
    # plt.plot(a3,label='Attention head number=1')

    plt.plot(a1, label='Episode reward')
    plt.plot(a11, label='Average reward')
    '''
    plt.plot(a3,label='Zone 3')
    plt.plot(a4,label='Zone 4')

'''
    '''
    plt.plot(a1,label='奖励（区域1）')
    plt.plot(a2,label='奖励（区域2）')
    plt.plot(a3,label='奖励（区域3）')
    plt.plot(a4,label='奖励（区域4）')
    '''

    # plt.plot(a3_1,label='片段奖励')
    # plt.plot(a3_2,label='Average reward')
    # plt.plot(d,label='Discomfort cost')
    # plt.xlabel('奖励')
    # plt.xlim([0,1000])
    plt.ylabel('Rewards')
    plt.xlabel('Episodes')
    plt.legend(loc='best')
    plt.show()





