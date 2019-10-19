import matplotlib.pyplot as plt
import numpy as np
import math
import random
import benchmark as bmk  #自制的benchmark函数

class SA(object):
    def __init__(self, dim, interval, tab='min'):
        self.interval = interval   #带求解的区间
        self.T_max = 10000     #初始退火温度  温度上限
        self.T_min = 100       #截止退火温度  温度下限
        self.iterMax = 1000   #内部降温迭代次数
        self.rate = 0.8       #退火降温速率
        self.dim = dim       #搜索空间维数
        self.V = 0.1 #更新步长
        self.x_seed = np.random.uniform(interval[0], interval[1],size=(self.dim,1)) #搜索的初始值
        self.tab = tab.strip()     #求解最大值还是最小值

    def p_min(self, delta, T):          #计算最小值时，容忍解得迁移概率
        return np.exp(-delta/T)

    def p_max(self, delta, T):        #计算最大值时，容忍解得迁移概率
        return np.exp(delta/T)

    def deal_min(self, x1,x2,delta,T):
        if delta < 0:                   #更优解
            return x2
        else:
            P = self.p_min(delta, T)    #容忍解
            if P > random.random():
                return x2
            else:
                return x1
        
    def deal_max(self, x1,x2,delta,T):
        if delta > 0:
            return x2
        else:
            P = self.p_min(delta, T)
            if P > random.random():
                return x2
            else:
                return x1

    def SaSolution(self, Pg, T):
        while T >= self.T_min:
            deltaP = np.random.uniform(-1,1,size=(self.dim))*self.V
            if (Pg + deltaP).all() >= self.interval[0] and (Pg + deltaP).all() <= self.interval[1]:
                PgTrial = Pg + deltaP
            else:
                PgTrial = Pg - deltaP        #将随机解限制在解空间内
            fitnessTrial = bmk.GRF(PgTrial)
            fitness = bmk.GRF(Pg)
            deta = fitnessTrial - fitness
            if deta <= 0:      #如果满足最优解
                Pg = PgTrial
                T *= self.rate
            else:
                P = math.exp(-deta/T)
                if P > random.random():   #判断metropolis准则
                    Pg = PgTrial
                    T *= self.rate
                else:
                    break
            plt.clf()
            plt.scatter(Pg[0], Pg[1], s=30, color='r')
            plt.xlim(self.interval[0], self.interval[1])
            plt.ylim(self.interval[0], self.interval[1])
            plt.grid()
            plt.pause(0.01)
        return Pg


class PSO(object):
    def __init__(self, dim, bound):
        #self.w = 0.6  #惯性权重
        self.c1 = 0.8
        self.c2 = 1.7  #学习因子
        self.population_size = 50  #种群数量
        self.max_steps = 100 #迭代次数
        self.w_min = 0.1
        self.w_max = 0.55     #权重的最大和最小值

        self.v_max = 0.1*bound[1]#粒子速度的最大值,实验证明为每一个变量范围的10%

        self.dim = dim #搜索空间维度,即benchmark函数的变量个数
        self.bound = bound  #解空间的范围，即benchmark函数的定义域
        self.x = np.random.uniform(self.bound[0], self.bound[1],
                                    size=(self.population_size,self.dim))  #初始化粒子群的位置
        self.v = np.zeros((self.population_size, self.dim) )  #初始速度为0，默认为64float

        #适应值初始化
        fitness = np.zeros((self.population_size,1))
        k = 0
        for x_position in self.x:
            fitness[k] = bmk.GRF(x_position)   #计算每个粒子位置的适应值
            k = k+1
        self.p = self.x #个体的最佳位置
        self.pg = self.x[np.argmin(fitness)]  #全局的最佳位置
        self.individual_best_fitness = fitness  #个体最优适应度
        self.global_best_fitness = np.min(fitness) #全局最优适应度

        self.sa = SA(dim, bound)  #初始化SA算法

    def evolve(self):
        fig = plt.figure()
        T = self.sa.T_max  #模拟退火算法的初始温度
        for step in range(self.max_steps):
            R1 = np.random.random()
            R2 = np.random.random()

            self.w =  self.w_max - step * (self.w_max - self.w_min)/self.max_steps   #时变权重

            self.v = self.w*self.v + self.c1*R1*(self.p - self.x)\
                + self.c2*R2*(self.pg-self.x) #速度更新

            great_id = np.greater_equal(self.v, self.v_max)   #将速度限制在范围内
            self.v[great_id] = self.v_max
            less_id = np.less_equal(self.v, -self.v_max)
            self.v[less_id] = -self.v_max

            self.x = self.v + self.x    #位置更新

            plt.clf()
            plt.scatter(self.x[:,0], self.x[:,1], s=30, color='k')
            plt.xlim(self.bound[0], self.bound[1])
            plt.ylim(self.bound[0], self.bound[1])
            plt.grid()
            plt.pause(0.01)

            #计算适应值
            fitness = np.zeros((self.population_size,1))
            k=0
            for x_position in self.x:
                fitness[k] = bmk.GRF(x_position)
                k = k+1 
            #个体最优值更新
            update_id = np.greater(self.individual_best_fitness, fitness)
            k=0
            for change in update_id:
                if change == True:
                    self.p[k,:] = self.x[k,:]
                k=k+1
            self.individual_best_fitness[update_id] = fitness[update_id]#个体最优适应度更新
            #全局最优值更新
            if np.min(fitness) < self.global_best_fitness:   
                self.pg = self.x[np.argmin(fitness)]   #全局最优位置更新
                self.global_best_fitness = np.min(fitness)  #全局最优适应度更新
                T *= self.sa.rate    #如果全局最优值有更新，不调用sa算法，但是降温
            else:           #如果全局最优值不更新的话，调用SA算法
                self.pg = self.sa.SaSolution(self.pg, T)   #调用SA算法,当sa算法失败时，返回迭代得到的全局最优值 

        print('best fitness: {0} , mean fitness: {1} '.format(self.global_best_fitness, np.mean(fitness)))

pso = PSO(2,[-5.12,5.12])
pso.evolve()
plt.show() 