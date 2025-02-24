import math

class doublePendulum:
    def __init__(self):
        #constants
        if True:
            self.g = -9.81
            self.L1 = 100
            self.L2 = 100
            self.m1 = 100
            self.m2 = 100
            self.dt = 0.025
            self.width = 500

        #Angle Parameters
        if True:
            self.theta1 = 0
            self.theta2 = 0

            self.theta1_ = 0
            self.theta2_ = 0

            self.theta1__ = 0
            self.theta2__ = 0

        #Base Parameters:
        if True:
            self.xb = 0
            self.xb_ = 0
            self.xb__ = 0
        
    
    def get_theta1__(self):
        self.theta1__ =  -(self.m1 + self.m2)*self.xb__*math.cos(self.theta1) 
        self.theta1__ -=  self.m2*self.L2*self.theta2__*math.cos(self.theta1 - self.theta2) 
        self.theta1__ -= self.m2*self.L2*(self.theta2_*self.theta2_)*math.sin(self.theta1 - self.theta2) 
        self.theta1__ -= self.g*(self.m1+self.m2)*math.sin(self.theta1)
        self.theta1__ /= (self.m1 + self.m2)*self.L1
    
    def get_theta2__(self):
        self.theta2__ = -self.g*math.sin(self.theta2)
        self.theta2__ -= self.xb__*math.cos(self.theta2)
        self.theta2__ += self.L1*self.theta1_*self.theta1_*math.sin(self.theta1 - self.theta2)
        self.theta2__ -= self.L1*self.theta1__*math.cos(self.theta1 - self.theta2)
        self.theta2__ /= self.L2
    
    def update(self):

        self.xb__ = max(self.xb__, 100)

        self.xb_ += self.xb__*self.dt
        self.xb += self.xb_*self.dt
        self.xb_ *= 0.9

        self.xb = max(self.xb, 70)
        self.xb = min(self.xb, self.width - 70)
        
        
        self.get_theta1__()
        self.get_theta2__()

        self.theta1_ += self.theta1__*self.dt
        self.theta2_ += self.theta2__*self.dt

        self.theta1 += self.theta1_*self.dt
        self.theta2 += self.theta2_*self.dt