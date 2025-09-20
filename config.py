

class Config:
    def __init__(self):
        self.v_max = 6.0
        self.a_max = 6.0
        self.r = 10.0
        self.Tf = self.r / self.v_max

        self.smoothness_weight = 10.0
        self.safety_weight = 0.1
        self.goal_weight = 0.1

        self.smoothness_weight = self.smoothness_weight / self.v_max ** 5
        self.safety_weight = self.safety_weight * self.v_max
        self.goal_weight = self.goal_weight


cfg = Config()