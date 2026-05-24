class Position:
    def __init__(self, x, y):
        self.x = x
        self.y = y

    @property
    def xy(self):
        return (self.x, self.y)
    
    def set_xy(self, x, y):
        self.x += x
        self.y += y
        
class OutputValues:
    def __init__(self, angle, speed):
        self.angle = angle
        self.speed = speed