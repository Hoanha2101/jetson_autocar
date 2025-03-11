class DIRECTION_CLASSIFICATION:
    def __init__(self):
        self.DIRECTION = "STRAIGHT"
        self.DIRECTION_PREVIOUS = None

    def change(self, dir_real):
        self.DIRECTION = dir_real
        
    def check(self):
        return self.DIRECTION
    
    def check_previous(self):
        return self.DIRECTION_PREVIOUS
    
    def change(self, new_direction):
        self.DIRECTION_PREVIOUS = self.DIRECTION
        self.DIRECTION = new_direction
    
USE_CLASSIFICATION = DIRECTION_CLASSIFICATION()

