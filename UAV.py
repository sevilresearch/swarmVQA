"""
Jason Carnahan
5/14/2021

Represents the individual UAV.
"""
from Management import Management
from Calculations import Calculations

class UAV:

    def __init__(self, ID, x, y, z, min_distance, move_distance, max_velocity, strategy):
        self.manager = Management(self)
        self.calculator = Calculations(self)
        self.strategy = strategy
        self.ID = ID

        # Initial Values
        self.initialX = x
        self.initialY = y
        self.initialZ = z

        self.minDistance = min_distance
        self.maxDistance = move_distance
        self.maxVelocity = max_velocity

        # Current Values
        self.x = x
        self.y = y
        self.z = z
        self.theta = None
        self.entropy = None
        self.waypoint = None

    def identify(self):
        """
        UAV can identify itself for any other actor.
        """
        return self.ID

    def getEntropy(self):
        """
        UAV can call to get its own entropy.
        """
        # comms to ask for the status of all the other uavs
        otherUAVs = self.strategy.get_other_uav_positions()
        return self.calculator.calculateEntropy(otherUAVs)

    def getDistances(self):
        """
        UAV communicates to the other UAVs for their position.
        """
        # comms to ask for the status of all the other uavs
        otherUAVs = self.strategy.get_other_uav_positions()
        return self.calculator.calculateDistancesAndAngles(otherUAVs)
