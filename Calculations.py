"""
Jason Carnahan
Edited By: Nathan Rayon
5/14/2021

Contains any necessary calculations needed for the UAV and performed onboard
"""

from turtle import st
import numpy as np


class Calculations:

    def __init__(self, parent):
        self.parent = parent  # The parent UAV

    def calculateDistancesAndAngles(self, otherUAVs):
        """
        This function uses the parent x,y position and the positions of all the other UAVs and calculates
        the distance and angle to face the each other UAV.
        :param otherUAVs is an array of all the other UAVs in the system
        :returns a numpy array of the distances from this drone to all other drones
        """
        # create an array for distances and angles
        distances = np.zeros(len(otherUAVs), dtype=float)
        angles = np.zeros(len(otherUAVs), dtype=float)

        # Loop over all the UAVs and calculate distance with the distance formula and the angle with arctan
        i = 0
        for uav in otherUAVs:
            distances[i] = np.sqrt(((uav.x - self.parent.x) ** 2) + ((uav.y - self.parent.y) ** 2))
            angles[i] = np.arctan2((uav.y - self.parent.y), (uav.x - self.parent.x))
            i = i + 1
        return [distances, angles]

    def calculateEntropy(self, otherUAVs):
        """
        This function takes in the parent UAV and each other UAV to calculate the parent UAV's entropy in the system
        :param otherUAVs is an array of all the other UAVs in the system
        """
        # Get the number of UAVs in the system (+ 1 since it does not include parent UAV)
        numOfUAV = len(otherUAVs) + 1

        distances = np.zeros(numOfUAV, dtype=float)  # Contains distances to all the other drones
        e = np.zeros(numOfUAV, dtype=float)  # Contains the first entropy calculation dependent on distance
        q = 0.5  # Weighted value for entropy?
        summation = 0  # Summation part of the entropy equation

        # Loop for all the uavs
        i = 0
        for uav in otherUAVs:
            # Calculate this UAVs distance to every other UAV and get the angle to them (Th in Luke's code)
            distances[i] = np.sqrt(((uav.x - self.parent.x) ** 2) + ((uav.y - self.parent.y) ** 2))

            # Determine what the first step in the entropy calculation is
            if distances[i] >= self.parent.maxDistance:
                e[i] = self.parent.maxDistance
            elif distances[i] <= self.parent.minDistance:
                e[i] = self.parent.minDistance
            else:
                e[i] = distances[i]

            # Add this value to the summation of the Entropy equation
            summation = summation + ((e[i] / self.parent.maxDistance) ** q)
            i = i + 1

        # Calculate entropy
        return (1 - summation) / (q - 1)

    def calculateNextMove(self, otherUAVs):
        """
        This function takes in the parent UAV and all the other UAVs in the system to calculate what the parent's next move
        should be.

        (This May have to be moved to the Manager class.)

        :param otherUAVs is an array of all the other UAVs in the system
        """
        # Calculate entropy and update the value to the UAV
        St = self.calculateEntropy(otherUAVs)
        self.parent.entropy = St

        # Get the distances and angles to other uavs
        store = self.calculateDistancesAndAngles(otherUAVs)
        #if len(store) >= 1:
        d = store[0]  # Contains the distance to every other uav
        #else:
        #    d = 0
        Th = store[1]  # Contains the angle to every other uav

        # Get the distance and angle to the next waypoint
        xd = self.parent.waypoint[0] - self.parent.x  # Delta x to waypoint
        yd = self.parent.waypoint[1] - self.parent.y  # Delta y to waypoint
        dd = np.sqrt(xd ** 2 + yd ** 2)  # Delta d to waypoint
        waypoint_theta = np.arctan2(yd, xd)  # Target Theta to waypoint

        # Find the closest uav
        #if d != 0:
        closest = d[0]
        closest_index = 0
        i = 0
        for distance in d:
            if distance < closest:
                closest = distance
                closest_index = i
            i = i + 1
        #else:
            #closest = self.parent.minDistance + 1

        # This threshold will need to be made into an equation eventually
        # threshold = 0.8   # Entropy threshold 3 UAV
        # threshold = 21  # 20 UAV
        #Default = 0.5
        threshold = 6
        print(St)


        maxGroupingDistance = self.parent.minDistance * 5
        theta_com = 0  # Desired theta
        V_com = 0  # Desired velocity

        # Determine what the UAV should do
        if closest < self.parent.minDistance:
            # UAV is too close to another, move away
            V_com = self.parent.maxVelocity / 1.5  # Set speed

            if Th[closest_index] > 2 * np.pi:
                Th[closest_index] -= 2 * np.pi
            elif Th[closest_index] < 0:
                Th[closest_index] += 2 * np.pi

            # Determine if the other UAV is in front or behind
            if Th[closest_index] > self.parent.theta + np.pi / 2 and Th[closest_index] < 2 * np.pi:
                # other is behind, move straight away
                theta_com = (Th[closest_index] - self.parent.theta) + np.pi
                print(f"{self.parent.ID} moving away from {otherUAVs[closest_index].ID}")
            else:
                # Determine if the other uav is front left or front right
                if Th[closest_index] > self.parent.theta:
                    # Front left, move RIGHT
                    theta_com = Th[closest_index] - self.parent.theta - np.pi / 2
                    print(f"{self.parent.ID} moving away from {otherUAVs[closest_index].ID} to the RIGHT")
                else:
                    # Front right, move LEFT
                    theta_com = Th[closest_index] - self.parent.theta + np.pi / 2
                    print(f"{self.parent.ID} moving away from {otherUAVs[closest_index].ID} to the LEFT")

        # Check if the entropy level is less than the threshold
        elif St < threshold:
            V_com = self.parent.maxVelocity

            # Check if this UAV is too close to another
            if closest > self.parent.minDistance:
                # not too close, move to waypoint
                print(f"{self.parent.ID} moving to waypoint: [{self.parent.waypoint[0]}, {self.parent.waypoint[1]}]")
                theta_com = waypoint_theta - self.parent.theta
                self.parent.target = self.parent.waypoint

        else:
            # Not in threshold, move to the closest UAV outside the min grouping distance
            V_com = 2 * self.parent.maxVelocity

            # Find what UAV is the closest outside the min grouping distance
            if closest < maxGroupingDistance:
                closest = 999999
                closest_index = 0
                i = 0
                for distance in d:
                    if distance > maxGroupingDistance and distance < closest:
                        closest = distance
                        closest_index = i
                    i = i + 1

            # Move toward it
            print(f"{self.parent.ID} moving to closest neighbor {otherUAVs[closest_index].ID}")
            theta_com = Th[closest_index] - self.parent.theta

        if theta_com > np.pi:
            theta_com = -2 * np.pi + theta_com
        elif theta_com < -np.pi:
            theta_com = 2 * np.pi + theta_com

        # Return the determined Velocity and angle
        return [V_com, theta_com]
