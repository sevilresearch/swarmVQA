"""
Jason Carnahan
7/29/2021

Abstract class for any UAV strategy
"""
from abc import ABC, abstractmethod


class Strategy(ABC):

    # Communications ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    @abstractmethod
    def establish_connection(self, central_control):
        """
        Creates a connection to central control
        """
        pass

    @abstractmethod
    def get_other_uav_positions(self):
        """
        Returns the positions of all the other UAVs in the system
        """
        pass

    @abstractmethod
    def get_self_position(self):
        """
        Returns tuple own position as array of X, Y, Z and Facing angle in Radians
        :return: X, Y, Z, Theta
        """
        pass

    @abstractmethod
    def connect_to_environment(self):
        """
        Performs any necessary actions to start the UAVs depending on the environment (strategy) they are in.
        """
        pass

    @abstractmethod
    def close_connection(self):
        """
        Performs any necessary actions to close the UAVs depending on the environment (strategy) they are in.
        """
        pass

    # Movement ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    @abstractmethod
    def move_by_heading(self, x_velocity, y_velocity, z_velocity, duration, angle_in_radians):
        """
        Moves UAV at specified speeds, in a specified direction, by specified duration
        """
        pass