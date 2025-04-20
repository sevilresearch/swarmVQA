"""
Jason Carnahan
7/29/2021

Strategy specifically for the Tello Implementation.
"""
import tello
import Strategy

class TelloStrategy(Strategy.Strategy):

    # Communications ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

    def establish_connection(self, central_control):
        """
        Creates a connection to central control
        """
        pass

    def get_other_uav_positions(self):
        """
        Returns the positions of all the other UAVs in the system
        """
        pass

    def get_self_position(self):
        """
        Returns own position
        """
        pass

    def connect_to_environment(self):
        """
        Performs any necessary actions to start the UAVs depending on the environment (strategy) they are in.
        """
        pass

    def close_connection(self):
        """
        Performs any necessary actions to close the UAVs depending on the environment (strategy) they are in.
        """
        pass

    # Movement ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

    def move_by_heading(self, x_velocity, y_velocity, z_velocity, duration, angle_in_radians):
        """
        Moves UAV at specified speeds, in a specified direction, by specified duration
        """
        pass