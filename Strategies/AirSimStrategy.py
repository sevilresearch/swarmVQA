"""
Jason Carnahan
Edited by: Nathan Rayon
7/29/2021

Strategy specifically for the AirSim Implementation.
"""
import airsim
import os
import cv2
import numpy as np
import Strategy
import json
import uuid
from squaternion import Quaternion
import math

import albumentations as A
from albumentations.pytorch import ToTensorV2


class AirSimStrategy(Strategy.Strategy):

    def __init__(self, parent=None):
        self.parent = parent  # Parent UAV of this strategy
        self.master = None  # Reference to the connection to the central control
        self.default_json_save_location = "data-files/settings.json"  # Where to store the settings by default
        self.client = None  # AirSim client (The Unreal Simulation)
        self.UNIQUE_SAVE_LOCATION = r"C:\Users\NATHANRAYON\Documents\AirSim\settings.json"  # CHANGE THIS TO WHERE EVER YOUR settings.json IS LOCATED

    def take_uav_images(self, uav, client):
        """
        Captures images from the UAV's bottom-facing camera and returns them as an array of RGB images.
        Returns:
            List of numpy arrays representing the captured images.
        """



        print(client.armDisarm(True))
        client.enableApiControl(True)
        # Capture the image
        responses = client.simGetImages([
                airsim.ImageRequest("bottom_view", airsim.ImageType.Scene, False, False)
            ], vehicle_name=uav.ID)
        response = responses[0]
        


    
        uuidid = uuid.uuid1()

      

        # Convert image data to numpy array
        img1d = np.frombuffer(response.image_data_uint8, dtype=np.uint8)
    
        # Ensure correct reshape
        try:
            img_rgb = img1d.reshape(response.height, response.width, 3)
        except ValueError as e:
            print(f"Reshape failed: {e}")
            return None

        # Flip the image vertically
        img_rgb = np.flipud(img_rgb)

        transform = A.Compose([
        A.Resize(600, 600),  # Resize image to 600x600
        ToTensorV2()  # Convert image to tensor
          ])
        augmented = transform(image=img_rgb)
        image = augmented['image']
        return image.div(255)

        # Save the image as PNG
        #airsim.write_png(os.path.normpath('filename.png'), img_rgb)
                        
                    
        airsim.write_png(os.path.normpath(str(uuidid) + '.png'), img_rgb)

        return img_rgb


    def connect_new_client_to_environment(self):
        """
        Connection specifically for the UAVs in the AirSim simulator.
        """
        # Create the JSON for the AirSim settings
        if self.master.num_of_uav:
            self.generate_settings_json(self.master.num_of_uav, self.master.sep_distance, self.master.row_length,
                                        json_location=self.UNIQUE_SAVE_LOCATION)

        # Connect to AirSim
        self.client = airsim.MultirotorClient()
        self.client.confirmConnection()

        # Enable all UAVs
        i = 1
        while i <= len(self.master.uavs):
            self.client.enableApiControl(True, f"Drone{i}")
            self.client.armDisarm(True, f"Drone{i}")
            i = i + 1

        # Assume UAVs could be enabled successfully, it will auto crash if an error occurs
        return True

    

    # Communications ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    def establish_connection(self, master):
        """
        For AirSim, this will simply provide a pointer to the central controller
        """
        self.master = master
        return True

    def get_other_uav_positions(self):
        """
        Since central control has a list of all the UAVs and they have their positions attached, get that list and
        remove self from the list
        """
        # Communicate to the central control for the other UAVs
        otherUAVs = self.master.uavs.copy()

        # Find parent UAV from the list and remove it
        i = 0
        for uav in otherUAVs:
            if uav.ID == self.parent.ID:
                otherUAVs.pop(i)
                break
            i = i + 1

        return otherUAVs

    def get_self_position(self):
        """
        Calls to Sim to get a UAV's absolute position
        :return: X, Y, Z, Theta
        """
        # Call to get uav kinematics in environment
        kinematics = self.client.simGetGroundTruthKinematics(vehicle_name=self.parent.ID)

        # Grab the [x, y, z] positions from the kinematics
        q = np.array([kinematics.position.x_val, kinematics.position.y_val, kinematics.position.z_val])

        # Grab the [w, x, y, z] orientations from the kinematics
        qo = np.array([kinematics.orientation.w_val, kinematics.orientation.x_val, kinematics.orientation.y_val,
                       kinematics.orientation.z_val])

        # Add Initial coordinates
        qdi = np.array([q[0] + self.parent.initialX, q[1] + self.parent.initialY, q[2] + self.parent.initialZ])

        # Turn the orientations into a quaternion then to euler
        qua = Quaternion(qo[0], qo[1], qo[2], qo[3])
        euler = qua.to_euler(degrees=False)

        # return X, Y, Z position and facing angle in radians
        return qdi[0], qdi[1], qdi[2], euler[2]

    def connect_to_environment(self):
        """
        Connection specifically for the UAVs in the AirSim simulator.
        """
        # Create the JSON for the AirSim settings
        if self.master.num_of_uav:
            self.generate_settings_json(self.master.num_of_uav, self.master.sep_distance, self.master.row_length,
                                        json_location=self.UNIQUE_SAVE_LOCATION)

        # Connect to AirSim
        self.client = airsim.MultirotorClient()
        self.client.confirmConnection()

        # Enable all UAVs
        i = 1
        while i <= len(self.master.uavs):
            self.client.enableApiControl(True, f"Drone{i}")
            self.client.armDisarm(True, f"Drone{i}")
            i = i + 1

        # Assume UAVs could be enabled successfully, it will auto crash if an error occurs
        return True

    def close_connection(self):
        """
        Disarms and closes communication to UAVs in the Sim.
        Sucks we are doing two loops over the UAVs but apparently it has to be in this order???
        """

        # Disarm UAVs
        i = 1
        while i <= len(self.master.uavs):
            self.client.armDisarm(True, f"Drone{i}")
            i = i + 1

        # Reset client
        self.client.reset()

        # Disable Control
        i = 1
        while i <= len(self.master.uavs):
            self.client.enableApiControl(False, f"Drone{i}")
            i = i + 1

        # Remove the saved client
        self.client = None

    # Movement ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    def move_by_heading(self, vx, vy, vz, duration, angle_in_rads):
        """
        Calls to move the UAV in the Sim
        """
        return self.client.moveByVelocityZAsync(vx, vy, vz, duration, airsim.DrivetrainType.MaxDegreeOfFreedom,
                                           airsim.YawMode(False, np.degrees(angle_in_rads)),
                                           vehicle_name=self.parent.ID)

    def orbit(self, target, rotation=0):
        """
        Moves to target then orbits around it
        :param target: the target to orbit around
        :param rotation: the direction to orbit (0 = CW, 1 = CCW)
        """
        # Get the position of the target
        kinematics = self.client.simGetObjectPose(target)

        # Grab the [x, y] positions from the kinematics
        target_position = [kinematics.position.x_val, kinematics.position.y_val]

        # Get the distance and angle to target
        distance = np.sqrt((target_position[0] - self.parent.x)**2 + (target_position[1] - self.parent.y)**2)
        angle_to_target = np.arctan2((target_position[1] - self.parent.y), (target_position[0] - self.parent.x))

        # Stop spinning
        if angle_to_target > 2 * np.pi:
            angle_to_target -= 2 * np.pi
        elif angle_to_target < 0:
            angle_to_target += 2 * np.pi

        # Set the desired speed
        desired_speed = self.parent.maxVelocity

        # Check the target distance
        if distance < 10:
            # I am close enough, which way will I orbit?
            if rotation:
                """
                Not working as well when the orbit angle flips from -2pi to 0
                """
                # orbit CCW
                if angle_to_target - self.parent.theta > 0:
                    angle_to_target -= 2 * np.pi
                facing_angle = angle_to_target - self.parent.theta
                moving_angle = facing_angle + (90 * np.pi/180)
                print("orbiting CCW")
            else:
                # orbit CW
                if angle_to_target - self.parent.theta < 0:
                    angle_to_target += 2 * np.pi
                facing_angle = angle_to_target - self.parent.theta
                moving_angle = facing_angle - (159 * np.pi/180)
                print("orbiting CW")
        else:
            # I was too far, move closer
            facing_angle = None
            moving_angle = angle_to_target - self.parent.theta
            #print("Going to target")

        return [desired_speed, moving_angle, facing_angle]

    # Other Necessary Functions ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    def generate_settings_json(self, num_of_uavs, separating_distance, row_length, json_location=None):
        """
        The settings.json for airsim is how the sim knows where to put UAVs and how to reference them. Anytime you want
        to change the UAVs, a new settings.json must be generated for airsim.
        """
        # Generate the drones themselves
        vehicles = {}
        i = 1
        row_num = 1
        col_num = 1
       
        while i <= num_of_uavs:
            if row_num > row_length:
                row_num = 1
                col_num += 1
                print("123More Details")
                print(separating_distance * -row_num - 10)
                #"X": 0,
                #"Y": 0,
            print("123More Details")
            print(-row_num)
            print(separating_distance)

            print(separating_distance * -row_num - 10)

            # Creating the json layout for the specific UAV data
            vehicles.update({f"Drone{i}": {
                "VehicleType": "SimpleFlight",
                "EnableTrace": True,
                "X": separating_distance * -row_num - 10,
                "Y": 25 * col_num,
                #"X": 0,
                #"Y": 0,
                "Z": -71,
                "Yaw": 90,
                "AllowAPIAlways": True, 
                "Cameras": {
                    "bottom_view": {
                        "CaptureSettings": [
                            {
                                "ImageType": 0,
                                "Width": 1920,
                                "Height": 1080,
                                "FOV_Degrees": 90,
                                "AutoExposureSpeed": 100,
                                "AutoExposureBias": 0,
                                "MotionBlurAmount": 0,
                                "TargetGamma": 2.2,
                                  
                            }
                        ],
                        "X": 0, "Y": 0, "Z": 1,
                        "Pitch": -90, "Roll": 0, "Yaw": 0
                    }
                }
            }})
            i += 1
            row_num += 1

        # Add the drones to the rest of the doc
        data = {
            "SeeDocsAt": "https://github.com/Microsoft/AirSim/blob/master/docs/settings.md",
            "SettingsVersion": 1.2,
            "RpcEnabled": True,
            "SimMode": "Multirotor",
            "Vehicles": vehicles
        }

        # Write it out
        if json_location is not None:
            with open(json_location, "w") as outfile:
                json.dump(data, outfile)
        else:
            with open(self.default_json_save_location, "w") as outfile:
                json.dump(data, outfile)

    def set_parent(self, parent):
        """
        Gives the strategy a reference back to the UAV
        """
        self.parent = parent
