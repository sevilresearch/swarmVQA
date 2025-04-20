"""
Nathan Rayon
4/20/2025

Image Capture Class
"""

import airsim
import os
import numpy as np
import cv2

class UAVImageCapture:

    def __init__(self, strategy, uavs):
        self.strategy = strategy
        self.client = self.strategy.client
        self.uavs = uavs

    def take_uav_images(self, save_directory='EntropyRewrite/data-files/images'):
        # Ensure the save directory exists
        os.makedirs(save_directory, exist_ok=True)

        # Loop through each UAV
        for uav in self.uavs:

            # Capture image from the UAV's camera
            responses = self.client.simGetImages([
                airsim.ImageRequest("0", airsim.ImageType.Scene, False, False)
            ], vehicle_name=uav)
