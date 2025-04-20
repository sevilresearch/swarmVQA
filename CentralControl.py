"""
Jason Carnahan
Edited by: Nathan Rayon
5/14/2021

Handles meta data manipulation and analytics. Keeps track of the drones in the system,
and handles interaction for the user
"""
import csv
import numpy as np

from UAV import UAV
import pandas as pd
from matplotlib.animation import FuncAnimation, PillowWriter
import matplotlib.pyplot as plt
import time
import os
import airsim
from uav_camera import UAVImageCapture
from VQA import VQA
import math
# 1) Import shapely libraries
from shapely.geometry import Polygon
from shapely.ops import unary_union
import time
import threading
import torchvision

class CentralControl:
    """
    Central Controler that
    """

    def __init__(self, strategy, vqa):
        self.strategy = strategy  # Strategy to use
        self.uav_camera = None # UAV Camera class for each strategy
        self.uavs = []  # Storage for the UAVs
        self.Vmax = 2   # Max Speed of the UAVs default 1.25
        self.maxDistance = 100  # Max Survey Distance between UAVs DEFAULT IS 100
        self.minDistance = 5  # Min physical distance between UAVs
        self.globalPositionOffsetx = 6000
        self.globalPositionOffsety = -13000

        self.ue_x = [6000, 7200, 6300, -4600, -8800, -3400]
        self.ue_y = [-13000, -6100, 7600, 7900, -500, -11000]
       
        self.waypoints = []
        self.generate_waypoints()

        # For the structure of the UAV init
        self.num_of_uav = None
        self.sep_distance = None
        self.row_length = None

        #Initialize vqa model
        self.vqa = vqa

        
        self.num_iters = 0
        self.start_time = 0
        self.end_time = 0
        self.total_delay = 0
        self.delay_count = 0

        self.total_shapely_area = 0
        self.shapely_area_count = 0

        self.finished = False
        self.program_start_time = None
        self.program_end_time = None

    def generate_waypoints(self):
        for i, uex in enumerate(self.ue_x):
            x = (uex) / 100
            y = (self.ue_y[i]) / 100
           
            new_waypoint = [x, y]
            self.waypoints.append(new_waypoint)

    def init_system(self, num_of_uavs, separating_distance, row_length):
        """
        Since all that is needed to add UAVs to the simulation is num_of_uavs, separating_distance, and row_length;
        the UAV positions will need to be calculated when added to the system
        """

        # Save the Structure
        self.num_of_uav = num_of_uavs
        self.sep_distance = separating_distance
        self.row_length = row_length

        # Init the structure
        i = 1
        row = 1
        col = 1

        while i <= num_of_uavs:
            if row > row_length:
                row = 1
                col += 1

            # Create the UAV
            #self.add_uav(f"Drone{i}", separating_distance * -row + 100, 25 * col, -2)
            self.add_uav(f"Drone{i}", separating_distance * -row - 10, 25 * col, -2)

            i += 1
            row += 1
        
        

    def add_uav(self, ID, x, y, z):
        """
        Used apart of the initialization to add UAVs to the internal system.
        :param ID: Name of the UAV
        :param x: x coordinate
        :param y: y coordinate
        :param z: z coordinate
        """
        # Create a new UAV object with a new strategy class
        strategy_class = self.strategy.__class__
        copied_strategy = strategy_class()

        new_uav = UAV(ID, x, y, z, self.minDistance, self.maxDistance, self.Vmax, copied_strategy)
        new_uav.strategy.set_parent(new_uav)

        # Create a connection to the UAV
        new_uav.strategy.establish_connection(self)

        # Add it to the internal data
        self.uavs.append(new_uav)

    def connect_to_environment(self):
        """
        Establishes connection the environment of the UAVs
        """


        for uav in self.uavs:
            uav.strategy.connect_to_environment()

        #self.uav_camera.strategy.connect_to_environment()


        return True

    def close_connection(self):
        """
        Disable connection to the UAVs. Typically done on shut down.
        """

        #self.uav_camera.strategy.close_connection()

        for uav in self.uavs:
            uav.strategy.close_connection()

    def call_uav_update(self):
        """
        Calls to the UAV to update its own Kinematics.
        This should not be done here with the idea that the UAV handles the calculations.
        """
        for uav in self.uavs:
            uav.x, uav.y, uav.z, uav.theta = uav.strategy.get_self_position()

            if uav.theta > 2 * np.pi:
                uav.theta -= 2 * np.pi
            elif uav.theta < 0:
                uav.theta += 2 * np.pi

    def uav_init(self):
        """
        Sends a 'takeoff' command to get each UAV ready for formation
        """
        for uav in self.uavs:
            move = uav.strategy.move_by_heading(0, 0, -2, 1, 0)

    def entropyFormation(self):
        """
        Provides commands for the Entropy formation
        """

        self.program_start_time = time.time()


        atFinalWaypoint = False  # Flag to check if the formation is complete
        commandDuration = 1  # How long each movement action takes (in s?)
        iterationCount = 0  # User statements to see something happening
        waypoints = self.waypoints.copy()  # copy of the waypoints
        nextWaypoint = waypoints.pop(0)  # [x, y] of the next waypoint
        z = -2  # Consistent height

        print("Starting Formation Control")
        print(os.getcwd())

        # Print the first line to the save file
        with open('EntropyRewritePlusVQA/data-files/entropySave.csv', 'w') as csvfile:
            fileWrite = csv.writer(csvfile, delimiter=' ', quotechar='|')
            writeOut = []
            i = 1

            while i <= len(self.uavs):
                writeOut.append(f"X{i}")
                writeOut.append(f"Y{i}")
                writeOut.append(f"EntropyDrone{i}")

                if i == len(self.uavs):
                    writeOut.append("TimeDifference")
                i += 1

            fileWrite.writerow(writeOut)

        

            # Capture the start time
        beginningTime = time.time()
        iter_count = 0

        cameraclient = airsim.MultirotorClient()
        cameraclient.confirmConnection()

        footprintClient = airsim.MultirotorClient()
        footprintClient.confirmConnection()
        t_thread = threading.Thread(target=self.perform_vqa, args=(cameraclient,))
        t_thread.start()

        f_thread = threading.Thread(target=self.calculate_uav_footprint, args=(footprintClient,))
        f_thread.start()
        


        # Main loop until all UAVs have reached the final waypoint
        while not atFinalWaypoint:
            iter_count += 1

            iterationCount = iterationCount + 1

            # Capture the time difference
            currentTime = time.time()
            timeDiff = (currentTime - beginningTime) / 1000000000

            # choose when to print to console and file
            if iterationCount % 100 == 0:
                print(f"Iteration = {iterationCount}")
                if iterationCount % 500 == 0:
                    self.write_data(self.uavs, timeDiff)

            #waypoint_offset = self.minDistance
            waypoint_offset = 2 * self.minDistance  # The uav will seek the waypoint with this off set so they dont collide into the same point THIS IS THE ORIGINAL
            # In the future, may one to try and calculate the center of mass of the swarm and use that???
            closeEnough = 1  # offset from waypoint so the uav does not have to be exactly on it
            targetReached = len(self.uavs)  # flag for the amount of uavs on target. Bound between 0 and Max Number of UAVs

            # Check UAV and waypoint relationship
            i = 0
            for uav in self.uavs:
                # check if the UAV has a target
                if uav.waypoint is not None:
                    # Check if the uav has made it to its target
                    if np.sqrt(((uav.waypoint[0] - uav.x)**2) + ((uav.waypoint[1] - uav.y)**2)) < closeEnough:
                        if targetReached < len(self.uavs):
                            targetReached = targetReached + 1
                        print(f"{uav.ID} has reached its target")

                        uav.strategy.move_by_heading(0, 0, z, commandDuration, 0)
                        continue
                    else:
                        if targetReached > 0:
                            targetReached = targetReached - 1
                else:
                    uav.waypoint = [nextWaypoint[0] - (i * waypoint_offset), nextWaypoint[1] + (i * waypoint_offset)]
                    targetReached = targetReached - 1

                # Tell the UAV to update its position values
                self.call_uav_update()

                # Get the UAV's position status
                uavStatus = [uav.x, uav.y, uav.theta]
                #print(uavStatus)

                # Get the desired V and theta of the uav
                otherUAVs = uav.strategy.get_other_uav_positions()
                desired = uav.calculator.calculateNextMove(otherUAVs)  # [speed, angle]
                uavDesires = np.array([round(desired[0], 4), round(desired[1], 4)])  # create numpy array

                # Smooth init transition
                if iterationCount != 1:
                    transition_angle = uavStatus[2] + uavDesires[1] * 0.5
                else:
                    transition_angle = uavStatus[2]

                # Convert from [-pi, pi] to [0, 2pi] -
                transition_angle = np.where(transition_angle < 0, 2 * np.pi + transition_angle, transition_angle)

                # Account for multiples of pi due to spinning
                if transition_angle > 2 * np.pi:
                    transition_angle = transition_angle % (2 * np.pi)

                # Calculate the desired velocity XY vectors
                veloVector = np.array([round(uavDesires[0] * np.cos(transition_angle), 4), round(uavDesires[0] * np.sin(transition_angle), 4)])

                # Call to move the uav
                uav.strategy.move_by_heading(veloVector[0], veloVector[1], z, commandDuration, transition_angle)
                i = i + 1

            # Get the next waypoint to move
            if targetReached == len(self.uavs):
                       
             
                # All UAVs have reach their target
                if len(waypoints) != 0:

                    # There are more waypoints to reach
                    targetReached = 0  # reset flag
                    nextWaypoint = waypoints.pop(0)  # get next waypoint

                    # assign new waypoint to the UAVs
                    i = 0
                    for uav in self.uavs:
                        uav.waypoint = [nextWaypoint[0] - (i * waypoint_offset), nextWaypoint[1] + (i * waypoint_offset)]
                        i = i + 1
                else:
                    # There were no more waypoints, finished mission
                    #self.vqa.predict()
                    atFinalWaypoint = True
                    self.finished = True
                    csvfile.close()
          

        t_thread.join()
        f_thread.join()
            
         
                
    def perform_vqa(self, client):
        #Perform VQA on each uav
        while self.finished != True:
            print("Taking UAV images...")
            self.start_time = time.time()
            for index, uav in enumerate(self.uavs):
                image = self.take_uav_image(uav, client)
                self.vqa.add_image(image)
            
                self.vqa.predict()
              
                if(index == len(self.uavs) - 1):
                    self.end_time = time.time()
                    self.delay_count += 1
                    image_delay = self.end_time - self.start_time
                    self.total_delay += image_delay



    def orbit(self):
        """

        """
        target = "TemplateCube_Rounded_117"
        iterationCount = 0
        z = -2
        commandDuration = 1

        t_thread = threading.Thread(target=self.printVQAStats)
        t_thread.start()
        t_thread.join()

        while True:
            iterationCount = iterationCount + 1

            for uav in self.uavs:
                # Tell the UAV to update its position values
                self.call_uav_update()

                # Get the UAV's position status
                uavStatus = [uav.x, uav.y, uav.theta]

                # get speed and direction
                desired = uav.strategy.orbit(target)
                uavDesires = np.array([round(desired[0], 4), round(desired[1], 4)])

                # Smooth init transition
                if iterationCount != 1:
                    if desired[2] is not None:
                        # Orbiting
                        facing_angle = uavStatus[2] + desired[2] * 0.5
                        moving_angle = uavStatus[2] + uavDesires[1] * 0.5
                    else:
                        # Not orbiting
                        moving_angle = uavStatus[2] + uavDesires[1] * 0.5
                        facing_angle = moving_angle
                else:
                    # Initial command
                    moving_angle = uavStatus[2]
                    facing_angle = moving_angle

                # Convert from [-pi, pi] to [0, 2pi] -
                moving_angle = np.where(moving_angle < 0, 2 * np.pi + moving_angle, moving_angle)
                facing_angle = np.where(facing_angle < 0, 2 * np.pi + facing_angle, facing_angle)

                # Calculate the desired velocity XY vectors
                veloVector = np.array([round(uavDesires[0] * np.cos(moving_angle), 4), round(uavDesires[0] * np.sin(moving_angle), 4)])

                # Call to move the uav
                uav.strategy.move_by_heading(veloVector[0], veloVector[1], z, commandDuration, facing_angle)

    def write_data(self, uavs, timeDiff):
        """
        Writes information out to the save file
        """
        with open('EntropyRewritePlusVQA/data-files/entropySave.csv', 'a') as csvfile:
            fileWrite = csv.writer(csvfile, delimiter=' ', quotechar='|')
            writeOut = []

            for uav in uavs:
                writeOut.append(uav.x)
                writeOut.append(uav.y)
                writeOut.append(uav.entropy)
            writeOut.append(timeDiff)

            fileWrite.writerow(writeOut)


    def get_camera_footprint(self, center_x, center_y, altitude, fov_deg, image_width, image_height):
        fov_rad = math.radians(fov_deg)
        W = 2 * altitude * math.tan(fov_rad / 2)  # Ground width covered
        H = (W * image_height) / image_width      # Ground height covered
        half_W, half_H = W / 2, H / 2
        # Return the bounding box (min_x, min_y, max_x, max_y, area)
        return (center_x - half_W, center_y - half_H,
                center_x + half_W, center_y + half_H,
                W * H)

    def calculate_overlap(self, box1, box2):
        # Extract coordinates
        x1_min, y1_min, x1_max, y1_max, _ = box1
        x2_min, y2_min, x2_max, y2_max, _ = box2

        # Check if there is NO overlap
        if x1_max <= x2_min or x2_max <= x1_min or y1_max <= y2_min or y2_max <= y1_min:
            return 0  # No overlap

        # Calculate overlapping area if they intersect
        x_overlap = max(0, min(x1_max, x2_max) - max(x1_min, x2_min))
        y_overlap = max(0, min(y1_max, y2_max) - max(y1_min, y2_min))

        # Area of overlap
        overlap_area = x_overlap * y_overlap
        return overlap_area

    def take_uav_image(self, uav, client):
       
        print("Taking image from ", uav.ID)
            
        img = uav.strategy.take_uav_images(uav, client)

        return img

    def take_uav_images_grid(self):
        imgs = []
        for uav in self.uavs:
            imgs.append(uav.strategy.take_uav_images(uav))


        if(len(imgs) == 4):
            grid_tensor = torchvision.utils.make_grid(imgs, nrow=2)
            #grid_np = grid_tensor.permute(1,2,0).numpy()

            #plt.figure(figsize=(8, 8))
            #plt.imshow(grid_np)
            #plt.title("2x2 UAV Images")
            #plt.axis("off")
            #plt.show()

            return grid_tensor


        #return imgs

    def calculate_uav_footprint(self, client):
        print("Calculating UAV Footprints...")

        while self.finished == False:

            fov_deg = 90
            image_width, image_height = 1920, 1080

            # You mentioned altitude=29, but if you truly want 29 m above ground, 
            # check that your local frame matches that. For now, let's use a fixed 29.
            altitude = 30

            # Store UAV footprints
            footprints = []
            total_area_covered = 0

            for uav in self.uavs:

                # If you want to read altitude from AirSim:
                # state = uav.strategy.client.getMultirotorState(vehicle_name=uav.ID)
                # altitude = -state.kinematics_estimated.position.z_val
                # For demonstration, you are hardcoding 29

                x = client.getMultirotorState(vehicle_name=uav.ID).kinematics_estimated.position.x_val
                y = client.getMultirotorState(vehicle_name=uav.ID).kinematics_estimated.position.y_val

                footprint = self.get_camera_footprint(x, y, altitude, fov_deg, image_width, image_height)
                footprints.append(footprint)
                total_area_covered += footprint[4]  # area component

    #            print(f"UAV {uav.ID} altitude: {altitude:.2f} m")

            # Print bounding boxes for debugging
     #       for i, fp in enumerate(footprints):
     #           print(f"UAV {i+1} footprint:", fp)

            # ---- Pairwise overlap approach (only partial solution if triple overlaps exist) ----
            total_overlap_area = 0
            for i in range(len(footprints)):
                for j in range(i + 1, len(footprints)):
                    overlap_area = self.calculate_overlap(footprints[i], footprints[j])
                    total_overlap_area += overlap_area

            naive_unique_area = total_area_covered - total_overlap_area

            # ---- Shapely approach for correct multi-UAV union coverage ----
            # Convert each bounding box to a polygon and union them
            footprint_polys = []
            for (x_min, y_min, x_max, y_max, area) in footprints:
                poly = Polygon([
                    (x_min, y_min),
                    (x_min, y_max),
                    (x_max, y_max),
                    (x_max, y_min)
                ])
                footprint_polys.append(poly)

            # Merge (union) all footprints
            union_poly = unary_union(footprint_polys)
            shapely_unique_area = union_poly.area

            # Print results
            #print(f"Total (sum of individual) area covered by UAVs: {total_area_covered:.2f} m^2")
            #print(f"Pairwise total overlapping area: {total_overlap_area:.2f} m^2")
            #print(f"Naive 'unique' area (sum - pairwise overlap): {naive_unique_area:.2f} m^2")
            print(f"**Accurate unique coverage (Shapely union): {shapely_unique_area:.2f} m^2**")

            self.total_shapely_area += shapely_unique_area
            self.shapely_area_count += 1

    

    def plotter(self):
        """
        Creates the gif for data analysis
        """
        # Array placeholders for all the data
        uavX = []
        uavXB = []
        uavXFull = []

        uavY = []
        uavYB = []
        uavYFull = []

        uavE = []
        uavEFull = []

        time = []
        timeFull = []

        # Init the plot
        plt.style.use('fivethirtyeight')
        fig = plt.figure(figsize=(15, 10))
        ax1 = fig.add_subplot(2, 1, 1)
        ax2 = fig.add_subplot(2, 1, 2)
        ax1.set(xlim=(-250, 160), ylim=(-5, 450))
        ax2.set(xlim=(-250, 160), ylim=(-5, 450))

        # Get all the data from the file
        data = pd.read_csv('EntropyRewritePlusVQA/data-files/entropySave.csv', delimiter=" ")

        i = 0
        while i < len(self.uavs):
            # Fill the second dimension with the number of UAVs
            uavX.append([])
            uavXB.append([])
            uavXFull.append([])
            uavY.append([])
            uavYB.append([])
            uavYFull.append([])
            uavE.append([])
            uavEFull.append([])
            i += 1

        # Live Data Setting
        i = 0
        for uav in self.uavs:
            # Grab the X,Y, Entropy, and time data
             uavXFull[i].extend(data[f"X{i+1}"])
             uavYFull[i].extend(data[f"Y{i+1}"])
             uavEFull[i].extend(data[f"EntropyDrone{i+1}"])
             i += 1
        timeFull = data["TimeDifference"]

        global iterator
        iterator = 0
        colors = ['blue', 'red', 'green', 'yellow', 'brown', 'cyan', 'deepPink']

        # This function will be called for each frame creation of the gif
        def animate(iterator):
            print(f"Timer {iterator}")
            i = 0
            while i < len(self.uavs):
                # Get the data for this iteration, appending for that 'trail' look
                uavX[i].append(uavXFull[i][iterator])
                uavY[i].append(uavYFull[i][iterator])
                uavE[i].append(uavEFull[i][iterator])
                if iterator > 1:
                    uavXB[i].append(uavXFull[i][iterator - 1])
                    uavYB[i].append(uavYFull[i][iterator - 1])
                i += 1

            time.append(timeFull[iterator])

            # Get the plots
            ax1.cla()
            ax2.cla()
            i = 0
            uLegend = []
            eLegend = []
            while i < len(self.uavs):
                # plot the data
                ax1.plot(uavXB[i], uavYB[i], linestyle="-", color=colors[i])
                ax2.plot(time, uavE[i], linestyle="-", color=colors[i])
                uLegend.append(f"UAV {i + 1}")
                eLegend.append(f"Entropy {i + 1}")
                i += 1

            # Add details to the plots
            ax1.set(xlim=([-250, 160]), ylim=([20, 200]))
            ax1.legend(uLegend)
            ax1.set_xlabel("X values")
            ax1.set_ylabel("Y Values")
            ax1.set_title('Positions')

            ax2.legend(eLegend)
            ax2.set_xlabel("Time")
            ax2.set_ylabel("Entropy")
            ax2.set_title('Entropy for each UAV')

            plt.tight_layout()

        # Create the Gif (Will finish in an ERROR by iterating too high, still works)
        ani = FuncAnimation(fig, animate, frames=10000, interval=20, repeat=False)
        ani.save('plotting.gif', writer="ffmpeg")

    def printVQAStats(self):
        self.program_end_time = time.time()
        shapely_union = self.total_shapely_area / self.shapely_area_count
        delay = self.total_delay / self.delay_count
        time_to_complete = self.program_end_time - self.program_start_time
        model_accuracy = self.vqa.get_vqa_accuracy()
        #print mean of area covered and the delay
        print(f"Unique coverage Average (Shapely union): {shapely_union:.2f} m^2")
        print(f"Average delay between images: {delay:.2f} s")
        print(f"Time to complete mission: {time_to_complete :.2f}")
        print(f"Model Accuracy: {model_accuracy}")





