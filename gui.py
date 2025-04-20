"""
Jason Carnahan
Edited By: Nathan Rayon
8/30/2021

This is the GUI for the UAV project
"""
from tkinter import *  # the basic tkinter GUI library
from tkinter.ttk import *  # the tkinter themed widgets

from PIL import ImageTk  # for loading and manipulating images onto the GUI
from win32api import GetSystemMetrics  # to get the monitor dimensions
from EntropyRewritePlusVQA.models.VQAModel import VQAModel
from jason_python_lib import *  # my helper functions

from Strategies.AirSimStrategy import AirSimStrategy  # all the strategies I want to use
from CentralControl import CentralControl  # Simulation control
from VQA import VQA

# VQA Imports
import torch
import torchvision
from torchvision import datasets, models
from torchvision.transforms import functional as FT
from torchvision import transforms as T
from torch import nn, optim
from torch.nn import functional as F
from torch.utils.data import DataLoader, sampler, random_split, Dataset
from PIL import Image
import cv2
import albumentations as A
from albumentations.pytorch import ToTensorV2
from torchvision.models.detection.rpn import AnchorGenerator

class App:
    """
    This is the GUI class for the UAV program
    """

    def __init__(self):
        # Load default parameters
        self.win_width = GetSystemMetrics(0)  # Get monitor width
        self.win_height = GetSystemMetrics(1)  # Get monitor height
        self.strategies = ["AirSim", "Tello"]  # All the strategies available
        self.inputs = [
            "1. Choose the number of UAVs:",
            "2. Choose the separation distance:",
            "3. Choose the row length:"
        ]  # All the necessary inputs

        # Load the window
        self.window = Tk()  # Default window creation
        self.window.title("UAV Entropy Strategy Selection")
        self.window.geometry(f"{int(self.win_width * .50)}x{int(self.win_height * .50)}")

        # Style the window
        self.style = Style(self.window)
        self.style.configure('default.TButton', font=12)  # create a 'class name' and style for the class
        self.style.configure('default.TRadiobutton', font=12)
        self.style.theme_use("vista")  # general theme

        # Create tkinter variables
        self.strategy_choice = IntVar(self.window)  # Used in the radio buttons and track user selection
        self.strategy_choice = -1

        self.number_of_uavs = StringVar(self.window)  # records the user's input for the num of uav
        self.number_of_uavs.set(None)

        self.separating_distance = StringVar(self.window)  # records the user's input for distance between uavs
        self.separating_distance.set(None)

        self.row_length = StringVar(self.window)  # records the user's input for the row length
        self.row_length.set(None)

        # Create the menu bar
        self.create_menu()

        # Place content on the window
        self.title = Label(
            self.window,
            text="Welcome to the UAV Entropy Strategy selection menu.\nPlease select what UAV strategy you would like to use.",
            font=('Arial', 18)
        )
        self.title.pack()

        # Place all the options onscreen
        self.load_options()

        # Load window
        self.window.mainloop()

    def create_menu(self):
        """
        This function will create the menu at the top of the screen so it feels more modern
        :return: none
        """

        # Create the menu bar
        menubar = Menu(self.window)
        file_menu = Menu(menubar, tearoff=0)

        # add buttons to file menu
        file_menu.add_command(label="Exit", command=self.window.quit)

        # Place the file tab on the menu
        menubar.add_cascade(label="File", menu=file_menu)

        # Add the menu to the window
        self.window.configure(menu=menubar)

    def load_options(self):
        """
        Creates a button for each strategy option and dynamically calls its corresponding function
        :param strategies: list of all the option available
        :return: none
        """

        choice_separation = 20  # Distance between each strategy option

        # Loop over the strategy options and add a radio button for each option
        # The function is a dynamic call that will set the option beased on the name of the choice
        for idx, strategy in enumerate(self.strategies):
            button = Radiobutton(
                self.window,
                text=strategy,
                variable=self.strategy_choice,
                value=idx,
                style='default.TRadiobutton',
                command=lambda i=idx: self.set_strategy(f"{i}")
            )
            button.place(relx=.33, rely=idx / choice_separation + .15)

        # Some variables to help position the options together
        bottom_labels_x = .37
        bottom_textbox_x = bottom_labels_x + .25
        bottom_starting_y = 0.65

        # Loop over the options and create an entry field for each label
        for idx, input_label in enumerate(self.inputs):

            # Label for the asked input
            label = Label(
                self.window,
                text=input_label,
                font=("Arial", 12)
            )
            label.place(relx=bottom_labels_x, rely=idx / choice_separation + bottom_starting_y)

            # Each text box needs a different variable assigned to it.
            if idx == 0:
                entry_var = self.number_of_uavs
            elif idx == 1:
                entry_var = self.separating_distance
            elif idx == 2:
                entry_var = self.row_length
            else:
                entry_var = 'ERROR'

            # Entry field for the input
            text_box = Entry(
                self.window,
                width=5,
                font=("Arial", 12),
                textvariable=entry_var
            )
            text_box.place(relx=bottom_textbox_x, rely=idx / choice_separation + bottom_starting_y)

        # Help label for the image
        help_label = Label(
            text="UAV Selection Help"
        )
        help_label.place(relx=.55, rely=.12)
        create_tooltip(
            help_label,
            "System will start from the top left and fill UAVs moving right until row length has been met,"
            "\nthen will move to the next row and continue until all UAVs have been placed."
        )

        # Help image to visualize the asked inputs
        #help_canvas = Canvas(self.window, width=300, height=250)
        #help_canvas.place(relx=.55, rely=0.15)

        #help_image = open_relative_path('UAV Program Help.png', (300, 250))
        #render = ImageTk.PhotoImage(help_image)
        #help_canvas.create_image(0, 0, anchor=NW, image=render)
        #help_canvas.image = render

        # Button to run the simulation
        submit = Button(
            self.window,
            text="Submit",
            style='default.TButton',
            command=lambda: self.submit_check()
        )
        submit.place(relx=.45, rely=.85)
        create_tooltip(submit, "Make sure to choose a strategy, and\nuse integers only.")

    def set_strategy(self, strategy_index):
        """
        Sets the currently selected strategy. -1 be default means none is selected
        :param strategy_index: the index position of the strategy chosen
        :return: none
        """
        self.strategy_choice = strategy_index

    def submit_check(self):
        """
        The final check before the simulation starts
        :return:
        """
        # Only run if a strategy is chosen
        if self.strategy_choice != -1:

            # dynamically load strategy on choice
            s = self.strategies.__getitem__(int(self.strategy_choice))
            strategy_class = globals()[f"{s}Strategy"]
            strategy = strategy_class()

            # Ensure that the necessary inputs are valid integers
            try:
                num_of_uav = int(self.number_of_uavs.get())
                sep_dist = int(self.separating_distance.get())
                row_length = int(self.row_length.get())

                if num_of_uav < 1 or sep_dist < 1 or row_length < 1:
                    print("\033[91m" + "Use positive integers only.")
                    return

            except ValueError:
                print("\033[91m" + "Use integers only.")
                return

            self.camera_labels = []
            for i in range(num_of_uav):
                camera_label = Label(self.window, text=f"Camera{i}")
                camera_label.place(relx=0.5, rely=0.5)
                # camera_label.pack()
                self.camera_labels.append(camera_label)

            # All inputs are valid, run
            self.run_simulation(strategy, num_of_uav, sep_dist, row_length)

        else:
            print("\033[91m" + "Choose a strategy")

    def run_simulation(self, strategy, num_of_uav, sep_distance, row_length):
        """
        Starts the central control with chosen strategy and inits the system
        """
        n_classes = 9
        cv_model = models.detection.fasterrcnn_mobilenet_v3_large_fpn(pretrain=True)
        in_features = cv_model.roi_heads.box_predictor.cls_score.in_features # we need to change the head

        cv_model.roi_heads.box_predictor = models.detection.faster_rcnn.FastRCNNPredictor(in_features, n_classes)
        cv_model.load_state_dict(torch.load(r'F:\UAVProject\AirSim\PythonClient\EntropyRewritePlusVQA\models\rcnn_uav_model.pt'))
        cv_model.eval()

        model = VQAModel(26, 32, 21)
        model.load_state_dict(torch.load(r'F:\UAVProject\AirSim\PythonClient\EntropyRewritePlusVQA\models\model_vqa_sequential_uav_new_model.pt'))
        model.eval()
        
        vqa = VQA(cv_model, model)
        c = CentralControl(strategy=strategy, vqa=vqa)  # Create the central control
        
        c.init_system(
            num_of_uavs=num_of_uav,
            separating_distance=sep_distance,
            row_length=row_length
        )  # create internal structure of UAVs to track



        
        connected = c.connect_to_environment()  # connect to whatever environment they are in
        
        if connected:
            c.uav_init()  # give UAVs their initial command
            c.entropyFormation()  # run the entropy formation
            c.orbit()  # run orbit
            #c.plotter()
           

            

        c.close_connection()  # formation complete, close connection and return to gui

    


    
      
       
            




