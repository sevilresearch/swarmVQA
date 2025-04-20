from tkinter import *
from tkinter.ttk import *
import sys
import os
from PIL import Image


class ToolTip(object):
    """
    Class for the popup tooltips that appear
    """

    def __init__(self, widget):
        self.widget = widget
        self.tip_window = None
        self.id = None
        self.x = self.y = 0

    def showtip(self, text):
        """Display text in tooltip window"""
        self.text = text
        if self.tip_window or not self.text:
            return
        x, y, cx, cy = self.widget.bbox("insert")
        x = x + self.widget.winfo_rootx() + 27
        y = y + cy + self.widget.winfo_rooty() + 27
        self.tip_window = tw = Toplevel(self.widget)
        tw.wm_overrideredirect(1)
        tw.wm_geometry("+%d+%d" % (x, y))
        try:
            # For Mac OS
            tw.tk.call("::tk::unsupported::MacWindowStyle",
                       "style", tw._w,
                       "help", "noActivates")
        except TclError:
            pass
        label = Label(tw, text=self.text, justify=LEFT,
                      background="#ffffff", relief=SOLID, borderwidth=1,
                      font=("Segoe UI", "10", "normal"))
        label.pack(ipadx=1)

    def hidetip(self):
        tw = self.tip_window
        self.tip_window = None
        if tw:
            tw.destroy()


def create_tooltip(widget, text):
    """
    The constructor to make a tooltip for any widget.
    :param widget: the widget to attach the tooltip
    :param text:  the text to apply to the tooltip
    """
    toolTip = ToolTip(widget)

    def enter(event):
        toolTip.showtip(text)

    def leave(event):
        toolTip.hidetip()

    widget.bind('<Enter>', enter)
    widget.bind('<Leave>', leave)


def open_relative_path(path, resized=(300, 300)):
    """
    Determines if a tkinter app is in an EXE or running in command line and will return an image object using the correct path
    :param path: the path to the image
    :param resized: size of the image
    :return: PIL image object
    """
    x, y = resized

    if getattr(sys, "frozen", False) and hasattr(sys, '_MEIPASS'):
        # application running in exe
        return Image.open(os.path.abspath(os.path.join(os.path.dirname(__file__), path))).resize((x, y))
    else:
        # application running in cmd line
        return Image.open(path).resize((x, y))
