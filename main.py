from model_builder import launch
import numpy as np
import os
import wx
__author__ = 'Itai'

import matplotlib.pyplot as plt
plt.rcdefaults()


APP_SIZE_X = 500
APP_SIZE_Y = 600

d = {0: 'zero', 1: 'one', 2: 'two', 3: 'three', 4: 'four',
     5: 'five', 6: 'six', 7: 'seven', 8: 'eight', 9: 'nine'}


class GUI(wx.Dialog):
    """
    Yada Yada
    """

    def __init__(self, parent, id, title):
        """
        Initialize the GUI
        """
        wx.Dialog.__init__(self, parent, id, title,
                           size=(APP_SIZE_X, APP_SIZE_Y))
        self.locale = wx.Locale(wx.LANGUAGE_ENGLISH)
        self.font = wx.Font(15, wx.DECORATIVE, wx.NORMAL, wx.BOLD)
        self.current_directory = os.getcwd()
        self.path = None

        self.choose_file_button = wx.Button(
            self, 1, 'Choose File', (20, 130), size=(200, 50))
        self.choose_file_button.SetFont(self.font)

        self.start_button = wx.Button(
            self, 2, 'Start', (20, 190), size=(200, 50))
        self.start_button.SetFont(self.font)

        self.close_button = wx.Button(
            self, 3, 'Close', (20, 250), size=(200, 50))
        self.close_button.SetFont(self.font)

        self.instructions = wx.StaticText(
            self, label="Please choose the PNG consisting the digit\nyou wish to test.",
            pos=(20, 60))
        self.instructions.SetFont(self.font)

        self.image_txt = wx.StaticText(self, label="", pos=(20, 320))
        self.prediction = wx.StaticText(self, label="", pos=(250, 320))

        wx.FileDialog(self, "Open", style=wx.FD_OPEN)

        self.picture = wx.StaticBitmap(self, pos=(20, 350))

        self.neural_gen = None

        self.save_graph_button = wx.Button(
            self, 4, 'Save Graph', (250, 130), size=(200, 50))
        self.save_graph_button.SetFont(self.font)

        self.Bind(wx.EVT_BUTTON, self.choose_file, id=1)
        self.Bind(wx.EVT_BUTTON, self.start, id=2)
        self.Bind(wx.EVT_BUTTON, self.on_close, id=3)
        self.Bind(wx.EVT_BUTTON, self.save_graph, id=4)

        self.Centre()
        self.ShowModal()

    def on_close(self, event):
        """
        Close the program
        """
        self.Destroy()

    def choose_file(self, event):
        """
        Create and show the Open FileDialog
        """
        # Create the dialog
        dlg = wx.FileDialog(
            self, message="Choose a file",
            defaultDir=self.current_directory,
            defaultFile="",
            style=wx.FD_OPEN | wx.FD_MULTIPLE | wx.FD_CHANGE_DIR
        )
        # Show the dialog
        if dlg.ShowModal() == wx.ID_OK:
            self.path = dlg.GetPaths()[0]
            self.neural_gen = launch(self.path)

        # Destroy the dialog
        dlg.Destroy()

    def start(self, event):
        """
        Launch the digit identifier
        """
        if self.path is not None:
            # Show answer and confidence
            prediction, confidence = next(self.neural_gen)[:2]
            self.prediction.SetLabel(
                f'Prediction: {d[prediction]}\nConfidence: {(confidence * 100):.2f}%')
            self.prediction.SetFont(self.font)

            # Show the image
            self.image_txt.SetLabel('Your image:')
            self.image_txt.SetFont(self.font)

            # Load the image
            bitmap = wx.Bitmap(self.path)
            image = bitmap.ConvertToImage()
            image = image.Scale(100, 100, wx.IMAGE_QUALITY_HIGH)
            result = wx.Bitmap(image)

            # Display the image
            self.picture.SetBitmap(result)
        else:
            # No file chosen
            raise Exception('No file chosen')

    def save_graph(self, event):
        """
        Save the graph
        """
        if self.path is not None:
            # Ready the parameters
            digits = [d[i] for i in range(10)]
            y_pos = np.arange(len(digits))
            probabilities = next(self.neural_gen)[2]

            # Create the graph
            plt.bar(y_pos, probabilities, align='center', alpha=0.5)
            plt.xticks(y_pos, digits)
            plt.ylabel('Probability')

            # Set y-axis to be from 0 to 1
            plt.setp(plt.gca(), ylim=(0, 1))

            # Save the graph
            plt.savefig('../graphs/graph.png')

            # Close the graph
            plt.clf()
        else:
            # No file chosen
            raise Exception('No file chosen')


def main():
    """
    Launch the GUI
    """
    app = wx.App(0)
    GUI(None, -1, 'Hello There')
    app.MainLoop()


if __name__ == '__main__':
    main()
