
"""
This module provides an easy way to interact in real-time and graphically with a
running RL algorithm to trigger different features like saving the network
weights or rendering an episode.

This interface is implemented with tkinter.
"""

from tkinter import *
from settings import Settings


class Feature:
    """
    A Feature corresponds to an action to be realized during the training of an
    agent. A Feature can be activated by hand with a button, or automatically
    with a given frequency.
    """
    nb_features = 0

    def __init__(self, name, freq, text="update", render_needed=False):
        """
        A Feature is defined by :
            name: a word describing its role
            freq: a frequency to automatically activate itself
                            Usually defined in a 'settings' file
                            A frequency of <= 0 means the Feature is not
                            activated
            text: the text to be written on the activation button
            render_needed: whether the DISPLAY settings has to be True to
                            activate the Feature or not
        """
        self.name = name
        self.text = text.lower()
        self.request = False
        self.auto = True
        self.freq = freq
        self.render_needed = render_needed

    def build(self, window):
        """
        Builds the tkinter objects and displays them in the window given in
        parameter.
        """
        self.label = Label(window, text=self.name.upper())
        self.update = Button(window, text=self.text.capitalize(),
                             command=self.update_cmd)
        self.switch = Button(window, text='Set auto {} Off'.format(self.text),
                             command=self.switch_cmd)
        self.freq_entry = Entry(window, justify='center')
        self.freq_entry.bind("<Return>", self.set_freq)

        self.label.grid(column=Feature.nb_features, row=0)
        self.update.grid(column=Feature.nb_features, row=1)
        self.switch.grid(column=Feature.nb_features, row=2)
        self.freq_entry.grid(column=Feature.nb_features, row=3)
        Feature.nb_features += 1

    def update_cmd(self):
        """
        The command bound to the update button to activate the Feature once.
        """
        self.request = True

    def switch_cmd(self):
        """
        The command bound to the switch button to activate or deactivate the
        auto-activation of the Feature.
        """
        self.auto = not self.auto
        on_off = ('Off' if self.auto else 'On')
        self.switch.config(text='Set auto {} {}'.format(self.text, on_off))

    def set_freq(self, event):
        """
        The function that reads the freq Entry when the user presses Return and
        set the auto-activation frequency to that new value.
        """
        try:
            self.freq = int(self.freq_entry.get())
            self.freq_entry.delete(0, END)
        except:pass

    def get(self, nb_ep):
        """
        Function that returns whether the Feature must be activated or not.
        Returns True if :
            - The GUI is ON and
                * the user pressed the request button
                * the automatic activation is ON and the number of episodes
                    since the last activation is equal to the Feature's
                    frequency
            - The GUI is OFF and the number of episodes since the last
                activation is equal to the Feature's frequency.
        """
        if self.render_needed and not Settings.DISPLAY:
            return False

        if Settings.GUI:
            if self.request:
                self.request = False
                return True
            elif self.auto and self.freq > 0:
                return nb_ep % self.freq == 0
            return False
        return self.freq > 0 and nb_ep % self.freq == 0

class NullFeature:
    def get(self, *args, **kwargs):
        return False

class Interface:
    """
    Defines the main tkinter window that wraps the different Features.
    """

    def __init__(self, features):
        """
        Build the tkinter window.

        Args:
            features: a list of string with the names of the features to be
                        activated.
                The possible features are :
                    - ep_reward : print informations about the current episode
                    - plot : plot the episode rewards graph
                    - plot_distrib : plot the real-time evolution of the Q-value
                      distribution
                    - render : render the environment during an episode
                    - gif : save a GIF of an episode
                    - save : save the weights of the network
        """

        features = " ".join(features).lower()
        self.list_features = []

        if 'ep_reward' in features:
            self.ep_reward = Feature('EP REWARD', Settings.EP_REWARD_FREQ, 'display')
            self.list_features.append(self.ep_reward)

        if ' plot ' in features:
            self.plot = Feature('PLOT', Settings.PLOT_FREQ, 'update')
            self.list_features.append(self.plot)

        if 'plot_distrib' in features:
            self.plot_distrib = Feature('PLOT DISTRIB', 0, 'update', True)
            self.list_features.append(self.plot_distrib)

        if 'render' in features:
            self.render = Feature('RENDER', Settings.RENDER_FREQ, 'render', True)
            self.list_features.append(self.render)

        if 'gif' in features:
            self.gif = Feature('GIF SAVER', Settings.GIF_FREQ, 'snap')
            self.list_features.append(self.gif)

        if 'save' in features:
            self.save = Feature('MODEL SAVER', Settings.SAVE_FREQ, 'save')
            self.list_features.append(self.save)


        self.STOP = False

    def stop_run(self):
        """
        Method that kills the current GUI and request the algorithm to stop.
        """
        self.STOP = True
        self.window.destroy()

    def run(self):
        """
        Method that display the GUI and run the main event-loop.
        """

        if Settings.GUI:
            self.window = Tk()
            self.window.title("Control Panel")
            self.window.attributes('-topmost', 1)

            for feature in self.list_features:
                feature.build(self.window)

            # Build the stop button
            stop_button = Button(self.window, text='Stop the Run', command=self.stop_run)
            stop_button.grid(column=0, row=4, columnspan=Feature.nb_features, sticky='NSEW')

            self.window.mainloop()

    def __getattr__(self, attr):
        """
        If the user wants to access a feature that doesn't exist (not
        implemented or not declared in the feature initialization list), we
        create a new Null Feature with a method get that always return False.
        """
        setattr(self, attr, NullFeature())
        return getattr(self, attr)
