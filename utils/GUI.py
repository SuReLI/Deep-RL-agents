
from tkinter import *
from settings import Settings


class Feature:
    nb_column = 0

    def __init__(self, name, settings_freq, text):
        self.name = name
        self.text = text.lower()
        self.request = False
        self.auto = True
        self.settings_freq = settings_freq
        self.freq = settings_freq

    def build(self, window):
        self.label = Label(window, text=self.name.upper())
        self.update = Button(window, text=self.text.capitalize(), command=self.update_cmd)
        self.switch = Button(window, text='Set auto {} Off'.format(self.text), command=self.switch_cmd)
        self.freq_entry = Entry(window, justify='center')
        self.freq_entry.bind("<Return>", self.set_freq)

        self.label.grid(column=Feature.nb_column, row=0)
        self.update.grid(column=Feature.nb_column, row=1)
        self.switch.grid(column=Feature.nb_column, row=2)
        self.freq_entry.grid(column=Feature.nb_column, row=3)
        Feature.nb_column += 1

    def update_cmd(self):
        self.request = True

    def switch_cmd(self):
        self.auto = not self.auto
        on_off = ('Off' if self.auto else 'On')
        self.switch.config(text='Set auto {} {}'.format(self.text, on_off))

    def set_freq(self, event):
        try:
            self.freq = int(self.freq_entry.get())
            self.freq_entry.delete(0, END)
        except:pass

    def get(self, nb_ep):
        if not Settings.DISPLAY:
            return False

        if Settings.GUI:
            if self.request:
                self.request = False
                return True
            elif self.auto and self.freq > 0:
                return nb_ep % self.freq == 0
            return False
        return self.settings_freq > 0 and nb_ep % self.settings_freq == 0


class Interface:

    def __init__(self, features):

        features = " ".join(features).lower()
        self.list_features = []

        if 'ep_reward' in features:
            self.ep_reward = Feature('EP REWARD', Settings.EP_REWARD_FREQ, 'display')
            self.list_features.append(self.ep_reward)

        if ' plot ' in features:
            self.plot = Feature('PLOT', Settings.PLOT_FREQ, 'update')
            self.list_features.append(self.plot)

        if 'plot_distrib' in features:
            self.plot_distrib = Feature('PLOT DISTRIB', 0, 'update')
            self.list_features.append(self.plot_distrib)

        if 'render' in features:
            self.render = Feature('RENDER', Settings.RENDER_FREQ, 'render')
            self.list_features.append(self.render)

        if 'gif' in features:
            self.gif = Feature('GIF SAVER', Settings.GIF_FREQ, 'snap')
            self.list_features.append(self.gif)

        if 'save' in features:
            self.save = Feature('MODEL SAVER', Settings.SAVE_FREQ, 'save')
            self.list_features.append(self.save)


        self.STOP = False

    def stop_run(self):
        self.STOP = True
        self.window.destroy()

    def run(self):

        if Settings.GUI:
            self.window = Tk()
            self.window.title("Control Panel")
            self.window.attributes('-topmost', 1)

            for feature in self.list_features:
                feature.build(self.window)

            stop_button = Button(self.window, text='Stop the Run', command=self.stop_run)
            stop_button.grid(column=0, row=4, columnspan=Feature.nb_column, sticky='NSEW')

            self.window.mainloop()
