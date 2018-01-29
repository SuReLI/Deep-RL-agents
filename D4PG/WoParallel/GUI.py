
from tkinter import *
import settings


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
        try:self.freq = int(self.freq_entry.get())
        except:pass

    def get(self, nb_ep):
        if not settings.DISPLAY:
            return False

        if settings.GUI:
            if self.request:
                self.request = False
                return True
            elif self.auto and self.freq > 0:
                return nb_ep % self.freq == 0
            return False
        return self.settings_freq > 0 and nb_ep % self.settings_freq == 0


STOP = False

ep_reward = Feature('EP REWARD', settings.EP_REWARD_FREQ, 'display')
plot = Feature('PLOT', settings.PLOT_FREQ, 'update')
plot_distrib = Feature('PLOT DISTRIB', 0, 'update')
gif = Feature('GIF SAVER', settings.GIF_FREQ, 'snap')
render = Feature('RENDER', settings.RENDER_FREQ, 'render')
save = Feature('MODEL SAVER', settings.SAVE_FREQ, 'save')


def main():

    window = Tk()
    window.title("Control Panel")
    window.attributes('-topmost', 1)

    def stop_run():
        global STOP
        STOP = True
        window.destroy()

    ep_reward.build(window)
    plot.build(window)
    plot_distrib.build(window)
    render.build(window)
    gif.build(window)
    save.build(window)

    stop_button = Button(window, text='Stop the Run', command=stop_run)
    stop_button.grid(column=0, row=4, columnspan=Feature.nb_column, sticky='NSEW')

    window.mainloop()
