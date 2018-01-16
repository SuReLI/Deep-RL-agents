
from tkinter import *
import settings


STOP = False

REQUEST_EP_REWARD = False
REQUEST_PLOT = False
REQUEST_RENDER = False

AUTO_EP_REWARD = True
AUTO_PLOT = True
AUTO_RENDER = True

FREQ_EP_REWARD = settings.EP_REWARD_FREQ
FREQ_PLOT = settings.PLOT_FREQ
FREQ_RENDER = settings.RENDER_FREQ


def ep_reward_dislay(nb_ep):
    global REQUEST_EP_REWARD
    output = display(REQUEST_EP_REWARD, AUTO_EP_REWARD,
                     FREQ_EP_REWARD, settings.EP_REWARD_FREQ, nb_ep)
    REQUEST_EP_REWARD = False
    return output

def plot_display(nb_ep):
    global REQUEST_PLOT
    output = display(REQUEST_PLOT, AUTO_PLOT,
                     FREQ_PLOT, settings.PLOT_FREQ, nb_ep)
    REQUEST_PLOT = False
    return output

def render_display(nb_ep):
    global REQUEST_RENDER
    output = display(REQUEST_RENDER, AUTO_RENDER,
                     FREQ_RENDER, settings.RENDER_FREQ, nb_ep)
    REQUEST_RENDER = False
    return output

def display(request, auto, freq, settings_freq, nb_ep):
    if not settings.DISPLAY:
        return False

    if settings.GUI:
        if request:
            return True
        elif auto and freq > 0:
            return nb_ep % freq == 0
        return False
    return settings_freq > 0 and nb_ep % settings_freq == 0


def main():
    
    window = Tk()
    window.title("Control Panel")
    window.attributes('-topmost', 1)

    def stop_run():
        global STOP
        STOP = True
        window.destroy()

    def set_ep_reward_freq(event):
        global FREQ_EP_REWARD
        try:FREQ_EP_REWARD = int(ep_reward_freq.get())
        except:pass

    def update_ep_reward():
        global REQUEST_EP_REWARD
        REQUEST_EP_REWARD = True

    def switch_ep_reward():
        global AUTO_EP_REWARD
        AUTO_EP_REWARD = not AUTO_EP_REWARD
        on_off = ("Off" if AUTO_EP_REWARD else "On")
        ep_reward_switch.config(text='Set auto update ' + on_off)


    def set_plot_freq(event):
        global FREQ_PLOT
        try:FREQ_PLOT = int(plot_freq.get())
        except:pass

    def update_plot():
        global REQUEST_PLOT
        REQUEST_PLOT = True

    def switch_plot():
        global AUTO_PLOT
        AUTO_PLOT = not AUTO_PLOT
        on_off = ("Off" if AUTO_PLOT else "On")
        plot_switch.config(text='Set auto update ' + on_off)


    def set_render_freq(event):
        global FREQ_RENDER
        try:FREQ_RENDER = int(render_freq.get())
        except:pass

    def update_render():
        global REQUEST_RENDER
        REQUEST_RENDER = True

    def switch_render():
        global AUTO_RENDER
        AUTO_RENDER = not AUTO_RENDER
        on_off = ("Off" if AUTO_RENDER else "On")
        render_switch.config(text='Set auto update ' + on_off)


    ep_reward_label = Label(window, text="Ep reward")
    ep_reward_update = Button(window, text='Display', command=update_ep_reward)
    ep_reward_switch = Button(window, text='Set auto display Off', command=switch_ep_reward)
    ep_reward_freq = Entry(window, justify='center')
    ep_reward_freq.bind("<Return>", set_ep_reward_freq)

    plot_label = Label(window, text='PLOT')
    plot_update = Button(window, text='Update', command=update_plot)
    plot_switch = Button(window, text='Set auto update Off', command=switch_plot)
    plot_freq = Entry(window, justify='center')
    plot_freq.bind("<Return>", set_plot_freq)

    render_label = Label(window, text='RENDER')
    render_update = Button(window, text='Render', command=update_render)
    render_switch = Button(window, text='Set auto render Off', command=switch_render)
    render_freq = Entry(window, justify='center')
    render_freq.bind("<Return>", set_render_freq)

    stop_button = Button(window, text='Stop the Run', command=stop_run)

    ep_reward_label.grid(column=1, row=1)
    ep_reward_update.grid(column=1, row=2)
    ep_reward_switch.grid(column=1, row=3)
    ep_reward_freq.grid(column=1, row=4)

    plot_label.grid(column=2, row=1)
    plot_update.grid(column=2, row=2)
    plot_switch.grid(column=2, row=3)
    plot_freq.grid(column=2, row=4)

    render_label.grid(column=3, row=1)
    render_update.grid(column=3, row=2)
    render_switch.grid(column=3, row=3)
    render_freq.grid(column=3, row=4)

    stop_button.grid(column=1, row=5, columnspan=3)

    window.mainloop()


if __name__ == '__main__':
    main()
