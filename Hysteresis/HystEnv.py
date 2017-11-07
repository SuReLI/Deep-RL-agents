
import tkinter
import random


# RANDOM_STALL = 1/2500

window = tkinter.Tk()
canvas = tkinter.Canvas(window, width=300, height=100, bg='white')

background = [canvas.create_rectangle(0, 0, 200, 100, fill='green'),
              canvas.create_rectangle(200, 0, 300, 100, fill='red')]

pos = random.randint(50, 150)
cursor = canvas.create_rectangle(pos - 2, 0, pos + 2, 100, fill='black')

SIGMA = 20

canvas.grid()


class HystEnv:
    """Class that simulate a very simple environment with an hysteresis"""

    def __init__(self):
        self.nb_step = 0

    def reset(self):
        global pos

        self.speed = 0
        self.stalled = False
        self.render_change = True
        self.stall_limit = 200 + random.normalvariate(0, SIGMA) // 2

        pos = random.randint(50, 150)
        return [pos, self.speed]

    def step(self, action):
        global pos

        if action == 0:
            pos = max(pos - 1, 0)
        elif action == 2:
            pos = min(pos + 1, 300)

        canvas.coords(cursor, (pos - 2, 0, pos + 2, 100))

        reward = pos
        if (self.stalled and pos > 100) or (not self.stalled and pos > self.stall_limit):
            reward = 0
        if not self.stalled and 0 < self.stall_limit - pos <= 30:
        	reward += (self.stall_limit - pos)**2

        if self.stalled and pos < 100:
            self.stalled = False
            self.stall_limit = 200 + random.normalvariate(0, SIGMA) // 2
            self.render_change = True

        if not self.stalled and pos > self.stall_limit:
            self.stalled = True
            self.stall_limit = 100
            self.render_change = True

        self.nb_step += 1

        if not self.stalled and self.nb_step % 20 == 0:
            self.stall_limit = 200 + random.normalvariate(0, SIGMA) // 2
            self.render_change = True

        self.speed = (self.speed + 3 * reward) // 4

        return [pos, self.speed], reward / 600, False, None

    def render(self):
        if self.render_change:
            canvas.coords(background[0], (0, 0, self.stall_limit, 100))
            canvas.coords(background[1], (self.stall_limit, 0, 300, 100))
            self.render_change = False

        window.update()

    def close(self):
        pass
