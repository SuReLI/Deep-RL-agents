
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from matplotlib.widgets import Button

NSTEP = 1
def switch_nstep(event):
    global NSTEP
    NSTEP = 1 - NSTEP
    button.label.set_text("NSTEP :" + str(NSTEP))
    disp()

FILE = 1
def switch_file(event):
    global FILE
    FILE = 3 - FILE
    button2.label.set_text("File :" + str(FILE))
    disp()

def read(name, pos):
    img = mpimg.imread('results_'+name+'/Reward'+str(FILE)+'.png')
    plt.subplot(*pos)
    plt.axis('off')
    plt.imshow(img)

def disp():
    for line in range(4):
        for col in range(4):
            name = str(NSTEP) + ''.join(map(str, [col // 2, col % 2, line // 2, line % 2]))
            read(name[::-1], (4, 4, 4*line+col+1))
    plt.show()

button = Button(plt.axes([0.5, 0.0, 0.1, 0.075]), 'NSTEP :' + str(NSTEP))
button.on_clicked(switch_nstep)
button2 = Button(plt.axes([0.0, 0.0, 0.1, 0.075]), 'File :' + str(FILE))
button2.on_clicked(switch_file)

switch_nstep(None)
