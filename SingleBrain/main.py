
from time import sleep

import parameters
from Agent import Agent
from Brain import Brain, Optimizer
from Displayer import DISPLAYER


if __name__ == '__main__':

    main_agent = Agent(0, None, render=True)
    brain = Brain(main_agent.state_size, main_agent.action_size)

    main_agent.brain = brain

    agents = [Agent(i+1, brain) for i in range(parameters.THREADS)]
    optimizers = [Optimizer(brain) for i in range(parameters.OPTIMIZERS)]

    for optimizer in optimizers:
        optimizer.start()

    for agent in agents:
        agent.start()

    try:
        sleep(parameters.LIMIT_RUN_TIME)
    except KeyboardInterrupt:
        print("End of the training")

    for agent in agents:
        agent.stop()
    for agent in agents:
        agent.join()

    for optimizer in optimizers:
        optimizer.stop()
    for optimizer in optimizers:
        optimizer.join()

    print("Training finished")
    try:
        main_agent.run(10)
    except KeyboardInterrupt:
        print("End of the session")

    DISPLAYER.disp_all()
    DISPLAYER.disp_one()
    DISPLAYER.disp_seq()
    main_agent.stop()
