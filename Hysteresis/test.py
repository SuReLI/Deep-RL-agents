
import random
from time import sleep
from HystEnv import HystEnv

env = HystEnv()

# env.reset()
# for j in range(600, -1, -1):
# 	s, r, d, _ = env.step(j//200)
# 	env.render()
# 	print("Reward: ", r)
# 	sleep(0.05)
	# sleep(0.002)

for i in range(100):
	env.reset()
	for j in range(1000):
		action = random.randint(0, 2)
		if j%10 == 0:
			action = 2
		a = env.step(action)
		print("Reward: ", a[1])
		env.render()
		sleep(0.03)
