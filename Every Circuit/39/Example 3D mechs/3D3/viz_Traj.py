import numpy as np
import matplotlib.pyplot as plt

runs = 5
# traj = np.loadtxt("./biastrack_HPontest.dat").reshape(runs,-1,3)
traj = np.loadtxt("../teststatestrack.dat").reshape(-1,3)

backup=1000
# ax = plt.figure().add_subplot(projection='3d')
# for run in range(runs):
#     ax.plot(traj[run,-backup:,0],traj[run,-backup:,1],traj[run,-backup:,2])

for i in range(3):
    plt.plot(traj[-backup:,i])
plt.show()