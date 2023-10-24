import matplotlib.pyplot as plt
import numpy as np
from utils.step_reference import accel_decel_step, accel_step

dt = 0.1
v_low = 2
t_1 = 2
T_1 = 10

T_2 = 20
v_high = 4
accel_start = 1
accel_end = 3
decel_start = 11
decel_end = 13

x_ref_1, t_range_1 = accel_step(v_low, t_1, dt, T_1)
x_ref_2, t_range_2 = accel_decel_step(T_2, dt, accel_start, accel_end, 
                                      decel_start, decel_end, v_low, v_high)

# fig, ax = plt.subplots(3, 1, figsize=(5, 8), sharex=True)
# fig.subplots_adjust(top=0.94)
# for i in range(3):
#     ax[i].plot(t_range_1, x_ref_1[i, :])
# fig.suptitle('Reference trajectory 1')
# ax[0].set_ylabel('position [m]')
# ax[1].set_ylabel('velocity [m/s]')
# ax[2].set_ylabel('acceleration [m/s^2]')
# ax[2].set_xlabel('time [s]')

# fig, ax = plt.subplots(3, 1, figsize=(5, 8), sharex=True)
# fig.subplots_adjust(top=0.94)
# for i in range(3):
#     ax[i].plot(t_range_2, x_ref_2[i, :])
# fig.suptitle('Reference trajectory 2')
# ax[0].set_ylabel('position [m]')
# ax[1].set_ylabel('velocity [m/s]')
# ax[2].set_ylabel('acceleration [m/s^2]')
# ax[2].set_xlabel('time [s]')

x_ref_2[0, :] += x_ref_1[0, -1]
x_ref = np.c_[x_ref_1[:, :-1], x_ref_2]
t_range = np.r_[t_range_1[:-1], t_range_2 + t_range_1[-1]]

fig, ax = plt.subplots(3, 1, figsize=(5, 8), sharex=True)
fig.subplots_adjust(top=0.94)
for i in range(3):
    ax[i].plot(t_range, x_ref[i, :])
fig.suptitle('Reference trajectory')
ax[0].set_ylabel('position [m]')
ax[1].set_ylabel('velocity [m/s]')
ax[2].set_ylabel('acceleration [m/s^2]')
ax[2].set_xlabel('time [s]')

plt.show()