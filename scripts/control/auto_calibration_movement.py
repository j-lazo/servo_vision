from control import mega_communication as mc
from general_functinos import pose_calculation as pc
import numpy as np
from time import sleep

pre_act_signal = [[7, 4, 0],
                  [-6, -2, 0],
                  [3, 2, 0],
                  [-8, -4, 0],
                  [2, 9, 0],
                  [-5, 8, 0],
                  [-10, 2, 0],
                  [-10, 2, 1],
                  [7, 7, 1],
                  [-6, -3, 1],
                  [3, 5, 1],
                  [-6, 8, 1],
                  [3, 2, 1],
                  [-13, 13, 1],
                  [-13, 13, 2],
                  [-4, 4, 2],
                  [8, 3, 2],
                  [8, 13, 2],
                  [-6, -6, 2],
                  [-8, 8, 2],
                  [2, -2, 2]]

global act_joint_variables
act_joint_variables = np.zeros((len(pre_act_signal), 3))

arduino_port_1 = mc.serial_initialization()

for i in range(len(pre_act_signal)):
    mc.serial_actuate(arduino_port_1)
    sleep(10)  # temporary
    current_act_joint_variable = mc.serial_request(arduino_port_1)
    act_joint_variables = mc.store_act_joint_variables(np.array(current_act_joint_variable), act_joint_variables, i)

tf_mat = pc.tf_calculation(act_joint_variables)
pc.save_pose_txt(tf_mat)





