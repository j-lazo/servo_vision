from scipy.optimize import fsolve
import math
from math import cos, sin, tan, sqrt, pi


def world_to_config(x, *world_coordinate):
    body_length = 70
    x_w, y_w, z_w = world_coordinate
    return (
        body_length * (1 - cos(x[0])) * cos(x[1]) / x[0] - x_w, body_length * (1 - cos(x[0])) * sin(x[1]) / x[0] - y_w,
        (body_length * sin(x[0]) / x[0]) + x[2] - z_w)


def config_to_actuate(x, *config_state):
    radius_channel = 4.5
    thread = 5
    phi, theta, dis = config_state
    return tan(x[0] / x[1]) - phi, sqrt(x[0] * x[0] + x[1] * x[1]) / radius_channel - theta, x[2] - dis


def actuate_to_motor(x, *act_state):
    radius_motor = 3
    thread_linear = 5
    one_rev_encoder_reading = 16000
    upper_length, side_length, stepper_length = act_state
    return x[0] * radius_motor - upper_length, x[1] * radius_motor - side_length, x[
        2] * thread_linear / one_rev_encoder_reading - stepper_length


def calculate_motor_actuation(target_in_camera):
    target_in_camera = tuple(target_in_camera)
    config_state = fsolve(world_to_config, [0.2, 0.2, 70.0], args=target_in_camera)
    config_state = tuple(config_state)
    act_state = fsolve(config_to_actuate, [0.0, 0.0, 0.0], args=config_state)
    act_state = tuple(act_state)
    motor_revolution = fsolve(actuate_to_motor, [0.0, 0.0, 0.0], args=act_state)

    return motor_revolution
