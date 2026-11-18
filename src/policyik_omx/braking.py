"""Discrete braking calculations for OM-X position references."""

import numpy as np


def stopping_distance(speed, dt, accel):
    """Travel if this velocity is used now and braked at the cap on later ticks."""
    speed = np.asarray(speed)
    steps = np.maximum(1.,np.ceil(np.abs(speed)/(accel*dt)))
    return np.sign(speed)*dt*(steps*np.abs(speed)-accel*dt*steps*(steps-1.)*.5)


def max_stoppable_speed(distance, dt, speed, accel):
    distance = np.maximum(0.,np.asarray(distance))
    upper = np.full_like(distance,speed)
    if np.all(distance>=stopping_distance(speed,dt,accel)):
        return upper
    lower = np.zeros_like(upper)
    for _ in range(25):
        trial = (lower+upper)*.5
        allowed = stopping_distance(trial,dt,accel)<=distance
        lower,upper = np.where(allowed,trial,lower),np.where(allowed,upper,trial)
    return lower


def braked_gripper(position, velocity, goal, lower, upper, dt, speed, accel):
    """Choose the velocity closest to the goal while retaining space to stop."""
    limits = (
        np.maximum(-max_stoppable_speed(position-lower,dt,speed,accel),velocity-accel*dt),
        np.minimum(max_stoppable_speed(upper-position,dt,speed,accel),velocity+accel*dt),
    )
    # Numerical tolerance for a velocity already on the viability boundary.
    if np.any(limits[0]>limits[1]+1e-7):
        raise ValueError('Gripper braking margin exhausted')
    return dt*np.clip((goal-position)/dt,*limits)


def coordinated_displacement(targets, position, velocity, dt, speed, accel, lo, hi):
    delta = targets-position
    duration = np.maximum(dt,np.max(np.abs(delta)/speed,axis=-1))
    change = delta/duration[...,None]-velocity
    ratio = np.minimum(1.,np.min(accel*dt/np.maximum(np.abs(change),1e-12),axis=-1))
    displacement = (velocity+ratio[...,None]*change)*dt
    overshoot = (np.sign(displacement)==np.sign(delta))&(np.abs(displacement)>np.abs(delta))
    displacement = np.where(overshoot,delta,displacement)
    displacement[...,5] = braked_gripper(position[5],velocity[5],targets[...,5],lo[5],hi[5],dt,speed[5],accel[5])
    return displacement
