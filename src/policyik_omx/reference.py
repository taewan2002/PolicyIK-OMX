"""Apply the OM-X joint-reference envelope with a caller-provided collision check."""
import numpy as np

from .braking import braked_gripper


class ReferenceEnvelope:
    """Stateful, rate-limited reference for policies with the same OMX geometry."""

    def __init__(self, position, low, high, clearance, *, gripper_braking=False):
        self.low = np.asarray(low,dtype=float)
        self.high = np.asarray(high,dtype=float)
        self.position = np.asarray(position,dtype=float).copy()
        self.velocity = np.zeros(6)
        self.clearance = clearance
        self.gripper_braking = gripper_braking
        self.speed = np.r_[np.repeat(.45,5),.8]
        self.acceleration = np.r_[np.repeat(1.5,5),2.]
        if self.low.shape!=(6,) or self.high.shape!=(6,) or not np.all(self.low<self.high):
            raise ValueError("Expected increasing six-channel calibrated limits")
        if not self.valid(self.position):
            raise ValueError("Initial measured position violates the calibrated envelope")

    def valid(self, position):
        return bool(position.shape==(6,) and np.isfinite(position).all()
                    and np.all((position>=self.low)&(position<=self.high))
                    and self.clearance(position)>=.010)

    def preview(self, target, measured, dt):
        if not .005<=dt<=.10 or not self.valid(target) or not self.valid(measured):
            return None
        if np.max(np.abs(self.position[:5]-measured[:5]))>.25:
            return None
        delta = target-self.position
        desired_velocity = delta/max(dt,float(np.max(np.abs(delta)/self.speed)))
        change = desired_velocity-self.velocity
        scale = min(1.,float(np.min(self.acceleration*dt/np.maximum(np.abs(change),1e-12))))
        displacement = (self.velocity+scale*change)*dt
        overshoot = (np.sign(displacement)==np.sign(delta)) & (np.abs(displacement)>np.abs(delta))
        displacement = np.where(overshoot,delta,displacement)
        if self.gripper_braking:
            try:
                displacement[5] = braked_gripper(self.position[5],self.velocity[5],target[5],
                                                self.low[5],self.high[5],dt,self.speed[5],self.acceleration[5])
            except ValueError:
                return None
        if any(not self.valid(self.position+fraction*displacement) for fraction in (.25,.50,.75,1.)):
            return None
        return self.position+displacement,displacement/dt

    def accept(self, target, measured, dt):
        candidate = self.preview(np.asarray(target),np.asarray(measured),dt)
        if candidate is None:
            raise ValueError("Reference rejected by the motion envelope")
        self.position,self.velocity = candidate
        return self.position.copy()
