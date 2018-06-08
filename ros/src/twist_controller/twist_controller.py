from pid import PID
from lowpass import LowPassFilter
from yaw_controller import YawController
import rospy

GAS_DENSITY = 2.858
ONE_MPH = 0.44704


class Controller(object):
    def __init__(self, vehicle_mass, fuel_capacity, brake_deadband, decel_limit, accel_limit, wheel_radius,
                 wheel_base, steer_ratio, max_lat_accel, max_steer_angle):
        # TODO: Implement
        self. yaw_controller = YawController(wheel_base, steer_ratio, 0.1, max_lat_accel, max_steer_angle)
        
        kp = 0.3
        ki = 0.1
        kd = 0.0
        
        minimum_throttle = 0.0 # Minimum Throttle Value
        maximum_throttle = 0.2 # Maximum Throttle Value
        
        self.throttle_controller = PID(kp, ki, kd, minimum_throttle, maximum_throttle)
        
        tau = 0.5 # 1 / (2 * pi * tau) = cutoff_frequency
        sample_time = 0.02 # Sample Time
        self.velocity_LPF = LowPassFilter(tau, sample_time)
        
        self.vehicle_mass = vehicle_mass
        self.fuel_capacity = fuel_capacity
        self.brake_deadband = brake_deadband
        self.decel_limit = decel_limit
        self.accel_limit = accel_limit
        self.wheel_radius = wheel_radius
        self.last_time = rospy.get_time()
        pass

    def control(self, current_velocity, dbw_enabled, linear_velocity, angular_velocity):
        # TODO: Change the arg, kwarg list to suit your needs
        # Return throttle, brake, steer
        if not dbw_enabled:
            self.throttle_controller.reset()
            return 0.0, 0.0, 0.0
            
        current_velocity = self.velocity_LPF.filt(current_velocity)
        
        steering = self.yaw_controller.get_steering(linear_velocity, angular_velocity, current_velocity)
        
        velocity_error = linear_velocity - current_velocity
        self.last_velocity = current_velocity
        
        current_time = rospy.get_time()
        sample_time = current_time - self.last_time
        self.last_time = current_time
        
        throttle = self.throttle_controller.step(velocity_error, sample_time)
        brake = 0
        
        if linear_velocity == 0.0 and current_velocity < 0.1:
            throttle = 0
            brake = 400 # Torque in Newton-meter. To hold the car stationary. Acceleration - 1m/s^2
        
        elif throttle < 0.1 and velocity_error < 0:
            throttle = 0
            deceleration = max(velocity_error, self.decel_limit)
            brake = abs(deceleration) * self.vehicle_mass * self.wheel_radius # Torque in Newton-meter
        
        return throttle, brake, steering
