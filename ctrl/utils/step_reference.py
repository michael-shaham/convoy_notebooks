import numpy as np


def gen_step_ref(total_time: float, dt: float,
                 dist_start: float, dist_end: float,
                 v_start: float, v_end: float):
    """
    Generates a reference trajectory to track.

    arguments:
        total_time: total time for the reference trajectory (s)
        dt: discrete timestep (s)
        dist_start: start time of the disturbance (s)
        dist_end: end time of the disturbance (s)
        v_start: starting velocity for the vehicle (m/s)
        v_end: ending velocity for the vehicle (m/s)
    
    returns:
        x_ref (np.ndarray): Shape (3, T), rows are p, v, a
        t_range (np.ndarray): Shape (T,), corresponding times
    """
    t_range = np.arange(start=0.0, stop=total_time+dt/2, step=dt)
    T = len(t_range)
    mask = np.logical_and(t_range > dist_start+dt/2, t_range <= dist_end+dt/2)

    accel = (v_end - v_start) / (dist_end - dist_start)

    v_ref = np.zeros(T)
    v_ref[t_range <= dist_start+dt/2] = v_start
    v_ref[mask] = v_start + accel * (t_range[mask] - t_range[mask][0])
    v_ref[t_range > dist_end+dt/2] = v_end

    a_ref = np.zeros(T)
    a_ref[mask] = accel

    s_ref = np.zeros(T)
    s_ref[t_range <= dist_start+dt/2] = v_start * t_range[t_range <= dist_start+dt/2]
    s_ref[mask] = 0.5 * accel * (t_range[mask] - dist_start)**2 + \
        v_start * (t_range[mask] - dist_start) + \
        s_ref[t_range <= dist_start+dt/2][-1]
    s_ref[t_range > dist_end-dt/2] = v_end * (t_range[t_range > dist_end-dt/2] - dist_end) + s_ref[mask][-1]

    x_ref = np.r_[s_ref, v_ref, a_ref].reshape((3, -1))
    
    return x_ref, t_range


def accel_decel_step(total_time, dt, accel_start, accel_end, 
                     decel_start, decel_end, v_low, v_high):
        """
        Generates a reference trajectory that accelerates from v_low to v_high 
        then decelerates back to v_low.

        arguments:
            total_time: float, total time for the reference trajectory (s)
            dt: float, discrete timestep (s)
            accel_start: float,time when acceleration starts (s)
            accel_end: float, time when acceleration ends (s)
            decel_start: float, time when deceleration starts (s)
            decel_end: float, time when deceleration ends (s)
            v_low: float, starting velocity for the vehicle (m/s)
            v_high: float, velocity the vehicle keeps before decelerating (m/s)
        
        returns:
            np.ndarray, np.ndarray: shapes (n, T), (T,), reference states at 
                timesteps (starts at position 0), corresponding timesteps
        """
        assert accel_start < accel_end, "start must be before end"
        assert accel_end < decel_start, "must accelerate before slowing down"
        assert decel_start < decel_end, "start must be before end"
        assert decel_end < total_time, "end time must be less than total time"

        t_range = np.arange(start=0.0, stop=total_time+dt, step=dt)
        T = len(t_range)

        mask1 = np.logical_and(t_range >= accel_start, t_range <= accel_end)
        mask2 = np.logical_and(t_range >= accel_end, t_range <= decel_start)
        mask3 = np.logical_and(t_range >= decel_start, t_range <= decel_end)

        accel = (v_high - v_low) / (accel_end - accel_start)
        decel = (v_low - v_high) / (decel_end - decel_start)

        v_ref = np.zeros(T)
        v_ref[t_range <= accel_start] = v_low
        v_ref[mask1] = v_low + accel * (t_range[mask1] - t_range[mask1][0])
        v_ref[mask2] = v_high
        v_ref[mask3] = v_high + decel * (t_range[mask3] - t_range[mask3][0])
        v_ref[t_range > decel_end] = v_low 

        a_ref = np.zeros(T)
        a_ref[mask1] = accel
        a_ref[mask3] = decel 

        s_ref = np.zeros(T)
        s_ref[t_range <= accel_start] = v_low * t_range[t_range <= accel_start]
        s_ref[mask1] = 0.5 * (t_range[mask1] - accel_start)**2 + \
            v_low * (t_range[mask1] - accel_start) + \
            s_ref[t_range <= accel_start][-1]
        offset = s_ref[mask1][-1] - v_high * t_range[mask2][0]
        s_ref[mask2] = v_high * t_range[mask2] + offset
        s_ref[mask3] = -0.5 * (t_range[mask3] - decel_start)**2 + \
            v_high * (t_range[mask3] - decel_start) + \
            s_ref[t_range <= decel_start][-1]
        s_ref[t_range > decel_end] = \
            v_low * (t_range[t_range > decel_end] - decel_end) + s_ref[mask3][-1]

        x_ref = np.r_[s_ref, v_ref, a_ref].reshape((3, -1))
        return x_ref, t_range


def accel_step(v_des, accel_time, dt, total_time):
    """
    Generates a reference trajectory that accelerates from zero initial state 
    (zero pos, vel, accel) to the desired velocity in des_time time.

    arguments:
        v_des: float, desired velocity to reach from zero velocity
        accel_time: float, desired time it takes to reach v_des
        dt: float, timestep
        total_time: float, total time for trajectory
    
    returns:
        np.ndarray, np.ndarray: shapes (n, T), (T,), reference states at 
            timesteps (starts at origin state), corresponding timesteps
    """
    t_range = np.arange(start=0.0, stop=total_time+dt/2, step=dt)
    T = len(t_range)
    a_des = v_des / accel_time
    mask1 = t_range <= accel_time
    mask2 = t_range > accel_time

    v_ref = np.zeros(T)
    v_ref[mask1] = a_des * (t_range[mask1])
    v_ref[mask2] = v_des

    a_ref = np.zeros(T)
    a_ref[mask1] = a_des
    
    s_ref = np.zeros(T)
    s_ref[mask1] = 0.5 * a_des * t_range[mask1]**2
    s_ref[mask2] =  v_des * (t_range[mask2] - t_range[mask1][-1]) + s_ref[mask1][-1]
    x_ref = np.r_[s_ref, v_ref, a_ref].reshape((3, -1))

    return x_ref, t_range
