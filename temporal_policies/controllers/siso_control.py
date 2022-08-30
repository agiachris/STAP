import numpy as np


class SISOControl:
    def __init__(
        self,
        kp=1,
        ki=0,
        kd=0,
        scale=1,
        A=None,
        b=None,
        c=None,
        d=None,
        max_sum_e=10,
        overshoot_reset=False,
    ):
        """Single Input Single Output (SISO) Controller for Linear Time Invariant
        Model of the form xdot = Ax + bu, y = transpose(C)x + du

        args:
            kp: proportional gain
            ki: integral gain
            kd: derivative gain
            scale: controller output scale
            A: system matrix -- np.array (m, m)
            b: input matrix -- np.array (m,)
            c: output matrix -- np.array(m,)
            d: feedforward matrix -- np.array (1,)
            overshoot_reset: reset integral term upon overshoot
        """
        # Control law
        self._kp = kp
        self._ki = ki
        self._kd = kd
        self._scale = scale

        # LTI SISO (only required for simulation)
        self._A = A
        self._b = b
        self._c = c
        self._d = d

        # Control params
        self._ref = None
        self._prev_e = None
        self._sum_e = 0
        self._max_sum_e = max_sum_e
        self._overshoot_reset = overshoot_reset

    def can_simulate(self):
        assert self._A is not None
        assert self._b is not None
        assert self._c is not None
        assert self._d is not None

    def reset(self, ref, y, scale=1):
        self._ref = ref
        self._prev_e = ref - y
        self._sum_e = 0
        self._scale = scale

    def u(self, y):
        assert self._ref is not None
        assert self._prev_e is not None
        # Proportional
        e = self._ref - y
        up = self._kp * e
        # Integral
        overshot = np.sign(e) != np.sign(self._prev_e)
        if self._overshoot_reset and overshot:
            self._sum_e = 0
        self._sum_e += e
        self._sum_e = np.clip(self._sum_e, -self._max_sum_e, self._max_sum_e)
        ui = self._ki * self._sum_e
        # Derivative
        ud = self._kd * (e - self._prev_e)
        self._prev_e = e
        return self._scale * (up + ui + ud)
