from typing import Callable
import numpy as np
from scipy.integrate import odeint
import matplotlib.pyplot as plt


def model_pharmacokinetics(vars_dict, sim_time, dosing):
    """
    Time unit:    Hour
    volume unit:  Litre
    dose unit:    mg

    Args:
        vars_dict : Dictionary of parameters
        sim_time : Tuple of simulation time. (total sim hours, sample interval)
        dosing : Dosage tuple. (Dose in mg, frequency of administration)

    Returns:
        tuple: timesteps, intestine drug mass (t), plasma drug mass (t), urine drug mass (t), plasma concentration(t)
    """
    t_half = vars_dict["half_life"]
    k_a = vars_dict["absorption_rate"]
    v_d = vars_dict["volume_of_distribution"]

    k_e = 0.693/t_half
    def c_p(p): return p/v_d

    dose, dose_interval = dosing

    def pulse(t):
        return dose if round(t) % dose_interval == 0.0 else 0

    def pharma_system(y, t, k_a, k_e):
        y_i, y_p, y_u = y
        return [pulse(t) - k_a * y_i, k_a * y_i - y_p * k_e, y_p * k_e]

    y0 = [0, 0, 0]
    t = np.linspace(0, sim_time[0], 1+round(sim_time[0]/sim_time[1]))

    sol = odeint(pharma_system, y0, t, args=(k_a, k_e))

    return t, sol[:, 0], sol[:, 1], sol[:, 2], np.array(list(map(c_p, sol[:, 1])))


if __name__ == "__main__":
    params_1 = {
        "half_life": 18,
        "absorption_rate": 1,
        "volume_of_distribution": 70
    }
    t, i_t, p_t, u_t, c_p_t = model_pharmacokinetics(
        params_1, (192, 0.25), (100, 24))

    params_2 = {
        "half_life": 18,
        "absorption_rate": 1,
        "volume_of_distribution": 70
    }
    t2, i_t2, p_t2, u_t2, c_p_t2 = model_pharmacokinetics(
        params_2, (192, 0.25), (50, 12))

    params_3 = {
        "half_life": 18,
        "absorption_rate": 1,
        "volume_of_distribution": 70
    }
    t3, i_t3, p_t3, u_t3, c_p_t3 = model_pharmacokinetics(
        params_3, (192, 0.25), (4.16666, 1))

    plt.plot(t, c_p_t, 'r', label='100mg per 24 hours')
    plt.plot(t2, c_p_t2, 'b', label='50mg per 12 hours')
    plt.plot(t3, c_p_t3, 'g', label='4.1666mg per 1 hour (iv drip)')
    plt.legend(loc='best')
    plt.title('Plasma concentration over time - equivalent dosage of 100mg per day')
    plt.ylabel('plasma concentration (mg/L)')
    plt.xlabel('t (h)')
    plt.grid()
    plt.show()




# Dosing more frequently ALWAYS leads to more consistent plasma concentration. A combination of the half life and the total volume of drugs being administered (ie, mg per day) is responsible for
# the concentration around which oscillations are seen - dosing more frequently will see the plasma concentration level stick more closely to this level while more infrequent administration will see us
# eventually settle into symmetrical oscillations above and below the 'average' steadt state concentration.

# assuming that we want to dose the drugs as infrequently as possible (for the convenience of the patient), we can set maximum and minimum levels, and try different values of dosing frequency (and the associated dosage) 
# to find the point at which the oscillations are within the acceptable range.