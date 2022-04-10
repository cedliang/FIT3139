from statistics import mode
import numpy as np
from scipy.integrate import odeint
import matplotlib.pyplot as plt


# cedric liang 29674662


def model_sir(vars_dict, number_gens, initial_s, initial_i):
    beta = vars_dict["rate_of_transmission"]
    alpha = vars_dict["rate_of_recovery"]

    def sir_system(y, t, beta, alpha):
        y_s, y_i, y_r = y
        return [
            -beta*y_i * y_s,
            beta*y_i*y_s - alpha*y_i,
            alpha*y_i
        ]

    y0 = [initial_s, initial_i, 0]
    t = np.linspace(0, number_gens, 10*(number_gens+1))

    sol = odeint(sir_system, y0, t, args=(beta, alpha))

    return t, sol[:, 0], sol[:, 1], sol[:, 2]


def model_seir(vars_dict, number_gens, initial_s, initial_e):
    beta = vars_dict["rate_of_transmission"]
    eta = vars_dict["rate_of_becoming_infectious"]
    alpha = vars_dict["rate_of_recovery"]
    lamb = vars_dict["birth_rate"]
    mu = vars_dict["death_rate"]

    def seir_system(y, t, beta, eta, alpha, lamb, mu):
        y_s, y_e, y_i, y_r = y
        return [
            lamb*(y_s+y_e+y_i+y_r) - (beta*y_i + mu) * y_s,
            beta*y_i*y_s - (eta + mu)*y_e,
            eta*y_e - (alpha + mu)*y_i,
            alpha*y_i - y_r*mu
        ]

    y0 = [initial_s, initial_e, 0, 0]
    t = np.linspace(0, number_gens, 20*(number_gens+1))

    sol = odeint(seir_system, y0, t, args=(beta, eta, alpha, lamb, mu))

    return t, sol[:, 0], sol[:, 1], sol[:, 2], sol[:, 3]


def model_seiar(vars_dict, number_gens, initial_s, initial_e):
    beta = vars_dict["rate_of_transmission"]
    eta = vars_dict["rate_of_becoming_infectious"]
    alpha = vars_dict["rate_of_recovery"]
    lamb = vars_dict["birth_rate"]
    mu = vars_dict["death_rate"]

    p = vars_dict["prob_symptomatic"]
    q = vars_dict["rate_of_transmission_a"]
    gamma = vars_dict["rate_of_recovery_a"]

    def seiar_system(y, t, beta, eta, alpha, lamb, mu, p, q, gamma):
        y_s, y_e, y_i, y_a, y_r = y
        return [
            lamb*(y_s+y_e+y_i+y_a+y_r) - (beta*y_i + q*y_a + mu) * y_s,
            y_s*(beta*y_i + q*y_a) - (eta + mu)*y_e,
            p*eta*y_e - (alpha + mu)*y_i,
            (1-p)*eta*y_e - (gamma + mu)*y_a,
            alpha*y_i + gamma*y_a - y_r*mu
        ]

    y0 = [initial_s, initial_e, 0, 0, 0]
    t = np.linspace(0, number_gens, 20*(number_gens+1))

    sol = odeint(seiar_system, y0, t, args=(
        beta, eta, alpha, lamb, mu, p, q, gamma))

    return t, sol[:, 0], sol[:, 1], sol[:, 2], sol[:, 3], sol[:, 4]


if __name__ == "__main__":

    params = {
        "prob_symptomatic": 0.0,
        "rate_of_transmission": 0.005,
        "rate_of_transmission_a": 0.002,
        "rate_of_recovery": 0.025,
        "rate_of_recovery_a": 0.04,
        "rate_of_becoming_infectious": 0.02,
        "birth_rate": 0.0001,
        "death_rate": 0.00008
    }

    t, s_t, e_t, i_t, a_t, r_t = model_seiar(params, 1000, 100, 5)

    plt.plot(t, s_t, label='Susceptible population')
    plt.plot(t, e_t, label='Exposed population')
    plt.plot(t, i_t, label='Infectious population')
    plt.plot(t, a_t, label='Asymptomatic population')
    plt.plot(t, r_t, label='Recovered population')
    plt.legend(loc='best')
    plt.title(
        f'SEIAR model of disease spread in population with vital dynamics\nbeta={params["rate_of_transmission"]}, eta={params["rate_of_becoming_infectious"]}, alpha={params["rate_of_recovery"]}, lambda={params["birth_rate"]}, mu={params["death_rate"]}\np={params["prob_symptomatic"]}, q={params["rate_of_transmission_a"]}, gamma={params["rate_of_recovery_a"]}')
    plt.ylabel('Population count')
    plt.xlabel('Time (generations)')
    plt.grid()
    plt.show()

    # t, s_t, e_t, i_t, r_t = model_seir(params, 1000, 100, 5)

    # plt.plot(t, s_t, label='Susceptible population')
    # plt.plot(t, e_t, label='Exposed population')
    # plt.plot(t, i_t, label='Infectious population')
    # plt.plot(t, r_t, label='Recovered population')
    # plt.legend(loc='best')
    # plt.title(f'SEIR model of disease spread in population with vital dynamics\nbeta={params["rate_of_transmission"]}, eta={params["rate_of_becoming_infectious"]}, alpha={params["rate_of_recovery"]}, lambda={params["birth_rate"]}, mu={params["death_rate"]}')
    # plt.ylabel('Population count')
    # plt.xlabel('Time (generations)')
    # plt.grid()
    # plt.show()
