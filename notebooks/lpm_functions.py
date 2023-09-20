# Integración numérica de sistemas de ecuaciones diferenciales ordinarias
from scipy.integrate import solve_ivp

# Minimización de funciones objetivos para ajuste de parámetros
from scipy.optimize import least_squares

# Ecuaciones no lineales
from scipy.optimize import fsolve

# Computación científica
import numpy as np

# Gráficos
import matplotlib.pyplot as plt
# Copiar variables creando nuevos objetos
import copy

# Procesamiento de datos
# Importar pandas para cargar datos desde archivos .csv y otros
import pandas as pd
# CSV puro y duro
import csv

##########################################################################################################

def p_eqn(p, n_0, R, T, alpha, beta, b, p_0, M_H2, m_s, volumen, epsilon, m_t):
    ''' 
        Función objetivo que representa la masa total en función de la presión
        del sistema. Esto lo utilizamos ya que calculamos la presión de forma implícita
        en función de otras variables.
    '''
    # Calcular moles de hidrógeno adsorbido f(T,p)
    n_a = n_0 * np.exp(-((R*T)/(alpha+beta*T))**b * np.log(p_0/p)**b)
    # Actualizar masa adsorbida
    m_a = n_a*M_H2*m_s
    # Actualizar masa gas
    n_g = p*(volumen*epsilon)/(R*T)
    m_g = n_g * M_H2
    return m_t - (m_a + m_g)

##########################################################################################################

def adsorcion_hidrogeno(t, y, c_s, c_p, c_w, m_s, m_w, M_H2, R, alpha, beta, epsilon, volumen, area, p_0, n_0, b, m_dot, h, h_f, T_f, charge, DEBUG=False):
    # if charge:
        # print(f'Simulando proceso de carga\n--------------------------')
    
    # Desempacar variables:
    m_t = y[0]
    T = y[1]
    
    # Calcular Variables dependientes de las variables independientes
    p_old = 0.033*1e6 # Presión inicial (Pa) para test No. 13

    p = fsolve(p_eqn, p_old, args = (n_0, R, T, alpha, beta, b, p_0, M_H2, m_s, volumen, epsilon, m_t))

    # Calcular masa adsorbida
    n_a = n_0 * np.exp(-((R*T)/(alpha+beta*T))**b * np.log(p_0/p)**b)
    m_a = n_a*M_H2*m_s

    # Actualizar numero de moles en la fase gas
    m_g = m_t - m_a

    # Calor esostérico
    # dH = alpha * (np.log(p_0/p))**(1/b)
    dH = alpha * (np.log(n_0/n_a))**(1/b)
    
    # print(f'm_a: {m_a}\nm_g: {m_g}')

    # ECUACIONES DIFERENCIALES
    # Ecuación diferencial de masa total
    if charge:
        dm_dt = m_dot
    elif not charge:
        dm_dt = -m_dot
    else:
        dm_dt = 0
        
    # Ecuación diferencial de temperatura
    if charge:
        dT_dt = (m_dot*h + dm_dt*(dH/M_H2) - h_f*area*(T-T_f))/(m_s*c_s + m_a*c_p + m_g*c_p + m_w*c_w)
    elif not charge:
        dT_dt = (-m_dot*h + dm_dt*(dH/M_H2) - h_f*area*(T-T_f))/(m_s*c_s + m_a*c_p + m_g*c_p + m_w*c_w)
    
    # DEBUG prints
    if DEBUG is True:
        print("\ndT_dt debugging \n")
        print("dT_dt = %.3e K/s" % dT_dt)
        print("m_dot = %.3e kg/s" % m_dot)
        print("h = %.3e Jm^2-" % h)
        print("dH = %.3e J/kg " % dH)
        print("M_H2 = %.3e kg/mol" % h)
        print("h_f = %.3e Wm^-2K^-1" % h_f)
        print("area = %.3e m^2" % area)
        print("T = %.3e K" % T)
        print("T_f = %.3e K" % T_f)        
        print("dm_dt = %.3e kg/s" % dm_dt)
        # Calor isoestérico
        print("\nIsosteric heat debugging\n")
        print("alpha = %.3e" % alpha)
        print("b= %.3e" % b)
        print("p_0 = %.3e Pa" % p_0)
        print("p = %.3e Pa" % p)

    # Empacar el vector del lado derecho en un vector 2x1
    
    print("dmdt", dm_dt)
    print("dTdt", dT_dt)

    dy = np.array([dm_dt, dT_dt[0]])
    
    return dy

##########################################################################################################

def resolver_modelo(index, parametros_1, parametros_2):
    # Seleccionar los valores del test deseado
    p_i = parametros_1[0][index]
    T_i = parametros_1[1][index]
    T_f = parametros_1[2][index]
    h_f = parametros_1[3][index]
    m_dot = parametros_1[4][index]
    t_0 = parametros_1[5][index]
    t_f = parametros_1[6][index]
    t_d = parametros_1[7][index]
    h_i = parametros_1[8][index]
    h_o = parametros_1[9][index]

    # Parámetros necesarios
    ## Constantes
    R = parametros_2[1]         # (J mol-1 K-1)
    alpha = parametros_2[2]     # Factor entalpico (J mol-1)
    beta = parametros_2[3]      # Factor entropico (J mol-1 K-1)
    epsilon_b = parametros_2[4]
    b = parametros_2[5]
    M_H2 = parametros_2[13]     # Masa molar del hidrogeno (kg mol-1)

    ## Calores específicos
    c_p = parametros_2[0]       # Calor especifico del hidrogeno (J kg-1 K-1)
    c_s = parametros_2[14]      # Calor especifico del carbón activado (J kg-1 K-1)     # datos del carbono
    c_w = parametros_2[15]      # Calor especifico paredes de acero (J kg-1 K-1)        # este es a 20°C segun internet

    ## Dimensiones estanque
    V = parametros_2[8]         # Volumen de estanque (m^3)
    l = parametros_2[9]         # Altura del estanque (m)
    A_e = parametros_2[10]      # Área superficial estanque (m^2)

    ## Masas
    m_s = parametros_2[11]      # Masa carbón activado (kg)
    m_w = parametros_2[12]      # Masa paredes de acero (kg)

    ## Otros
    p_0 = parametros_2[6]       # Presion de saturacion (Pa)
    n_0 = parametros_2[7]       # Cantidad limite de adsorcion (mol kg-1)
    

    n_a0 = n_0 * np.exp(- (R*T_i/(alpha + beta*T_i))**b * np.log(p_0/p_i)**b )

    n_gi = p_i*(V*epsilon_b)/(R*T_i)
    m_gi = n_gi * M_H2
    
    # Masa inicial total   (Lo agregué a la mala, no se si es coherente con el resto del código)
    m_ti = (n_a0+n_gi)*M_H2

    # PARAMETROS EN CARGA/DESCARGA
    # Carga o descarga?
    charge = True

    t_range = np.linspace(t_0, t_f, 100000)

    # Tupla de parametros que se deben pasar a solve_ivp
    DEBUG = False
    args = (c_s, c_p, c_w, m_s, m_w, M_H2, R, alpha, beta, epsilon_b, V, A_e, p_0, n_0, b, m_dot, h_i, h_f, T_f, charge, DEBUG)

    # En ingeniería química, sobre todo cuando hay reacciones o cambio de fases, se generan sistemas ultraestables
    # (ultrastiff) no conviene utilizar métodos explícitos. 
    sol = solve_ivp(adsorcion_hidrogeno, (t_0, t_f), [m_ti, T_i], args=args, t_eval= t_range, method = 'BDF')
    
    # Desempacar la sol.clear
    m_t = sol.y[0]
    T = sol.y[1]
    t = sol.t
    
    return t, m_t, T, p_i   # Agregué p_i para poder usarlo más tarde en el cálculo de la presión