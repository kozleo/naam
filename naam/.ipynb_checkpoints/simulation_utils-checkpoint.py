import numpy as np
import matplotlib.pyplot as plt
import torch


def make_matrix_positive_definite(B, scale=0.01):
    # # Make B positive definite by adding a scaled identity matrix
    e, _ = torch.linalg.eigh(B)
    min_eigenvalue = torch.min(e)
    I = torch.eye(B.size(0))
    if min_eigenvalue > 0:
        return B
    else:
        B += (
            scale - min_eigenvalue
        ) * I  # Ensure that the matrix is strictly positive definite

    return B


def phi(x, name="tanh", g=1.0):
    # returns neural activation function and integral of activation function
    if name == "tanh":
        return torch.tanh(g * x), (1 / g) * torch.log(torch.cosh(g * x))
    if name == "exp":
        return torch.exp(x), torch.exp(x) - 1
    if name == "linear":
        return x, 0.5 * (x**2)
    if name == "sigmoid":
        return torch.sigmoid(x), torch.log(torch.exp(x) + 1)


def g(P, name="linear"):
    if name == "linear":
        return P
    if name == "tanh":
        return torch.tanh(P)


def neuron_update(x, P, W, phi, g):
    # update for neural equations
    return -x + (W * g(P)) @ phi(x)[0]


def process_update(P, x, A, W, phi, g):
    # update for process equations
    return -torch.einsum("ijkl,kl->ij", A, P) + 0.5 * torch.diag(
        phi(x)[0]
    ) @ W @ torch.diag(phi(x)[0])


def network_update(x, P, W, A, phi, g):
    # overall netework update
    xtp1 = neuron_update(x, P, W, phi, g)
    Ptp1 = process_update(P, x, A, W, phi, g)
    return xtp1, Ptp1


def runge_kutta_4th_order(network_update, x, P, W, A, dt, tau, phi, g):
    # 1st step
    k1_x, k1_P = network_update(x, P, W, A, phi, g)
    x1 = x + dt * k1_x / (2 * tau)
    P1 = P + dt * k1_P / (2 * tau)

    # 2nd step
    k2_x, k2_P = network_update(x1, P1, W, A, phi, g)
    x2 = x + dt * k2_x / (2 * tau)
    P2 = P + dt * k2_P / (2 * tau)

    # 3rd step
    k3_x, k3_P = network_update(x2, P2, W, A, phi, g)
    x3 = x + dt * k3_x / tau
    P3 = P + dt * k3_P / tau

    # 4th step
    k4_x, k4_P = network_update(x3, P3, W, A, phi, g)

    # Update
    x_new = x + dt * (k1_x + 2 * k2_x + 2 * k3_x + k4_x) / (6 * tau)
    P_new = P + dt * (k1_P + 2 * k2_P + 2 * k3_P + k4_P) / (6 * tau)

    return x_new, P_new


def energy(x, P, A, W, phi, g):
    # overall energy function for the neuron-astrocyte network

    E_N = x @ torch.tanh(x) - torch.sum(phi(x)[1])  # neural energy
    E_P = 0.5 * torch.einsum("ijkl,ij,kl->", A, P, P)  # process energy
    E_NP = (
        -0.5 * phi(x)[0] @ (W * g(P)) @ phi(x)[0]
    )  # neuron-process interaction energy

    return E_N + E_P + E_NP

def generate_DAM_tensor(eta, n, mu):
    # Using einsum to perform the specified tensor multiplication and summation
    T = torch.einsum('mi,mj,mk,ml->ijkl', eta, eta, eta, eta)
    return T

def do_one_run(T,dt,x0,W,A,phi,g):
  
  #simulate params
  tau = 1
  times = np.arange(0,T,dt)
  n = x0.size(0)

  # define initial conditions
  x = x0
  P = (1/n)*torch.outer(x0,x0)
  P = 0.5*(P + P.T)

  xs = torch.zeros((len(times),n))
  Ps = torch.zeros((len(times),n,n))
  Es = torch.zeros((len(times)))

  # run network forward in time
  for i,t in enumerate(times):

    # update with Runge Kutta 4th order method
    x, P = runge_kutta_4th_order(network_update, x, P, W, A, dt, tau, phi,g)

    # store states
    xs[i] = x
    Ps[i] = P
    Es[i] = energy(x,P,A,W ,phi,g)

    # make sure that P remains symmetric
    assert torch.norm(P - P.T,p = 'fro') <=1e-4, "Process update not symmetric."

    # make sure energy is decreasing, while allowing for error in numerical integration
    if i > 1:
      E_dot = (Es[i] - Es[i-1])/dt
      if torch.abs(E_dot) >= dt:
        assert E_dot <= 0,"Invalid energy"

  return x, torch.tanh(x), xs


def check_if_close(h,mem,epsilon):
  distance_to_memory = (1/h.size(0))*torch.linalg.norm(h - mem)
  return distance_to_memory <= epsilon


def define_network_parameters(n,mu):

  W = torch.ones(n,n)
  eta = torch.sign(torch.randn(mu,n))
  eta = torch.unique(eta,dim = 0)
  T = generate_DAM_tensor(eta, n, mu)
  T = T/(mu*n**3)
  T = make_matrix_positive_definite(T.view(n**2,n**2),scale = 1)
  A = torch.linalg.inv(T)
  #T = T.view(n,n,n,n)

  A = A.view(n,n,n,n)

  return eta, W, A