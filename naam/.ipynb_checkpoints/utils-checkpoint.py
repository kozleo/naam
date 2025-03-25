import torch


def make_matrix_positive_definite(B, scale = 0.01):

  # # Make B positive definite by adding a scaled identity matrix
  e,_ = torch.linalg.eigh(B)
  min_eigenvalue = torch.min(e)
  I = torch.eye(B.size(0))

  if min_eigenvalue > 0:
      return B
  else:
      #print(f"Minimum eigenvalue is {min_eigenvalue}, scaling matrix..")
      B += (scale-min_eigenvalue)*I   # Ensure that the matrix is strictly positive definite
      e_new,_ = torch.linalg.eigh(B)
      min_eigenvalue_new = torch.min(e_new)
      #print(f"New minimum eigenvalue is {min_eigenvalue_new}.")

  return B

def make_psd_matrix_pd(B, nu = 0.01):
  
    # # Make B positive definite by adding a scaled identity matrix
    I = torch.eye(B.size(0))
  
    return B + nu*I


def phi(x, name = 'tanh',g = 5):
  # returns neural activation function and integral of activation function
  if name == 'tanh':
    return torch.tanh(g*x), (1/g)*torch.log(torch.cosh(g*x))
  if name == 'exp':
    return torch.exp(x), torch.exp(x) - 1
  if name == 'linear':
    return x, 0.5*(x**2)
  if name == 'sigmoid':
    return torch.sigmoid(x), torch.log(torch.exp(x) + 1)

def g(P, name = 'tanh',g = 5):
  if name == 'linear':
    return P
  if name == 'tanh':
    return torch.tanh(g*P)

def neuron_update(x,P,W,phi,g):
  # update for neural equations
  return -x + (W*g(P)) @ phi(x)[0]

def process_update(P,x,A, W ,phi,g):
  # update for process equations
  return -torch.einsum('ijkl,kl->ij', A, g(P)) + 0.5*torch.diag(phi(x)[0]) @ W @ torch.diag(phi(x)[0])

def network_update(x,P,W,A ,phi,g):
  # overall netework update
  xtp1 = neuron_update(x,P,W ,phi,g)
  Ptp1 = process_update(P,x,A,W ,phi,g)
  return xtp1 , Ptp1

def energy(x,P,A,W,phi,g):
  # overall energy function for the neuron-astrocyte network

  E_N = x @ phi(x)[0] - torch.sum(phi(x)[1]) # neural energy
  E_P = 0.5*torch.einsum('ijkl,ij,kl->', A, g(P), g(P)) # process energy
  E_NP = - 0.5*phi(x)[0] @ (W*g(P)) @ phi(x)[0] # neuron-process interaction energy

  return E_N + E_P + E_NP


def DAM_energy(x,phi,L_x,g,psi,eta,lam):
    n = eta.shape[1]
    E_NN = x @ phi
    E_NS = -0.5*phi @ (g) @ phi
    E_PS = -0.5*psi.flatten() @ g.flatten()
    E_PP = -(1/n**3)*0.25*(torch.einsum('i,j,k,l,ij,kl->', eta, eta, eta, eta, psi, psi) + lam*psi.flatten() @ psi.flatten())
    
    return E_NN + E_NS + E_PS + E_PP
    
def generate_DAM_tensor(eta, n, mu):
    # Using einsum to perform the specified tensor multiplication and summation
    T = torch.einsum('mi,mj,mk,ml->ijkl', eta, eta, eta, eta)
    return T