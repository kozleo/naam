import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from tqdm import tqdm




# function to "corrupt" memory vectors, for use as initial condition to system
def generate_corruption_vector(length, corruption, device):
    """
    Generates a vector of 1s and -1s based on a specified corruption probability.

    Parameters:
    - length: The length of the output vector.
    - corruption: The probability of an element being -1 (p(-1)).

    Returns:
    - A torch.Tensor of shape (length,) with elements set to 1 with probability `1 - corruption` and to -1 with probability `corruption`.
    """
    # Generate a random vector with values between 0 and 1
    random_vector = torch.rand(length,device = device)

    # Elements <= 1 - corruption are set to 1, others are set to -1
    # This inverts the condition to meet the specified probabilities
    output_vector = torch.where(random_vector <= (1 - corruption), torch.tensor(1), torch.tensor(-1))

    return output_vector

def integrate_dynamics(N=500, K=10, NUM_STEPS=1000, dt=0.05, beta=5, corruption_level=0.00, 
                        etas=None, memory_to_seed_from=2, device='cpu'):
    """
    Simulates the Euler integration of a Hopfield-like network with optional customization.
    
    Parameters:
    - N: Number of neurons
    - K: Number of memories
    - NUM_STEPS: Number of Euler steps for integration
    - dt: Euler step size
    - beta: Gain of nonlinearity
    - corruption_level: Controls the degree of corruption of the initial state
    - etas: Predefined memory matrix (optional)
    - memory_to_seed_from: Index of the memory to initialize dynamics near
    - device: Computation device ('cpu' or 'cuda')
    """
    
    # Generate random binary memories if not provided
    if etas is None:
        etas = 2 * torch.randint(2, (N, K), device=device).float() - 1
    
    # Select which memory to initialize dynamics near
    assert memory_to_seed_from <= etas.shape[-1], print("We do not  have that many memories!")
    mem = etas[:, memory_to_seed_from]
    
    alpha = 0

    # Initialize all dynamic variables near the desired fixed point
    x0 = mem * generate_corruption_vector(N, corruption_level, device)
    
    # Initial states calculations
    psi_0 = -torch.outer(torch.tanh(beta * x0), torch.tanh(beta * x0))
    P0 = (1 / beta) * torch.atanh(psi_0)
    S0 = (1 / beta) * torch.atanh(-(1 / N ** 3) * torch.einsum('im,jm,km,lm,kl->ij', etas, etas, etas, etas, psi_0))
    
    # Initializing variables for dynamics
    x, S, P = x0.clone(), S0.clone(), P0.clone()
    
    # Euler integration of dynamics
    for _ in tqdm(range(NUM_STEPS)):
        h = torch.tanh(beta * x)
        g = torch.tanh(beta * S)
        psi = torch.tanh(beta * P)

        x += dt * (-alpha*x + g @ h)
        S += dt * (-alpha*S + torch.outer(h, h) + psi)
        P += dt * (-alpha*P + (1 / N ** 3) * torch.einsum('im,jm,km,lm,kl->ij', etas, etas, etas, etas, psi) + g)
    
    # Check whether dynamics converged to the correct attractor
    initial_distance = torch.norm(torch.sign(x0) - mem)
    final_distance = torch.norm(torch.sign(h) - mem)
    
    print(f'Initial L2 distance of neural state to memory {memory_to_seed_from} is {initial_distance}')
    print(f'Final L2 distance of neural state to memory {memory_to_seed_from} is {final_distance}')
    
    return x, S, P, initial_distance, final_distance,x0









# Custom round function
class RoundWithGradient(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x):
        return torch.sign(x)

    @staticmethod
    def backward(ctx, grad_output):
        return grad_output


def round_with_gradient(x):
    return RoundWithGradient.apply(x)

class Autoencoder(nn.Module):
    def __init__(self):
        super(Autoencoder, self).__init__()
        # Input size: [batch, 3, 32, 32]
        # Output size: [batch, 3, 32, 32]
        self.encoder = nn.Sequential(
            nn.Conv2d(3, 12, 4, stride=2, padding=1),            # [batch, 12, 16, 16]
            nn.GELU(),
            nn.Conv2d(12, 24, 4, stride=2, padding=1),           # [batch, 24, 8, 8]
            nn.GELU(),
            nn.Conv2d(24, 48, 4, stride=2, padding=1),           # [batch, 48, 4, 4]
            nn.GELU(),
            nn.Flatten(),                                        # Flatten the output for the linear layer
            nn.Linear(48*4*4, 48*4*4),                             # Linear layer to get to the latent space
            nn.Tanh()                                            # Tanh activation to scale values to [-1, 1]
        )
        self.decoder = nn.Sequential(
            nn.Linear(48*4*4, 48*4*4),                             # Linear layer to go back from the latent space
            nn.GELU(),
            nn.Unflatten(1, (48, 4, 4)),                         # Unflatten to get back to the conv shape
            nn.ConvTranspose2d(48, 24, 4, stride=2, padding=1),  # [batch, 24, 8, 8]
            nn.GELU(),
            nn.ConvTranspose2d(24, 12, 4, stride=2, padding=1),  # [batch, 12, 16, 16]
            nn.GELU(),
            nn.ConvTranspose2d(12, 3, 4, stride=2, padding=1),   # [batch, 3, 32, 32]
            nn.Sigmoid(),
        )

    def forward(self, x):
        encoded = self.encoder(x)
        #latent = round_with_gradient(encoded)
        decoded = self.decoder(encoded)
        return decoded




import torch.nn.utils.parametrize as parametrize
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


class ContinuousRNN(nn.Module):
    def __init__(self, input_size, dt,steps,device, clamp = False,store_intermediate = True):
        super(ContinuousRNN, self).__init__()
        
        self.input_size = input_size # size of the input (number of neurons)
        self.dt = dt # euler int step size
        self.steps = steps # number of steps to integrate for
        
        self.dev = device # device
        self.clamp = clamp
        self.store_intermediate = store_intermediate
        
        # if clamp, then free units (units to update) but be provided
        # if self.clamp == False:
        #     self.free_inds = torch.ones(self.input_size,device = self.dev)
        # else:
        #     assert free_inds is not None, "Tried to clamp, but no units to clamp provided"
        
        self.W = nn.Linear(self.input_size,self.input_size,device = self.dev) # synaptic weights, modulated by s
        self.T = nn.Linear(self.input_size,self.input_size,device = self.dev) # process weights
        
        self.w_syn_to_proc = nn.Parameter(self.kaiming_init(self.input_size, device=self.dev))
        self.w_proc_to_syn = nn.Parameter(self.kaiming_init(self.input_size, device=self.dev))

        
        self.output_layer = nn.Linear(self.input_size,self.input_size,device = self.dev) # output linear layer
        
        
    def forward(self, x0, s0 = None, p0 = None, free_inds = None):
        
        if s0 == None:
            s0 = torch.zeros(x0.shape,device = self.dev) # initialize at zero
        if p0 == None:
            p0 = torch.zeros(x0.shape,device = self.dev) # initialize at zero
            
        x,s,p = x0,x0,x0
        
        xs = []
        
        # x is the batch of initial states with shape [batch_size, input_size]
        for _ in range(self.steps):
            
            phi_t = torch.relu(x) # neural nonlinearity
            g_t = torch.relu(s) # synaptic nonlinearity
            psi_t = torch.relu(p) # astrocyte nonlinearity
            
            dx = -x + self.W(g_t*phi_t) # neural update
            ds = -s + self.w_proc_to_syn*psi_t + phi_t # synapse update
            dp = -p + self.T(psi_t) + self.w_syn_to_proc*g_t # astrocyte update
            
            # euler steps
            x = x + self.dt * dx * free_inds # do not update clamped units
            s = p + self.dt * ds
            p = p + self.dt * dp
            
            if self.store_intermediate:
                xs.append(x)
        
        if self.store_intermediate:
            xs = torch.stack(xs)
        
        
        # last layer readout
        #x = x + x0 #torch.sigmoid(x + x0)
        # can try adding x0 as a residual connection
        #x = x0 + free_inds * (self.output_layer(x) - x0)
        #x = x0 + free_inds * (torch.sigmoid(x) - x0)
        
        return x, xs
    

    @staticmethod
    def kaiming_init(size, gain=np.sqrt(2), device='cpu'):
        """
        Custom Kaiming (He) initialization for 1D weights.

        Parameters:
        size (int): Size of the tensor (number of input units).
        gain (float): Gain factor for the initialization.
        device (str): The device on which the tensor will be allocated.

        Returns:
        torch.Tensor: Initialized tensor.
        """
        std = gain * np.sqrt(2.0 / size)
        return torch.randn(size, device=device) * std 

# class ContinuousRNN(nn.Module):
#     def __init__(self, input_size, dt,steps,device):
#         super(ContinuousRNN, self).__init__()
#         self.input_size = input_size
#         self.dt = dt
#         self.steps = steps
        
#         self.dev = device
        
        
        
#         self.W = nn.Linear(self.input_size,self.input_size,device = self.dev) #nn.Parameter(torch.randn(input_size, input_size,device = self.dev))
#         self.T = nn.Linear(self.input_size,self.input_size,device = self.dev)
        
#         self.w_syn_to_proc = nn.Parameter(torch.randn(self.input_size,device = self.dev))
#         self.w_proc_to_syn = nn.Parameter(torch.randn(self.input_size,device = self.dev))
        
#         self.output_layer = nn.Linear(self.input_size,self.input_size,device = self.dev)
#         self.T_sparsity_mask = self.__create_sparse_mask__()
        

#     def forward(self, x, s = None, p = None):
        
#         if s == None:
#             s = torch.zeros(x.shape,device = self.dev)
#         if p == None:
#             p = torch.zeros(x.shape,device = self.dev)
        
#         with torch.no_grad():
#             self.T.weight.data *=  self.T_sparsity_mask
        
#         # x is the batch of initial states with shape [batch_size, input_size]
#         for _ in range(self.steps):
            
#             phi_t = torch.tanh(x)
#             g_t = torch.tanh(s)
#             psi_t = torch.tanh(p)
            
#             dx = -x + self.W(g_t*phi_t)
#             ds = -s + self.w_proc_to_syn*psi_t + phi_t
#             dp = -p + self.T(psi_t) + self.w_syn_to_proc*g_t
            

#             x = x + self.dt * dx
#             s = p + self.dt * ds
#             p = p + self.dt * dp
        
#         x = self.output_layer(x)
#         x = torch.sigmoid(x)
        
#         return x
    
#     def __create_sparse_mask__(self, sparsity = 0.8):
#         """
#         Creates a sparse binary mask for weights with the given sparsity.

#         Parameters:
#             weight_shape (tuple): Shape of the weight matrix.
#             sparsity (float): Fraction of weights to be set to zero. Must be between 0 and 1.

#         Returns:
#             torch.Tensor: Binary mask with the same shape as weight_shape.
#         """
#         if not 0 <= sparsity <= 1:
#             raise ValueError("Sparsity must be between 0 and 1.")

#         weight_shape = self.T.weight.data.shape

#         # Total number of elements in the weight matrix
#         num_elements = weight_shape[0] * weight_shape[1]

#         # Number of elements to set to zero based on the sparsity
#         num_zeros = int(sparsity * num_elements)

#         # Create a flat mask with zeros and ones
#         mask_flat = torch.cat([torch.zeros(num_zeros,device = self.dev), torch.ones(num_elements - num_zeros,device = self.dev)])

#         # Shuffle the mask to distribute zeros randomly
#         mask_flat = mask_flat[torch.randperm(num_elements)]

#         # Reshape the mask to the original weight shape
#         mask = mask_flat.view(weight_shape)

#         return mask

