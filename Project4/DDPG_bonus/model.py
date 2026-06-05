import torch as torch 
import torch.nn as nn
import numpy as np 
import torch.nn.functional as F

def fanin_init(size, fanin=None):
    '''a helper function to initialize the weights of the model'''
    fanin = fanin or size[0]
    v = 1. / np.sqrt(fanin)
    return torch.Tensor(size).uniform_(-v, v)


class Actor(nn.Module):
    """Actor model for the DDPG algorithm.
    
    Layer 1: 400 units, ReLU activation, Fan-in weight initialization, ie each weight is initialized with a uniform distribution in the range of -1/sqrt(fan_in) to 1/sqrt(fan_in)
    Layer 2: 300 units, ReLU activation, Fan-in weight initialization, ie each weight is initialized with a uniform distribution in the range of -1/sqrt(fan_in) to 1/sqrt(fan_in)
    Layer 3: 1 unit, tanh activation, intialized with uniform weights in the range of -0.003 to 0.003
    
    """
    def __init__(self, input_size:tuple[int], action_size:int,CNN = None):
        """
        input: tuple[int]
            The input size, as a tuple of dimensions, for the DoubleInvertedPendulum environment, of shape (11,)
        action_size: int
            The number of actions
        """
        super(Actor, self).__init__()
        # ========== YOUR CODE HERE ==========
        # TODO:
        # define the fully connected layers for the actor
        # ====================================
        state_dim = input_size[0]
       
        # initialize layers
        self.fc1 = nn.Linear(state_dim, 400)
        self.fc2 = nn.Linear(400, 300)
        self.fc3 = nn.Linear(300, action_size)

        # initialize weights
        self.init_weights()
    
        # ========== YOUR CODE ENDS ==========
        
    def init_weights(self,init_w=3e-3):
        """
        Args:
            init_w (float, optional): the onesided range of the uniform distribution for the final layer. Defaults to 3e-3.
        """
        # ========== YOUR CODE HERE ==========
        # TODO:
        # initialize the weights of the model
        # ====================================

    def init_weights(self,init_w=3e-3):
        """
        Args:
            init_w (float, optional): the onesided range of the uniform distribution for the final layer. Defaults to 3e-3.
        """
        # ========== YOUR CODE HERE ==========
        # TODO:
        # initialize the weights of the model
        # ====================================
        # fc1
        self.fc1.weight.data = fanin_init(self.fc1.weight.data.size(), self.fc1.in_features) # weight
        self.fc1.bias.data.fill_(0.0)

        # fc2
        self.fc2.weight.data = fanin_init(self.fc2.weight.data.size(), self.fc2.in_features)
        self.fc2.bias.data.fill_(0.0)

        # fc3
        self.fc3.weight.data.uniform_(-init_w, init_w)
        self.fc3.bias.data.uniform_(-init_w, init_w)
        # ========== YOUR CODE ENDS ==========
    
    def forward(self, x:torch.Tensor)->torch.Tensor:
        # ========== YOUR CODE HERE ==========

        x = F.relu(self.fc1(x)) # layer 1
        x = F.relu(self.fc2(x)) # layer 2
        x = torch.tanh(self.fc3(x)) # layer 3
        return x

        # ========== YOUR CODE ENDS ==========
    


class Critic(nn.Module):
    """Critic model for the DDPG algorithm.
    Layer 1: 400 units, ReLU activation, Fan-in weight initialization, ie each weight is initialized with a uniform distribution in the range of -1/sqrt(fan_in) to 1/sqrt(fan_in)
    Layer 2: 300 units, ReLU activation, Fan-in weight initialization, ie each weight is initialized with a uniform distribution in the range of -1/sqrt(fan_in) to 1/sqrt(fan_in). Input is the concatenation of the 400 dimension embedding from the state, and the action taken.
    Layer 3: 1 unit, intialized with uniform weights in the range of -0.003 to 0.003
    """
    def __init__(self,input_size:tuple[int],action_size:int):
        """
        input: tuple[int]
            The input size, as a tuple of dimensions, for the DoubleInvertedPendulum environment, of shape (11,)
        action_size: int
            The number of actions
        """
        super(Critic, self).__init__()
        # ========== YOUR CODE HERE ==========
        # TODO: 
        # define the fully connected layers for the critic and initialize the weights
        # ====================================
        state_dim = input_size[0]
       
        # initialize layers
        self.fc1 = nn.Linear(state_dim, 400)
        self.fc2 = nn.Linear(400 +  action_size, 300)
        self.fc3 = nn.Linear(300, 1)

        # initialize weights
        self.init_weights()
    
        # ========== YOUR CODE ENDS ==========
        
    def init_weights(self,init_w=3e-3):
        # ========== YOUR CODE HERE ==========
        # TODO:
        # initialize the weights of the model
        # ====================================
        # fc1
        self.fc1.weight.data = fanin_init(self.fc1.weight.data.size(), self.fc1.in_features) # weight
        self.fc1.bias.data.fill_(0.0)
        # fc2
        self.fc2.weight.data = fanin_init(self.fc2.weight.data.size(), self.fc2.in_features)
        self.fc2.bias.data.fill_(0.0)
        # fc3
        self.fc3.weight.data.uniform_(-0.003, 0.003)
        self.fc3.bias.data.uniform_(-init_w, init_w)
        # ========== YOUR CODE ENDS ==========
        
    def forward(self, x:torch.Tensor, a:torch.Tensor)->torch.Tensor:
        # ========== YOUR CODE HERE ==========
        x = F.relu(self.fc1(x)) # layer 1
        x = torch.cat([x, a], dim=1) # concatenate state embedding + action
        x = F.relu(self.fc2(x)) # layer 2
        x = self.fc3(x) # layer 3
        return x
    
        # ========== YOUR CODE ENDS ==========
