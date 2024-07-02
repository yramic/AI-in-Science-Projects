import torch
import torch.optim as optim
from torch.distributions import Normal
import torch.nn as nn
import numpy as np
import torch.nn.functional as F
from torch.optim import Adam
from gym.wrappers.monitoring.video_recorder import VideoRecorder
import warnings
from typing import Union
from utils import ReplayBuffer, get_env, run_episode

warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", category=UserWarning)

class NeuralNetwork(nn.Module):
    '''
    This class implements a neural network with a variable number of hidden layers and hidden units.
    You may use this function to parametrize your policy and critic networks.
    '''
    def __init__(self, input_dim: int, output_dim: int, hidden_size: int, 
                                hidden_layers: int, activation: str):
        super(NeuralNetwork, self).__init__()
        
        # input layer
        self.input_layer = nn.Linear(input_dim, hidden_size)

        # hidden layers
        self.hidden_layers = nn.ModuleList()
        for _ in range(hidden_layers):
            self.hidden_layers.append(nn.Linear(hidden_size, hidden_size))
        
        # create a separate mean and log std dev layers - use both for actor and 
        # only the mean layer for critic
        self.mean_layer = nn.Linear(hidden_size, output_dim)
        self.log_std_layer = nn.Linear(hidden_size, output_dim)
        
        self.activation = activation

    def forward(self, s: torch.Tensor) -> torch.Tensor:

        s = self.activation(self.input_layer(s))
        for hidden_layer in self.hidden_layers:
            s = self.activation(hidden_layer(s))
        mean = self.mean_layer(s)
        log_std = self.log_std_layer(s)
        return mean, log_std
    
class Actor:
    def __init__(self,hidden_size: int, hidden_layers: int, actor_lr: float,
                state_dim: int = 3, action_dim: int = 1, device: torch.device = torch.device('cpu')):
        super(Actor, self).__init__()

        self.hidden_size = hidden_size
        self.hidden_layers = hidden_layers
        self.actor_lr = actor_lr
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.device = device
        self.LOG_STD_MIN = -20
        self.LOG_STD_MAX = 2
        self.setup_actor()

    def setup_actor(self):
        '''
        This function sets up the actor network in the Actor class.
        '''

        # network
        self.nn = NeuralNetwork(
            self.state_dim,
            self.action_dim,
            self.hidden_size,
            self.hidden_layers,
            F.relu
        ).to(self.device)

        # optimizer
        self.optimizer = Adam(
            self.nn.parameters(), 
            lr=self.actor_lr
        )

    def clamp_log_std(self, log_std: torch.Tensor) -> torch.Tensor:
        '''
        :param log_std: torch.Tensor, log_std of the policy.
        Returns:
        :param log_std: torch.Tensor, log_std of the policy clamped between LOG_STD_MIN and LOG_STD_MAX.
        '''
        return torch.clamp(log_std, self.LOG_STD_MIN, self.LOG_STD_MAX)

    def get_action_and_log_prob(self, state: torch.Tensor, 
                                deterministic: bool) -> (torch.Tensor, torch.Tensor):
        '''
        :param state: torch.Tensor, state of the agent
        :param deterministic: boolean, if true return a deterministic action 
                                otherwise sample from the policy distribution.
        Returns:
        :param action: torch.Tensor, action the policy returns for the state.
        :param log_prob: log_probability of the the action.
        '''
        assert state.shape == (3,) or state.shape[1] == self.state_dim, 'State passed to this method has a wrong shape'
        action, log_prob = torch.zeros(state.shape[0]), torch.ones(state.shape[0])
        
        # function which returns an action and its log probability.
        # If working with stochastic policies, make sure that its log_std are clamped 
        # using the clamp_log_std function.
        
        # here i only implement stochastic lol 

        eps = 1e-6
        mean, log_std = self.nn(state)
        log_std = self.clamp_log_std(log_std)
        std = log_std.exp()
        
        dist = Normal(0, 1)
        epsilon = dist.sample().to(self.device)
        x = mean + epsilon * std # reparameterization trick

        action = torch.tanh(x).cpu() 
        log_prob = Normal(mean, std).log_prob(x) - torch.log(1 - action.pow(2) + eps)
        
        return action, log_prob

class Critic:
    def __init__(self, hidden_size: int, 
                 hidden_layers: int, critic_lr: int, state_dim: int = 3, 
                    action_dim: int = 1,device: torch.device = torch.device('cpu')):
        super(Critic, self).__init__()
        self.hidden_size = hidden_size
        self.hidden_layers = hidden_layers
        self.critic_lr = critic_lr
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.device = device
        self.setup_critic()

    def setup_critic(self):
        # set up the critic(s).
        # Note that you can have MULTIPLE critic networks in this class.
        
        # network
        self.nn = NeuralNetwork(
            self.state_dim + self.action_dim,
            1,
            self.hidden_size,
            self.hidden_layers,
            F.relu
        ).to(self.device)

        # optimizer
        self.optimizer = Adam(
            self.nn.parameters(), 
            lr=self.critic_lr
        )

    def forward(self, states, actions):
        x = torch.cat([states, actions], dim=1)
        q1, _ = self.nn(x)
        return q1

class TrainableParameter:
    '''
    This class could be used to define a trainable parameter in your method. You could find it 
    useful if you try to implement the entropy temerature parameter for SAC algorithm.
    '''
    def __init__(self, init_param: float, lr_param: float, 
                 train_param: bool, device: torch.device = torch.device('cpu')):
        
        self.log_param = torch.tensor(np.log(init_param), requires_grad=train_param, device=device)
        self.optimizer = optim.Adam([self.log_param], lr=lr_param)

    def get_param(self) -> torch.Tensor:
        return torch.exp(self.log_param)

    def get_log_param(self) -> torch.Tensor:
        return self.log_param

class Agent:
    def __init__(self):
        # Environment variables. You don't need to change this.
        self.state_dim = 3  # [cos(theta), sin(theta), theta_dot]
        self.action_dim = 1  # [torque] in[-1,1]
        self.batch_size = 200
        self.min_buffer_size = 1000
        self.max_buffer_size = 100000

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print("Using device: {}".format(self.device))
        self.memory = ReplayBuffer(self.min_buffer_size, self.max_buffer_size, self.device)
        
        self.setup_agent()

    def setup_agent(self):
        # Setup off-policy agent with policy and critic classes. 
        
        # various hyperparameters
        self.tau = 0.035
        self.gamma = 0.99
        self.lr = 3e-4
        self.hidden_size = 256
        self.hidden_layers = 3 # had a big impact 
        
        # to use trainable entropy temperature parameter
        self.tune_entropy = True
        self.target_entropy = -torch.prod(
            torch.Tensor(self.action_dim).to(self.device)
        ).item()
        self.alpha = TrainableParameter(1, self.lr, self.tune_entropy, self.device)
        
        # define actor
        self.actor = Actor(
            self.hidden_size,
            self.hidden_layers,
            self.lr,
            self.state_dim,
            self.action_dim
        )

        # define critic 1
        self.critic1 = Critic(
            self.hidden_size,
            self.hidden_layers,
            self.lr,
            self.state_dim,
            self.action_dim
        )

        # define critic 2
        self.critic2 = Critic(
            self.hidden_size,
            self.hidden_layers,
            self.lr,
            self.state_dim,
            self.action_dim
        )

        # define target for critic1 and initialize from critic1
        self.critic1_target = Critic(
            self.hidden_size,
            self.hidden_layers,
            self.lr,
            self.state_dim,
            self.action_dim
        )
        self.critic_target_update(self.critic1_target.nn, self.critic1.nn, self.tau, False)

        # define target for critic2 and initialize from critic2
        self.critic2_target = Critic(
            self.hidden_size,
            self.hidden_layers,
            self.lr,
            self.state_dim,
            self.action_dim
        )
        self.critic_target_update(self.critic2_target.nn, self.critic2.nn, self.tau, False)
        
    def get_action(self, s: np.ndarray, train: bool) -> np.ndarray:
        """
        :param s: np.ndarray, state of the pendulum. shape (3, )
        :param train: boolean to indicate if you are in eval or train mode. 
                    You can find it useful if you want to sample from deterministic policy.
        :return: np.ndarray,, action to apply on the environment, shape (1,)
        """
        s = torch.from_numpy(s).float().to(self.device)
        action, _ = self.actor.get_action_and_log_prob(s, False)
        action = action.detach().numpy()
        assert action.shape == (1,), 'Incorrect action shape.'
        assert isinstance(action, np.ndarray ), 'Action dtype must be np.ndarray' 
        return action

    @staticmethod
    def run_gradient_update_step(object: Union[Actor, Critic], loss: torch.Tensor):
        '''
        This function takes in a object containing trainable parameters and an optimizer, 
        and using a given loss, runs one step of gradient update. If you set up trainable parameters 
        and optimizer inside the object, you could find this function useful while training.
        :param object: object containing trainable parameters and an optimizer
        '''
        object.optimizer.zero_grad()
        loss.mean().backward()
        object.optimizer.step()

    def critic_target_update(self, base_net: NeuralNetwork, target_net: NeuralNetwork, 
                             tau: float, soft_update: bool):
        '''
        This method updates the target network parameters using the source network parameters.
        If soft_update is True, then perform a soft update, otherwise a hard update (copy).
        :param base_net: source network
        :param target_net: target network
        :param tau: soft update parameter
        :param soft_update: boolean to indicate whether to perform a soft update or not
        '''
        for param_target, param in zip(target_net.parameters(), base_net.parameters()):
            if soft_update:
                param_target.data.copy_(param_target.data * (1.0 - tau) + param.data * tau)
            else:
                param_target.data.copy_(param.data)

    def train_agent(self):
        '''
        This function represents one training iteration for the agent. It samples a batch 
        from the replay buffer,and then updates the policy and critic networks 
        using the sampled batch.
        '''
        # One step of training for the agent.
        # This was done by run_gradient_update_step for each policy and critic.
        # Example: self.run_gradient_update_step(self.policy, policy_loss)

        # Batch sampling
        batch = self.memory.sample(self.batch_size)
        s_batch, a_batch, r_batch, s_prime_batch = batch
        deterministic = True # doesn't really do much now

        # get next action predicted by actor
        next_action, next_log_pis = self.actor.get_action_and_log_prob(s_prime_batch, deterministic)

        # pget q values from 2 target models for the predicted action and find the least q value
        q_target1_next = self.critic1_target.forward(s_prime_batch.to(self.device), next_action.squeeze(0).to(self.device))
        q_target2_next = self.critic2_target.forward(s_prime_batch.to(self.device), next_action.squeeze(0).to(self.device))
        q_target_next = torch.min(q_target1_next, q_target2_next)

        # calculate target q
        q_targets = r_batch + (self.gamma * (q_target_next.cpu() - self.alpha.get_param() * next_log_pis.squeeze(0).cpu()))

        # Critic(s) update here.

        # compute loss for the two critic networks and perform gradient update 
        q1 = self.critic1.forward(s_batch, a_batch).cpu()
        q2 = self.critic2.forward(s_batch, a_batch).cpu()
        
        critic1_loss = 0.5 * F.mse_loss(q1, q_targets.detach())
        critic2_loss = 0.5 * F.mse_loss(q2, q_targets.detach())

        self.run_gradient_update_step(self.critic1, critic1_loss)
        self.run_gradient_update_step(self.critic2, critic2_loss)

        # Policy Update
        # compute actor loss for current state and do a gradient update
        actions, log_pis = self.actor.get_action_and_log_prob(s_batch, deterministic)
        q1 = self.critic1.forward(s_batch, actions).cpu()
        q2 = self.critic2.forward(s_batch, actions).cpu()
        min_q = torch.min(q1, q2)
        actor_loss = ((self.alpha.get_param() * log_pis) - min_q).mean() 
        self.run_gradient_update_step(self.actor, actor_loss)

        # tune alpha if required
        if self.tune_entropy:
            alpha_loss = -(self.alpha.get_log_param() * (log_pis + self.target_entropy).detach()).mean()
            self.run_gradient_update_step(self.alpha, alpha_loss)

        # soft update target networks
        self.critic_target_update(self.critic1.nn, self.critic1_target.nn, self.tau, True)
        self.critic_target_update(self.critic2.nn, self.critic2_target.nn, self.tau, True)

if __name__ == '__main__':

    TRAIN_EPISODES = 50
    TEST_EPISODES = 300

    # You may set the save_video param to output the video of one of the evalution episodes, or 
    # you can disable console printing during training and testing by setting verbose to False.
    save_video = False
    verbose = True

    agent = Agent()
    env = get_env(g=10.0, train=True)

    for EP in range(TRAIN_EPISODES):
        run_episode(env, agent, None, verbose, train=True)

    if verbose:
        print('\n')

    test_returns = []
    env = get_env(g=10.0, train=False)

    if save_video:
        video_rec = VideoRecorder(env, "pendulum_episode.mp4")
    
    for EP in range(TEST_EPISODES):
        rec = video_rec if (save_video and EP == TEST_EPISODES - 1) else None
        with torch.no_grad():
            episode_return = run_episode(env, agent, rec, verbose, train=False)
        test_returns.append(episode_return)

    avg_test_return = np.mean(np.array(test_returns))

    print("\n AVG_TEST_RETURN:{:.1f} \n".format(avg_test_return))

    if save_video:
        video_rec.close()
