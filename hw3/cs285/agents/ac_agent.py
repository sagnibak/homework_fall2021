from collections import OrderedDict
from os import rename

from cs285.critics.bootstrapped_continuous_critic import \
    BootstrappedContinuousCritic
from cs285.infrastructure.replay_buffer import ReplayBuffer
from cs285.infrastructure.utils import *
from cs285.policies.MLP_policy import MLPPolicyAC
import cs285.infrastructure.pytorch_util as ptu
from .base_agent import BaseAgent


class ACAgent(BaseAgent):
    def __init__(self, env, agent_params):
        super(ACAgent, self).__init__()

        self.env = env
        self.agent_params = agent_params

        self.gamma = self.agent_params['gamma']
        self.standardize_advantages = self.agent_params['standardize_advantages']

        self.actor = MLPPolicyAC(
            self.agent_params['ac_dim'],
            self.agent_params['ob_dim'],
            self.agent_params['n_layers'],
            self.agent_params['size'],
            self.agent_params['discrete'],
            self.agent_params['learning_rate'],
        )
        self.critic = BootstrappedContinuousCritic(self.agent_params)

        self.replay_buffer = ReplayBuffer()

    def train(self, ob_no, ac_na, re_n, next_ob_no, terminal_n):
        # DONE Implement the following pseudocode:
        # for agent_params['num_critic_updates_per_agent_update'] steps,
        #     update the critic

        # advantage = estimate_advantage(...)

        # for agent_params['num_actor_updates_per_agent_update'] steps,
        #     update the actor

        for i in range(self.agent_params["num_critic_updates_per_agent_update"]):
            # update the critic
            critic_loss = self.critic.update(
                ob_no, ac_na, next_ob_no, re_n, terminal_n
            )
        
        adv_n = self.estimate_advantage(
            ob_no=ob_no,
            next_ob_no=next_ob_no,
            re_n=re_n,
            terminal_n=terminal_n,
        )

        for i in range(self.agent_params["num_actor_updates_per_agent_update"]):
            # update the actor
            actor_loss = self.actor.update(
                observations=ob_no,
                actions=ac_na,
                adv_n=adv_n,
            )

        loss = OrderedDict()
        loss['Critic_Loss'] = critic_loss
        loss['Actor_Loss'] = actor_loss["Training Loss"]

        return loss

    def estimate_advantage(self, ob_no, next_ob_no, re_n, terminal_n):
        # DONE Implement the following pseudocode:
        # 1) query the critic with ob_no, to get V(s)
        # 2) query the critic with next_ob_no, to get V(s')
        # 3) estimate the Q value as Q(s, a) = r(s, a) + gamma*V(s')
        # HINT: Remember to cut off the V(s') term (ie set it to 0) at terminal states (ie terminal_n=1)
        # 4) calculate advantage (adv_n) as A(s, a) = Q(s, a) - V(s)
        ob_no = ptu.from_numpy(ob_no)
        next_ob_no = ptu.from_numpy(next_ob_no)
        re_n = ptu.from_numpy(re_n)
        terminal_n = ptu.from_numpy(terminal_n).bool()
        
        V_s = self.critic(ob_no)
        V_s_ = self.critic(next_ob_no)
        V_s_[terminal_n] = 0
        Q_sa = re_n + self.gamma * V_s_

        adv_n = (Q_sa - V_s).detach().cpu().numpy()

        if self.standardize_advantages:
            adv_n = (adv_n - np.mean(adv_n)) / (np.std(adv_n) + 1e-8)
        return adv_n

    def add_to_replay_buffer(self, paths):
        self.replay_buffer.add_rollouts(paths)

    def sample(self, batch_size):
        return self.replay_buffer.sample_recent_data(batch_size)
