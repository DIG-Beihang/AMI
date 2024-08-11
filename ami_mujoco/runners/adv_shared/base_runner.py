import os
import numpy as np
import torch
from tensorboardX import SummaryWriter

from utils.shared_buffer import SharedReplayBuffer
from utils.shared_buffer_usenix import SharedReplayBufferUSENIX
from utils.shared_buffer_icml import SharedReplayBufferICML
from utils.shared_buffer_ami import SharedReplayBuffer_ami
from algorithms.fgsm import FGSM

def _t2n(x):
    """Convert torch tensor to a numpy array."""
    return x.detach().cpu().numpy()

class Runner(object):
    """
    Base class for training adversarial policies.
    :param config: (dict) Config dictionary containing parameters for training.
    """
    def __init__(self, config):
        self.config = config
        self.all_args = config['all_args']
        self.envs = config['envs']
        self.eval_envs = config['eval_envs']
        self.device = config['device']
        self.num_agents = config['num_agents']
        if config.__contains__("render_envs"):
            self.render_envs = config['render_envs']       

        # parameters
        self.env_name = self.all_args.env_name
        self.algorithm_name = self.all_args.algorithm_name
        self.experiment_name = self.all_args.experiment_name
        self.use_centralized_V = self.all_args.use_centralized_V
        self.use_obs_instead_of_state = self.all_args.use_obs_instead_of_state
        self.num_env_steps = self.all_args.num_env_steps
        self.episode_length = self.all_args.episode_length
        self.n_rollout_threads = self.all_args.n_rollout_threads
        self.n_eval_rollout_threads = self.all_args.n_eval_rollout_threads
        self.n_render_rollout_threads = self.all_args.n_render_rollout_threads
        self.use_linear_lr_decay = self.all_args.use_linear_lr_decay
        self.hidden_size = self.all_args.hidden_size
        self.use_render = self.all_args.use_render
        self.recurrent_N = self.all_args.recurrent_N


        # adversarial
        self.adv_agent_ids = self.all_args.adv_agent_ids
        self.num_adv_agents = len(self.adv_agent_ids)
        self.adv_agent_mask = [1 if i in self.adv_agent_ids else 0 for i in range(self.num_agents)]

        # interval
        self.save_interval = self.all_args.save_interval
        self.use_eval = self.all_args.use_eval
        self.eval_interval = self.all_args.eval_interval
        self.log_interval = self.all_args.log_interval

        # dir
        self.checkpoint_path = self.all_args.checkpoint_path
        self.adv_checkpoint_path = self.all_args.adv_checkpoint_path

        self.run_dir = config["run_dir"]
        self.log_dir = str(self.run_dir / 'logs')
        if not os.path.exists(self.log_dir):
            os.makedirs(self.log_dir)
        self.writter = SummaryWriter(self.log_dir)
        self.save_dir = str(self.run_dir / 'models')
        if not os.path.exists(self.save_dir):
            os.makedirs(self.save_dir)

        if self.all_args.algorithm_name == "mappo":
            from algorithms.mappo_policy import MAPPO_Policy as VictimPolicy

        if self.all_args.adv_algorithm_name == "mappo_iclr" or \
            self.all_args.adv_algorithm_name == "mappo_gma" or \
            self.all_args.adv_algorithm_name == "mappo_fgsm":
            from algorithms.mappo_trainer import MAPPO as TrainAlgo
            from algorithms.mappo_policy import MAPPO_Policy as Policy
        elif self.all_args.adv_algorithm_name == "mappo_usenix":
            from algorithms.usenix_trainer import USENIX as TrainAlgo
            from algorithms.usenix_policy import USENIX_Policy as Policy
        elif self.all_args.adv_algorithm_name == "mappo_icml":
            from algorithms.icml_trainer import ICML as TrainAlgo
            from algorithms.icml_policy import ICML_Policy as Policy
        elif self.all_args.adv_algorithm_name == "mappo_ami":
            from algorithms.ami_trainer import ami as TrainAlgo
            from algorithms.ami_policy import ami_Policy as Policy

        share_observation_space = self.envs.share_observation_space[0] if self.use_centralized_V else self.envs.observation_space[0]

        # policy network
        self.victim_policy = VictimPolicy(self.all_args,
                            self.envs.observation_space[0],
                            share_observation_space,
                            self.envs.action_space[0],
                            device = self.device)
        self.policy = Policy(self.all_args,
                            self.envs.observation_space[0],
                            share_observation_space,
                            self.envs.action_space[0],
                            device = self.device)

        if self.checkpoint_path is not None:
            print("Loading victim checkpoint from", self.checkpoint_path)
            policy_actor_state_dict = torch.load(str(self.checkpoint_path) + '/actor.pt')
            self.victim_policy.actor.load_state_dict(policy_actor_state_dict)
            if not self.all_args.use_render:
                policy_critic_state_dict = torch.load(str(self.checkpoint_path) + '/critic.pt')
                self.victim_policy.critic.load_state_dict(policy_critic_state_dict)

        if self.adv_checkpoint_path is not None:
            print("Loading adversarial checkpoint from", self.adv_checkpoint_path)
            policy_actor_state_dict = torch.load(str(self.adv_checkpoint_path) + '/actor.pt')
            self.policy.actor.load_state_dict(policy_actor_state_dict)
            if not self.all_args.use_render:
                policy_critic_state_dict = torch.load(str(self.adv_checkpoint_path) + '/critic.pt')
                self.policy.critic.load_state_dict(policy_critic_state_dict)
                if self.all_args.adv_algorithm_name == 'mappo_icml':
                    policy_critic_state_dict_icml = torch.load(str(self.adv_checkpoint_path) + '/critic_icml.pt')
                    self.policy.critic_icml.load_state_dict(policy_critic_state_dict_icml)

                if self.all_args.adv_algorithm_name == 'mappo_ami':
                    maa = torch.load(str(self.adv_checkpoint_path) + '/maa.pt')
                    self.policy.maa.load_state_dict(maa)

                    oracle = torch.load(str(self.adv_checkpoint_path) + '/oracle.pt')
                    self.policy.oracle.load_state_dict(oracle)

                    oracle_critic = torch.load(str(self.adv_checkpoint_path) + '/oracle_critic.pt')
                    self.policy.oracle_critic.load_state_dict(oracle_critic)

        # if self.all_args.adv_algorithm_name == "mappo_fgsm":
        #     self.attack = FGSM(self.all_args, self.victim_policy, self.policy, device = self.device)
        # if self.all_args.adv_algorithm_name == "mappo_iclr":
        #     self.attack = FGSM(self.all_args, self.victim_policy, self.trainer.policy, device = self.device)
        
        # algorithm
        self.trainer = TrainAlgo(self.all_args, self.policy, device = self.device)
        
        if self.all_args.adv_algorithm_name == "mappo_fgsm":
            self.attack = FGSM(self.all_args, self.victim_policy, self.policy, device = self.device)
        elif self.all_args.adv_algorithm_name == "mappo_gma":
            self.attack = FGSM(self.all_args, self.victim_policy, self.trainer.policy, device = self.device)
        
        # buffer
        self.victim_buffer = SharedReplayBuffer(self.all_args,
                                        self.num_agents,
                                        self.envs.observation_space[0],
                                        share_observation_space,
                                        self.envs.action_space[0])
        if self.all_args.adv_algorithm_name == 'mappo_iclr' or \
            self.all_args.adv_algorithm_name == 'mappo_fgsm':
            self.buffer = SharedReplayBuffer(self.all_args,
                                            self.num_adv_agents,
                                            self.envs.observation_space[0],
                                            share_observation_space,
                                            self.envs.action_space[0])
        elif self.all_args.adv_algorithm_name == 'mappo_usenix':
            self.buffer = SharedReplayBufferUSENIX(self.all_args,
                                        self.num_agents,
                                        self.envs.observation_space[0],
                                        share_observation_space,
                                        self.envs.action_space[0])
        elif self.all_args.adv_algorithm_name == 'mappo_icml':
            self.buffer = SharedReplayBufferICML(self.all_args,
                                        self.num_adv_agents,
                                        self.envs.observation_space[0],
                                        share_observation_space,
                                        self.envs.action_space[0])
        elif self.all_args.adv_algorithm_name == 'mappo_ami':
            self.buffer = SharedReplayBuffer_ami(self.all_args,
                                            self.num_agents,
                                            self.num_adv_agents,
                                            self.envs.observation_space[0],
                                            share_observation_space,
                                            self.envs.action_space[0])
            

    def run(self):
        """Collect training data, perform training updates, and evaluate policy."""
        raise NotImplementedError

    def warmup(self):
        """Collect warmup pre-training data."""
        raise NotImplementedError

    def collect(self, step):
        """Collect rollouts for training."""
        raise NotImplementedError

    def insert(self, data):
        """
        Insert data into buffer.
        :param data: (Tuple) data to insert into training buffer.
        """
        raise NotImplementedError
 
    @torch.no_grad()
    def compute(self):
        """Calculate returns for the collected data."""
        self.trainer.prep_rollout()
        if self.all_args.adv_algorithm_name == 'mappo_icml':
            next_values, next_values_icml = self.trainer.policy.get_values(np.concatenate(self.buffer.share_obs[-1]),
                                                    np.concatenate(self.buffer.rnn_states_critic[-1]),
                                                    np.concatenate(self.buffer.rnn_states_critic_icml[-1]),
                                                    np.concatenate(self.buffer.masks[-1]))
            next_values = np.array(np.split(_t2n(next_values), self.n_rollout_threads))
            next_values_icml = np.array(np.split(_t2n(next_values_icml), self.n_rollout_threads))
            self.buffer.compute_returns(next_values, next_values_icml, self.trainer.value_normalizer, self.trainer.value_normalizer_icml)
        elif self.all_args.adv_algorithm_name == 'mappo_ami':
            next_values = self.trainer.policy.get_values(np.concatenate(self.buffer.share_obs[-1]),
                                                    np.concatenate(self.buffer.rnn_states_critic[-1]),
                                                    np.concatenate(self.buffer.masks[-1]))
            oracle_next_values = self.trainer.policy.get_oracle_values(np.concatenate(self.buffer.share_obs[-1]),
                                                    np.concatenate(self.buffer.actions[-1]),
                                                    np.concatenate(self.buffer.rnn_states_critic_oracle[-1]),
                                                    np.concatenate(self.buffer.masks[-1]))
            next_values = np.array(np.split(_t2n(next_values), self.n_rollout_threads))
            oracle_next_values = np.array(np.split(_t2n(oracle_next_values), self.n_rollout_threads))
            self.buffer.compute_returns(next_values, self.trainer.value_normalizer)
            self.buffer.compute_oracle_returns(oracle_next_values, self.trainer.oracle_value_normalizer)
        else: 
            next_values = self.trainer.policy.get_values(np.concatenate(self.buffer.share_obs[-1]),
                                                    np.concatenate(self.buffer.rnn_states_critic[-1]),
                                                    np.concatenate(self.buffer.masks[-1]))
            next_values = np.array(np.split(_t2n(next_values), self.n_rollout_threads))
            self.buffer.compute_returns(next_values, self.trainer.value_normalizer)
    
    def train(self):
        """Train policies with data in buffer. """
        self.trainer.prep_training()
        train_infos = self.trainer.train(self.buffer)
        self.buffer.after_update()
        self.victim_buffer.after_update()
        return train_infos

    def save(self, total_num_steps):
        """Save policy's actor and critic networks."""
        save_dir = os.path.join(self.save_dir, str(total_num_steps))
        os.makedirs(save_dir, exist_ok=True)
        policy_actor = self.trainer.policy.actor
        torch.save(policy_actor.state_dict(), save_dir + "/actor.pt")
        policy_critic = self.trainer.policy.critic
        torch.save(policy_critic.state_dict(), save_dir + "/critic.pt")
        if self.all_args.adv_algorithm_name == 'mappo_icml':
            policy_critic_icml = self.trainer.policy.critic_icml
            torch.save(policy_critic_icml.state_dict(), save_dir + "/critic_icml.pt")
        if self.all_args.adv_algorithm_name == 'mappo_ami':
            maa = self.trainer.policy.maa
            torch.save(maa.state_dict(), save_dir + "/maa.pt")
            oracle = self.trainer.policy.oracle
            torch.save(oracle.state_dict(), save_dir + "/oracle.pt")
            oracle_critic = self.trainer.policy.oracle_critic
            torch.save(oracle_critic.state_dict(), save_dir + "/oracle_critic.pt")

    def restore(self):
        """Restore policy's networks from a saved model."""
        policy_actor_state_dict = torch.load(str(self.adv_checkpoint_path) + '/actor.pt')
        self.policy.actor.load_state_dict(policy_actor_state_dict)
        if not self.all_args.use_render:
            policy_critic_state_dict = torch.load(str(self.adv_checkpoint_path) + '/critic.pt')
            self.policy.critic.load_state_dict(policy_critic_state_dict)
            if self.all_args.adv_algorithm_name == 'mappo_icml':
                policy_critic_state_dict_icml = torch.load(str(self.adv_checkpoint_path) + '/critic_icml.pt')
                self.policy.critic_icml.load_state_dict(policy_critic_state_dict_icml)
 
    def log_train(self, train_infos, total_num_steps):
        """
        Log training info.
        :param train_infos: (dict) information about training update.
        :param total_num_steps: (int) total number of training env steps.
        """
        for k, v in train_infos.items():
            self.writter.add_scalar("train/" + k, v, total_num_steps)

    def log_env(self, env_infos, total_num_steps):
        """
        Log env info.
        :param env_infos: (dict) information about env state.
        :param total_num_steps: (int) total number of training env steps.
        """
        for k, v in env_infos.items():
            self.writter.add_scalar("test/" + k, np.mean(v), total_num_steps)
