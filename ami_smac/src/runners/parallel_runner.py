import random
from envs import REGISTRY as env_REGISTRY
from functools import partial
from components.episode_buffer import EpisodeBatch
from multiprocessing import Pipe, Process
import numpy as np
import torch as th


# Based (very) heavily on SubprocVecEnv from OpenAI Baselines
# https://github.com/openai/baselines/blob/master/baselines/common/vec_env/subproc_vec_env.py
class ParallelRunner:

    def __init__(self, args, logger):
        self.args = args
        self.logger = logger
        self.batch_size = self.args.batch_size_run

        # Make subprocesses for the envs
        self.parent_conns, self.worker_conns = zip(*[Pipe() for _ in range(self.batch_size)])
        env_fn = env_REGISTRY[self.args.env]
        env_args = [self.args.env_args.copy() for _ in range(self.batch_size)]
        for i in range(self.batch_size):
            env_args[i]["seed"] += i

        self.ps = [Process(target=env_worker, args=(worker_conn, CloudpickleWrapper(partial(env_fn, **env_arg))))
                            for env_arg, worker_conn in zip(env_args, self.worker_conns)]

        for p in self.ps:
            p.daemon = True
            p.start()

        self.parent_conns[0].send(("get_env_info", None))
        self.env_info = self.parent_conns[0].recv()
        self.episode_limit = self.env_info["episode_limit"]

        self.t = 0

        self.t_env = 0

        self.train_returns = []
        self.test_returns = []
        self.adv_train_returns = []
        self.adv_test_returns = []
        self.neg_train_returns = []
        self.neg_test_returns = []
        self.train_stats = {}
        self.test_stats = {}

        self.log_train_stats_t = -100000

    def setup(self, scheme, groups, preprocess, mac):
        self.new_batch = partial(EpisodeBatch, scheme, groups, self.batch_size, self.episode_limit + 1,
                                 preprocess=preprocess, device=self.args.device)
        self.mac = mac
        self.scheme = scheme
        self.groups = groups
        self.preprocess = preprocess

    def setup_adv(self, scheme, groups, preprocess, mac):
        self.new_adv_batch = partial(EpisodeBatch, scheme, groups, self.batch_size, self.episode_limit + 1,
                                 preprocess=preprocess, device=self.args.device)
        self.adv_mac = mac
        self.adv_scheme = scheme
        self.adv_groups = groups
        self.adv_preprocess = preprocess

    def get_env_info(self):
        return self.env_info

    def save_replay(self):
        self.parent_conns[0].send(("save_replay", None))

    def close_env(self):
        for parent_conn in self.parent_conns:
            parent_conn.send(("close", None))

    def generate_adv_agent_mask(self):
        rands = 0
        adv_agent_mask = [False for _ in range(self.args.n_agents)]
        for i in self.args.adv_agent_ids:
            if i < 0:
                rands += 1
            else:
                adv_agent_mask[i] = True
        rand_list = []
        for i in range(len(adv_agent_mask)):
            if adv_agent_mask[i] == 0:
                rand_list.append(i)
        ind = random.sample(rand_list, rands)
        for i in ind:
            adv_agent_mask[i] = True
        return adv_agent_mask
        

    def reset(self):
        self.batch = self.new_batch()
        if self.args.mode == "adv_policy" or self.args.mode == "def_policy" or self.args.mode == "adv_state":
            self.adv_batch = self.new_adv_batch()

        # Reset the envs
        for parent_conn in self.parent_conns:
            parent_conn.send(("reset", None))

        pre_transition_data = {
            "state": [],
            "avail_actions": [],
            "obs": []
        }

        adv_pre_transition_data = {
            "state": [],
            "avail_actions": [],
            "avail_actions_all": [],
            "obs": [],
            "obs_all": [],
            "adv_agent_mask": []
        }
        # Get the obs, state and avail_actions back
        for parent_conn in self.parent_conns:
            data = parent_conn.recv()
            pre_transition_data["state"].append(data["state"])
            pre_transition_data["avail_actions"].append(data["avail_actions"])
            pre_transition_data["obs"].append(data["obs"])
            if self.args.mode == "adv_policy" or self.args.mode == "def_policy" or self.args.mode == "adv_state":
                adv_agent_mask = self.generate_adv_agent_mask()
                adv_pre_transition_data["state"].append(data["state"])
                adv_pre_transition_data["avail_actions"].append(np.array(data["avail_actions"])[adv_agent_mask])
                adv_pre_transition_data["avail_actions_all"].append(data["avail_actions"])
                adv_pre_transition_data["obs"].append(np.array(data["obs"])[adv_agent_mask])
                adv_pre_transition_data["obs_all"].append(data["obs"])
                adv_pre_transition_data["adv_agent_mask"].append(adv_agent_mask)

        self.batch.update(pre_transition_data, ts=0)
        if self.args.mode == "adv_policy" or self.args.mode == "def_policy" or self.args.mode == "adv_state":
            self.adv_batch.update(adv_pre_transition_data, ts=0)
            self.batch.update({"adv_agent_mask": adv_pre_transition_data["adv_agent_mask"]}, ts=0)

        self.t = 0
        self.env_steps_this_run = 0

    def run(self, test_mode=False):
        self.reset()

        all_terminated = False
        episode_returns = [0 for _ in range(self.batch_size)]
        episode_lengths = [0 for _ in range(self.batch_size)]
        self.mac.init_hidden(batch_size=self.batch_size)
        terminated = [False for _ in range(self.batch_size)]
        envs_not_terminated = [b_idx for b_idx, termed in enumerate(terminated) if not termed]
        final_env_infos = []  # may store extra stats like battle won. this is filled in ORDER OF TERMINATION

        while True:

            # Pass the entire batch of experiences up till now to the agents
            # Receive the actions for each agent at this timestep in a batch for each un-terminated env
            actions = self.mac.select_actions(self.batch, t_ep=self.t, t_env=self.t_env, bs=envs_not_terminated, test_mode=test_mode)
            cpu_actions = actions.to("cpu").numpy()

            # Update the actions taken
            actions_chosen = {
                "actions": actions.unsqueeze(1)
            }
            self.batch.update(actions_chosen, bs=envs_not_terminated, ts=self.t, mark_filled=False)

            # Send actions to each env
            action_idx = 0
            for idx, parent_conn in enumerate(self.parent_conns):
                if idx in envs_not_terminated: # We produced actions for this env
                    if not terminated[idx]: # Only send the actions to the env if it hasn't terminated
                        parent_conn.send(("step", cpu_actions[action_idx]))
                    action_idx += 1 # actions is not a list over every env

            # Update envs_not_terminated
            envs_not_terminated = [b_idx for b_idx, termed in enumerate(terminated) if not termed]
            all_terminated = all(terminated)
            if all_terminated:
                break

            # Post step data we will insert for the current timestep
            post_transition_data = {
                "reward": [],
                "terminated": []
            }
            # Data for the next step we will insert in order to select an action
            pre_transition_data = {
                "state": [],
                "avail_actions": [],
                "obs": []
            }

            # Receive data back for each unterminated env
            for idx, parent_conn in enumerate(self.parent_conns):
                if not terminated[idx]:
                    data = parent_conn.recv()
                    # Remaining data for this current timestep
                    post_transition_data["reward"].append((data["reward"],))

                    episode_returns[idx] += data["reward"]
                    episode_lengths[idx] += 1
                    if not test_mode:
                        self.env_steps_this_run += 1

                    env_terminated = False
                    if data["terminated"]:
                        final_env_infos.append(data["info"])
                    if data["terminated"] and not data["info"].get("episode_limit", False):
                        env_terminated = True
                    terminated[idx] = data["terminated"]
                    post_transition_data["terminated"].append((env_terminated,))

                    # Data for the next timestep needed to select an action
                    pre_transition_data["state"].append(data["state"])
                    pre_transition_data["avail_actions"].append(data["avail_actions"])
                    pre_transition_data["obs"].append(data["obs"])

            # Add post_transiton data into the batch
            self.batch.update(post_transition_data, bs=envs_not_terminated, ts=self.t, mark_filled=False)

            # Move onto the next timestep
            self.t += 1

            # Add the pre-transition data
            self.batch.update(pre_transition_data, bs=envs_not_terminated, ts=self.t, mark_filled=True)

        if not test_mode:
            self.t_env += self.env_steps_this_run

        # Get stats back for each env
        for parent_conn in self.parent_conns:
            parent_conn.send(("get_stats",None))

        env_stats = []
        for parent_conn in self.parent_conns:
            env_stat = parent_conn.recv()
            env_stats.append(env_stat)

        cur_stats = self.test_stats if test_mode else self.train_stats
        cur_returns = self.test_returns if test_mode else self.train_returns
        log_prefix = "test/" if test_mode else "train/"
        infos = [cur_stats] + final_env_infos
        cur_stats.update({k: sum(d.get(k, 0) for d in infos) for k in set.union(*[set(d) for d in infos])})
        cur_stats["n_episodes"] = self.batch_size + cur_stats.get("n_episodes", 0)
        cur_stats["ep_length"] = sum(episode_lengths) + cur_stats.get("ep_length", 0)

        cur_returns.extend(episode_returns)

        n_test_runs = max(1, self.args.test_nepisode // self.batch_size) * self.batch_size
        if test_mode and (len(self.test_returns) == n_test_runs):
            self._log(cur_returns, cur_stats, log_prefix)
        elif self.t_env - self.log_train_stats_t >= self.args.runner_log_interval:
            self._log(cur_returns, cur_stats, log_prefix)
            if hasattr(self.mac.action_selector, "epsilon"):
                self.logger.log_stat("train/epsilon", self.mac.action_selector.epsilon, self.t_env)
            self.log_train_stats_t = self.t_env

        return self.batch

    def run_adv_policy(self, test_mode=False):
        if self.args.mode == "adv_state":
            return self.run_adv_state(test_mode)
        
        self.reset()
        all_terminated = False
        episode_returns = [0 for _ in range(self.batch_size)]
        episode_lengths = [0 for _ in range(self.batch_size)]
        adv_episode_returns = [0 for _ in range(self.batch_size)]
        adv_episode_lengths = [0 for _ in range(self.batch_size)]
        neg_episode_returns = [0 for _ in range(self.batch_size)]
        neg_episode_lengths = [0 for _ in range(self.batch_size)]
        
        self.mac.init_hidden(batch_size=self.batch_size)
        self.adv_mac.init_hidden(batch_size=self.batch_size)
        try:
            self.adv_mac.init_hidden_maa(batch_size=self.batch_size)
        except Exception:
            pass
        terminated = [False for _ in range(self.batch_size)]
        envs_not_terminated = [b_idx for b_idx, termed in enumerate(terminated) if not termed]
        final_env_infos = []  # may store extra stats like battle won. this is filled in ORDER OF TERMINATION

        while True:
            # Pass the entire batch of experiences up till now to the agents
            # Receive the actions for each agent at this timestep in a batch for each un-terminated env
            actions = self.mac.select_actions(self.batch, t_ep=self.t, t_env=self.t_env, bs=envs_not_terminated, test_mode=True)
            adv_actions = self.adv_mac.select_actions(self.adv_batch, t_ep=self.t, t_env=self.t_env, bs=envs_not_terminated, test_mode=test_mode)
            # actions shape: [batch_size, n_agents], adv_batch["adv_agent_mask"] shape: [batch_size, timesteps, n_agents, 1]
            adv_agent_mask = self.adv_batch["adv_agent_mask"][:, self.t, :, 0][envs_not_terminated]
            actions[adv_agent_mask] = adv_actions.flatten()

            cpu_actions = actions.to("cpu").numpy()

            # Update the actions taken. Batch saves real actions.
            actions_chosen = {
                "actions": actions.unsqueeze(1)
            }
            self.batch.update(actions_chosen, bs=envs_not_terminated, ts=self.t, mark_filled=False)

            # Update the adversarial actions
            adv_actions_chosen = {
                "actions": adv_actions.unsqueeze(1),
                "actions_all": actions.unsqueeze(1),
                #"benign_logit": benign_logit
            }
            self.adv_batch.update(adv_actions_chosen, bs=envs_not_terminated, ts=self.t, mark_filled=False)

            # Send actions to each env
            action_idx = 0
            for idx, parent_conn in enumerate(self.parent_conns):
                if idx in envs_not_terminated: # We produced actions for this env
                    if not terminated[idx]: # Only send the actions to the env if it hasn't terminated
                        parent_conn.send(("step", cpu_actions[action_idx]))
                    action_idx += 1 # actions is not a list over every env

            # Update envs_not_terminated
            envs_not_terminated = [b_idx for b_idx, termed in enumerate(terminated) if not termed]
            all_terminated = all(terminated)
            if all_terminated:
                break

            # Post step data we will insert for the current timestep
            post_transition_data = {
                "reward": [],
                "terminated": []
            }

            adv_post_transition_data = {
                "reward": [],
                "reward_positive": [],
                "reward_negative": [],
                "terminated": []
            }

            # Data for the next step we will insert in order to select an action
            pre_transition_data = {
                "state": [],
                "avail_actions": [],
                "obs": [],
                "adv_agent_mask": []
            }
            adv_pre_transition_data = {
                "state": [],
                "avail_actions": [],
                "avail_actions_all": [],
                "obs": [],
                "obs_all": [],
                "adv_agent_mask": []
            }

            # Receive data back for each unterminated env
            for idx, parent_conn in enumerate(self.parent_conns):
                if not terminated[idx]:
                    data = parent_conn.recv()
                    # Remaining data for this current timestep
                    post_transition_data["reward"].append((data["info"]["reward_positive"],))
                    
                    adv_reward = - data["reward"]
                    adv_post_transition_data["reward"].append((adv_reward,))
                    adv_post_transition_data["reward_positive"].append((data["info"]["reward_positive"],))
                    adv_post_transition_data["reward_negative"].append((data["info"]["reward_negative"],))

                    episode_returns[idx] += data["info"]["reward_positive"]
                    episode_lengths[idx] += 1
                    adv_episode_returns[idx] += adv_reward
                    adv_episode_lengths[idx] += 1
                    neg_episode_returns[idx] += data["info"]["reward_negative"]
                    neg_episode_lengths[idx] += 1
                    if not test_mode:
                        self.env_steps_this_run += 1

                    env_terminated = False

                    if data["terminated"]:
                        final_env_infos.append(data["info"])
                    if data["terminated"] and not data["info"].get("episode_limit", False):
                        env_terminated = True
                    terminated[idx] = data["terminated"]
                    post_transition_data["terminated"].append((env_terminated,))
                    adv_post_transition_data["terminated"].append((env_terminated,))

                    # Data for the next timestep needed to select an action
                    adv_agent_mask = self.adv_batch["adv_agent_mask"][idx, self.t, :, 0].cpu().numpy()

                    pre_transition_data["state"].append(data["state"])
                    pre_transition_data["avail_actions"].append(data["avail_actions"])
                    pre_transition_data["obs"].append(data["obs"])
                    pre_transition_data["adv_agent_mask"].append(adv_agent_mask)
                    
                    adv_pre_transition_data["state"].append(data["state"])
                    adv_pre_transition_data["avail_actions"].append(np.array(data["avail_actions"])[adv_agent_mask])
                    adv_pre_transition_data["avail_actions_all"].append(data["avail_actions"])
                    adv_pre_transition_data["obs"].append(np.array(data["obs"])[adv_agent_mask])
                    adv_pre_transition_data["obs_all"].append(data["obs"])
                    adv_pre_transition_data["adv_agent_mask"].append(adv_agent_mask)

            # Add post_transiton data into the batch
            self.batch.update(post_transition_data, bs=envs_not_terminated, ts=self.t, mark_filled=False)
            self.adv_batch.update(adv_post_transition_data, bs=envs_not_terminated, ts=self.t, mark_filled=False)

            # Move onto the next timestep
            self.t += 1

            # Add the pre-transition data
            self.batch.update(pre_transition_data, bs=envs_not_terminated, ts=self.t, mark_filled=True)
            self.adv_batch.update(adv_pre_transition_data, bs=envs_not_terminated, ts=self.t, mark_filled=True)

        if not test_mode:
            self.t_env += self.env_steps_this_run

        # Get stats back for each env
        for parent_conn in self.parent_conns:
            parent_conn.send(("get_stats", None))

        env_stats = []
        for parent_conn in self.parent_conns:
            env_stat = parent_conn.recv()
            env_stats.append(env_stat)

        cur_stats = self.test_stats if test_mode else self.train_stats
        cur_returns = self.test_returns if test_mode else self.train_returns
        adv_cur_returns = self.adv_test_returns if test_mode else self.adv_train_returns
        neg_cur_returns = self.neg_test_returns if test_mode else self.neg_train_returns
        log_prefix = "test/" if test_mode else "train/"
        infos = [cur_stats] + final_env_infos
        cur_stats.update({k: sum(d.get(k, 0) for d in infos) for k in set.union(*[set(d) for d in infos])})
        cur_stats["n_episodes"] = self.batch_size + cur_stats.get("n_episodes", 0)
        cur_stats["ep_length"] = sum(episode_lengths) + cur_stats.get("ep_length", 0)

        cur_returns.extend(episode_returns)
        adv_cur_returns.extend(adv_episode_returns)
        neg_cur_returns.extend(neg_episode_returns)
       
        n_test_runs = max(1, self.args.test_nepisode // self.batch_size) * self.batch_size
        if test_mode and (len(self.test_returns) == n_test_runs):
            self._log(cur_returns, cur_stats, log_prefix)
            self.logger.log_stat(log_prefix + "adv_return_mean", np.mean(adv_cur_returns), self.t_env)
            self.logger.log_stat(log_prefix + "adv_return_std", np.std(adv_cur_returns), self.t_env)
            self.logger.log_stat(log_prefix + "neg_return_mean", np.mean(neg_cur_returns), self.t_env)
            self.logger.log_stat(log_prefix + "neg_return_std", np.std(neg_cur_returns), self.t_env)
            adv_cur_returns.clear()
        elif self.t_env - self.log_train_stats_t >= self.args.runner_log_interval:
            self._log(cur_returns, cur_stats, log_prefix)
            self.logger.log_stat(log_prefix + "adv_return_mean", np.mean(adv_cur_returns), self.t_env)
            self.logger.log_stat(log_prefix + "adv_return_std", np.std(adv_cur_returns), self.t_env)
            self.logger.log_stat(log_prefix + "neg_return_mean", np.mean(neg_cur_returns), self.t_env)
            self.logger.log_stat(log_prefix + "neg_return_std", np.std(neg_cur_returns), self.t_env)
            adv_cur_returns.clear()
            if hasattr(self.mac.action_selector, "epsilon"):
                self.logger.log_stat("train/epsilon", self.mac.action_selector.epsilon, self.t_env)
            self.log_train_stats_t = self.t_env
        return self.batch, self.adv_batch

    def run_adv_state(self, test_mode=False):
        self.reset()
        all_terminated = False
        episode_returns = [0 for _ in range(self.batch_size)]
        episode_lengths = [0 for _ in range(self.batch_size)]
        adv_episode_returns = [0 for _ in range(self.batch_size)]
        adv_episode_lengths = [0 for _ in range(self.batch_size)]
        self.mac.init_hidden(batch_size=self.batch_size)
        self.adv_mac.init_hidden(batch_size=self.batch_size)
        try:
            self.adv_mac.init_hidden_maa(batch_size=self.batch_size)
        except Exception:
            pass
        terminated = [False for _ in range(self.batch_size)]
        envs_not_terminated = [b_idx for b_idx, termed in enumerate(terminated) if not termed]
        final_env_infos = []  # may store extra stats like battle won. this is filled in ORDER OF TERMINATION

        while True:
            if test_mode:
                obs = self.adv_mac.fgsm(self.adv_batch, self.t, self.mac).unsqueeze(1)
                self.batch["obs"][:, self.t, self.args.adv_agent_ids] = obs
                self.adv_batch["obs"][:, self.t] = obs

            # Pass the entire batch of experiences up till now to the agents
            # Receive the actions for each agent at this timestep in a batch for each un-terminated env
            actions = self.mac.select_actions(self.batch, t_ep=self.t, t_env=self.t_env, bs=envs_not_terminated, test_mode=True)
            adv_actions = self.adv_mac.select_actions(self.adv_batch, t_ep=self.t, t_env=self.t_env, bs=envs_not_terminated, test_mode=test_mode)
            adv_agent_mask = self.adv_batch["adv_agent_mask"][:, self.t, :, 0][envs_not_terminated]
            if not test_mode:
                actions[adv_agent_mask] = adv_actions.flatten()

            cpu_actions = actions.to("cpu").numpy()

            # Update the actions taken. Batch saves real actions.
            actions_chosen = {
                "actions": actions.unsqueeze(1)
            }
            self.batch.update(actions_chosen, bs=envs_not_terminated, ts=self.t, mark_filled=False)

            # Update the adversarial actions
            adv_actions_chosen = {
                "actions": adv_actions.unsqueeze(1),
                "actions_all": actions.unsqueeze(1),
                #"benign_logit": benign_logit
            }
            self.adv_batch.update(adv_actions_chosen, bs=envs_not_terminated, ts=self.t, mark_filled=False)

            # Send actions to each env
            action_idx = 0
            for idx, parent_conn in enumerate(self.parent_conns):
                if idx in envs_not_terminated: # We produced actions for this env
                    if not terminated[idx]: # Only send the actions to the env if it hasn't terminated
                        parent_conn.send(("step", cpu_actions[action_idx]))
                    action_idx += 1 # actions is not a list over every env

            # Update envs_not_terminated
            envs_not_terminated = [b_idx for b_idx, termed in enumerate(terminated) if not termed]
            all_terminated = all(terminated)
            if all_terminated:
                break

            # Post step data we will insert for the current timestep
            post_transition_data = {
                "reward": [],
                "terminated": []
            }

            adv_post_transition_data = {
                "reward": [],
                "reward_positive": [],
                "terminated": []
            }

            # Data for the next step we will insert in order to select an action
            pre_transition_data = {
                "state": [],
                "avail_actions": [],
                "obs": [],
                "adv_agent_mask": []
            }
            adv_pre_transition_data = {
                "state": [],
                "avail_actions": [],
                "avail_actions_all": [],
                "obs": [],
                "obs_all": [],
                "adv_agent_mask": []
            }

            # Receive data back for each unterminated env
            for idx, parent_conn in enumerate(self.parent_conns):
                if not terminated[idx]:
                    data = parent_conn.recv()
                    # Remaining data for this current timestep
                    adv_reward = - data["reward"]
                    adv_post_transition_data["reward"].append((adv_reward,))

                    post_transition_data["reward"].append((data["info"]["reward_positive"],))
                    adv_post_transition_data["reward_positive"].append((data["info"]["reward_positive"],))

                    episode_returns[idx] += data["info"]["reward_positive"]
                    episode_lengths[idx] += 1
                    adv_episode_returns[idx] += adv_reward
                    adv_episode_lengths[idx] += 1
                    if not test_mode:
                        self.env_steps_this_run += 1

                    env_terminated = False

                    if data["terminated"]:
                        final_env_infos.append(data["info"])
                    if data["terminated"] and not data["info"].get("episode_limit", False):
                        env_terminated = True
                    terminated[idx] = data["terminated"]
                    post_transition_data["terminated"].append((env_terminated,))
                    adv_post_transition_data["terminated"].append((env_terminated,))

                    # Data for the next timestep needed to select an action
                    adv_agent_mask = self.adv_batch["adv_agent_mask"][idx, self.t, :, 0].cpu().numpy()

                    pre_transition_data["state"].append(data["state"])
                    pre_transition_data["avail_actions"].append(data["avail_actions"])
                    pre_transition_data["obs"].append(data["obs"])
                    pre_transition_data["adv_agent_mask"].append(adv_agent_mask)
                    
                    adv_pre_transition_data["state"].append(data["state"])
                    adv_pre_transition_data["avail_actions"].append(np.array(data["avail_actions"])[adv_agent_mask])
                    adv_pre_transition_data["avail_actions_all"].append(data["avail_actions"])
                    adv_pre_transition_data["obs"].append(np.array(data["obs"])[adv_agent_mask])
                    adv_pre_transition_data["obs_all"].append(data["obs"])
                    adv_pre_transition_data["adv_agent_mask"].append(adv_agent_mask)

            # Add post_transiton data into the batch
            self.batch.update(post_transition_data, bs=envs_not_terminated, ts=self.t, mark_filled=False)
            self.adv_batch.update(adv_post_transition_data, bs=envs_not_terminated, ts=self.t, mark_filled=False)

            # Move onto the next timestep
            self.t += 1

            # Add the pre-transition data
            self.batch.update(pre_transition_data, bs=envs_not_terminated, ts=self.t, mark_filled=True)
            self.adv_batch.update(adv_pre_transition_data, bs=envs_not_terminated, ts=self.t, mark_filled=True)

        if not test_mode:
            self.t_env += self.env_steps_this_run

        # Get stats back for each env
        for parent_conn in self.parent_conns:
            parent_conn.send(("get_stats", None))

        env_stats = []
        for parent_conn in self.parent_conns:
            env_stat = parent_conn.recv()
            env_stats.append(env_stat)

        cur_stats = self.test_stats if test_mode else self.train_stats
        cur_returns = self.test_returns if test_mode else self.train_returns
        adv_cur_returns = self.adv_test_returns if test_mode else self.adv_train_returns
        log_prefix = "test/" if test_mode else "train/"
        infos = [cur_stats] + final_env_infos
        cur_stats.update({k: sum(d.get(k, 0) for d in infos) for k in set.union(*[set(d) for d in infos])})
        cur_stats["n_episodes"] = self.batch_size + cur_stats.get("n_episodes", 0)
        cur_stats["ep_length"] = sum(episode_lengths) + cur_stats.get("ep_length", 0)

        cur_returns.extend(episode_returns)
        adv_cur_returns.extend(adv_episode_returns)
       
        n_test_runs = max(1, self.args.test_nepisode // self.batch_size) * self.batch_size
        if test_mode and (len(self.test_returns) == n_test_runs):
            self._log(cur_returns, cur_stats, log_prefix)
            self.logger.log_stat(log_prefix + "adv_return_mean", np.mean(adv_cur_returns), self.t_env)
            self.logger.log_stat(log_prefix + "adv_return_std", np.std(adv_cur_returns), self.t_env)
            adv_cur_returns.clear()
        elif self.t_env - self.log_train_stats_t >= self.args.runner_log_interval:
            self._log(cur_returns, cur_stats, log_prefix)
            self.logger.log_stat(log_prefix + "adv_return_mean", np.mean(adv_cur_returns), self.t_env)
            self.logger.log_stat(log_prefix + "adv_return_std", np.std(adv_cur_returns), self.t_env)
            adv_cur_returns.clear()
            if hasattr(self.mac.action_selector, "epsilon"):
                self.logger.log_stat("train/epsilon", self.mac.action_selector.epsilon, self.t_env)
            self.log_train_stats_t = self.t_env
        return self.batch, self.adv_batch

    def _log(self, returns, stats, prefix):
        self.logger.log_stat(prefix + "return_mean", np.mean(returns), self.t_env)
        self.logger.log_stat(prefix + "return_std", np.std(returns), self.t_env)
        returns.clear()

        for k, v in stats.items():
            if k != "n_episodes":
                self.logger.log_stat(prefix + k + "_mean" , v/stats["n_episodes"], self.t_env)
        stats.clear()

def env_worker(remote, env_fn):
    # Make environment
    env = env_fn.x()
    while True:
        cmd, data = remote.recv()
        if cmd == "step":
            actions = data
            # Take a step in the environment
            reward, terminated, env_info = env.step(actions)
            if "reward_positive" not in env_info:
                env_info["reward_positive"] = reward
            if "reward_negative" not in env_info:
                env_info["reward_negative"] = - reward
            # Return the observations, avail_actions and state to make the next action
            state = env.get_state()
            avail_actions = env.get_avail_actions()
            obs = env.get_obs()
            remote.send({
                # Data for the next timestep needed to pick an action
                "state": state,
                "avail_actions": avail_actions,
                "obs": obs,
                # Rest of the data for the current timestep
                "reward": reward,
                "terminated": terminated,
                "info": env_info
            })
        elif cmd == "reset":
            env.reset()
            remote.send({
                "state": env.get_state(),
                "avail_actions": env.get_avail_actions(),
                "obs": env.get_obs()
            })
        elif cmd == "close":
            env.close()
            remote.close()
            break
        elif cmd == "get_env_info":
            remote.send(env.get_env_info())
        elif cmd == "get_stats":
            remote.send(env.get_stats())
        elif cmd == "save_replay":
            env.save_replay()
        else:
            raise NotImplementedError


class CloudpickleWrapper():
    """
    Uses cloudpickle to serialize contents (otherwise multiprocessing tries to use pickle)
    """
    def __init__(self, x):
        self.x = x
    def __getstate__(self):
        import cloudpickle
        return cloudpickle.dumps(self.x)
    def __setstate__(self, ob):
        import pickle
        self.x = pickle.loads(ob)

