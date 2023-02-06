"""
Method of wu2021adversarial (USENIX 2021).

Central loss term:
L_ad = maximize_theta(- ||ohat_v^(t+1) - o_v^(t+1)|| + ||ahat_v^(t+1) - a_v^(t+1)||)

State transition approximation model H (with parameter theta_H):
Input: o_v^t, a_v^t, a_a^t
Output: o_v^(t+1)
In multi-agent settings, the model predicts the observation of all victim agents.

Opponent policy network approximation model F (with parameter theta_F)
Input: o_v^t
Output: a_v^t
In multi-agent settings, the model predicts the action of all victim agents.
"""

from modules.agents import REGISTRY as agent_REGISTRY
from components.action_selectors import REGISTRY as action_REGISTRY
import torch as th


class UsenixMAC:
    # TODO: Do not support random adversarial agents
    def __init__(self, scheme, groups, args):
        self.n_agents = args.n_agents
        self.n_adv_agents = args.n_adv_agents
        self.args = args
        args.victim_agent_ids = []
        for i in range(args.n_agents):
            if i not in args.adv_agent_ids:
                args.victim_agent_ids.append(i)
        self.input_shape = self._get_input_shape(scheme)
        input_shape = self.input_shape
        self.obs_shape = scheme["obs"]["vshape"]
        self.state_shape = scheme["state"]["vshape"]
        state_input_shape = self._get_state_input_shape(scheme)

        self._build_agents(input_shape, state_input_shape)
        self.agent_output_type = args.agent_output_type

        self.action_selector = action_REGISTRY[args.action_selector](args)

        self.hidden_states = None
        self.state_hidden_states = None
        self.action_hidden_states = None

    def select_actions(self, ep_batch, t_ep, t_env, bs=slice(None), test_mode=False):
        # Only select actions for the selected batch elements in bs
        avail_actions = ep_batch["avail_actions"][:, t_ep]
        agent_outputs = self.forward(ep_batch, t_ep, test_mode=test_mode)
        chosen_actions = self.action_selector.select_action(agent_outputs[bs], avail_actions[bs], t_env, test_mode=test_mode)
        return chosen_actions

    def forward(self, ep_batch, t, test_mode=False):
        agent_inputs = self._build_inputs(ep_batch, t)
        avail_actions = ep_batch["avail_actions"][:, t]
        agent_outs, self.hidden_states = self.agent(agent_inputs, self.hidden_states)

        # Softmax the agent outputs if they're policy logits
        if self.agent_output_type == "pi_logits":

            if getattr(self.args, "mask_before_softmax", True):
                # Make the logits for unavailable actions very negative to minimise their affect on the softmax
                reshaped_avail_actions = avail_actions.reshape(ep_batch.batch_size * self.n_adv_agents, -1)
                agent_outs[reshaped_avail_actions == 0] = -1e10
            agent_outs = th.nn.functional.softmax(agent_outs, dim=-1)

        return agent_outs.view(ep_batch.batch_size, self.n_adv_agents, -1)

    def forward_no_softmax(self, ep_batch, t, test_mode=False):
        agent_inputs = self._build_inputs(ep_batch, t)
        avail_actions = ep_batch["avail_actions"][:, t]
        agent_outs, self.hidden_states = self.agent(agent_inputs, self.hidden_states)

        # Softmax the agent outputs if they're policy logits
        if self.agent_output_type == "pi_logits":

            if getattr(self.args, "mask_before_softmax", True):
                # Make the logits for unavailable actions very negative to minimise their affect on the softmax
                reshaped_avail_actions = avail_actions.reshape(ep_batch.batch_size * self.n_adv_agents, -1)
                agent_outs[reshaped_avail_actions == 0] = -1e10

        return agent_outs.view(ep_batch.batch_size, self.n_adv_agents, -1)

    def forward_state(self, ep_batch, t, test_mode=False):
        agent_inputs = self._build_state_inputs(ep_batch, t)
        agent_outs, self.state_hidden_states = self.state_model(agent_inputs, self.state_hidden_states)

        return agent_outs.view(ep_batch.batch_size, self.n_agents, -1)

    def forward_state_input(self, agent_inputs, ep_batch, t, test_mode=False):
        agent_outs, self.state_hidden_states = self.state_model(agent_inputs, self.state_hidden_states)

        return agent_outs.view(ep_batch.batch_size, self.n_agents, -1)

    def forward_action(self, ep_batch, t, test_mode=False):
        agent_inputs = self._build_action_inputs(ep_batch, t)
        return self.forward_action_input(agent_inputs, ep_batch, t, test_mode)

    def forward_action_input(self, agent_inputs, ep_batch, t, test_mode=False):
        avail_actions = ep_batch["avail_actions_all"][:, t, self.args.victim_agent_ids] # bav
        agent_outs, self.action_hidden_states = self.action_model(agent_inputs, self.action_hidden_states)

        if self.agent_output_type == "pi_logits":
            if getattr(self.args, "mask_before_softmax", True):
                # Make the logits for unavailable actions very negative to minimise their affect on the softmax
                reshaped_avail_actions = avail_actions.reshape(ep_batch.batch_size * (self.n_agents - self.n_adv_agents), -1)
                agent_outs[reshaped_avail_actions == 0] = -1e10
            agent_outs = th.nn.functional.softmax(agent_outs, dim=-1)

        return agent_outs.view(ep_batch.batch_size, (self.n_agents - self.n_adv_agents), -1)

    def forward_action_no_softmax(self, ep_batch, t, test_mode=False):
        agent_inputs = self._build_action_inputs(ep_batch, t)
        return self.forward_action_input(agent_inputs, ep_batch, t, test_mode)

    def forward_action_input_no_softmax(self, agent_inputs, ep_batch, t, test_mode=False):
        avail_actions = ep_batch["avail_actions_all"][:, t, self.args.victim_agent_ids] # bav
        agent_outs, self.action_hidden_states = self.action_model(agent_inputs, self.action_hidden_states)

        if self.agent_output_type == "pi_logits":
            if getattr(self.args, "mask_before_softmax", True):
                # Make the logits for unavailable actions very negative to minimise their affect on the softmax
                reshaped_avail_actions = avail_actions.reshape(ep_batch.batch_size * (self.n_agents - self.n_adv_agents), -1)
                agent_outs[reshaped_avail_actions == 0] = -1e10

        return agent_outs.view(ep_batch.batch_size, (self.n_agents - self.n_adv_agents), -1)

    def init_hidden(self, batch_size):
        self.hidden_states = self.agent.init_hidden().unsqueeze(0).expand(batch_size, self.n_adv_agents, -1)  # bav

    def init_hidden_state(self, batch_size):
        self.state_hidden_states = self.state_model.init_hidden().expand(batch_size, -1)  # bv

    def init_hidden_action(self, batch_size):
        self.action_hidden_states = self.action_model.init_hidden().expand(batch_size, -1)  # bv

    def parameters(self):
        return list(self.agent.parameters())

    def state_parameters(self):
        return list(self.state_model.parameters())

    def action_parameters(self):
        return list(self.action_model.parameters())

    def load_state(self, other_mac):
        self.agent.load_state_dict(other_mac.agent.state_dict())
        self.state_model.load_state_dict(other_mac.state_model.state_dict())
        self.action_model.load_state_dict(other_mac.action_model.state_dict())

    def cuda(self):
        self.agent.cuda()
        self.state_model.cuda()
        self.action_model.cuda()

    def save_models(self, path):
        th.save(self.agent.state_dict(), "{}/agent.th".format(path))
        th.save(self.state_model.state_dict(), "{}/state_model.th".format(path))
        th.save(self.action_model.state_dict(), "{}/action_model.th".format(path))

    def load_models(self, path):
        self.agent.load_state_dict(th.load("{}/agent.th".format(path), map_location=lambda storage, loc: storage))
        self.state_model.load_state_dict(th.load("{}/state_model.th".format(path), map_location=lambda storage, loc: storage))
        self.action_model.load_state_dict(th.load("{}/action_model.th".format(path), map_location=lambda storage, loc: storage))

    def _build_agents(self, input_shape, state_input_shape):
        self.agent = agent_REGISTRY[self.args.agent](input_shape, self.args)
        self.state_model = agent_REGISTRY["basic_rnn"](state_input_shape, self.obs_shape * self.args.n_agents, self.args.hidden_dim)
        self.action_model = agent_REGISTRY[self.args.agent](input_shape, self.args)

    def _build_inputs(self, batch, t):
        """
        Input for adv agent.
        Shape: [batch_size * n_adv_agents, input_shape]
        """
        bs = batch.batch_size
        inputs = []
        inputs.append(batch["obs"][:, t])  # b1av
        if self.args.obs_last_action:
            if t == 0:
                inputs.append(th.zeros_like(batch["actions_onehot"][:, t]))
            else:
                inputs.append(batch["actions_onehot"][:, t-1])
        if self.args.obs_agent_id:
            agent_index = th.eye(self.n_agents, device=batch.device).unsqueeze(0).expand(bs, -1, -1)
            # TODO: when an enviroment terminated, adv_agent_mask will not be updated, but mac continues to forward.
            adv_agent_mask = batch["adv_agent_mask"][:, 0].expand_as(agent_index)
            inputs.append(agent_index[adv_agent_mask])
        inputs = th.cat([x.reshape(bs*self.n_adv_agents, -1) for x in inputs], dim=1)
        return inputs

    def _build_inputs_with_obs(self, obs, batch, t):
        """
        Input for adv agent.
        Shape: [batch_size * n_adv_agents, input_shape]
        """
        bs = batch.batch_size
        inputs = []
        inputs.append(obs)  # b1av
        if self.args.obs_last_action:
            if t == 0:
                inputs.append(th.zeros_like(batch["actions_onehot"][:, t]))
            else:
                inputs.append(batch["actions_onehot"][:, t-1])
        if self.args.obs_agent_id:
            agent_index = th.eye(self.n_agents, device=batch.device).unsqueeze(0).expand(bs, -1, -1)
            inputs.append(agent_index)
        inputs = th.cat([x.reshape(bs, self.n_agents, -1) for x in inputs], dim=2)
        inputs = inputs[:, self.args.victim_agent_ids].reshape(bs * (self.n_agents - self.n_adv_agents), -1)
        return inputs

    def _build_state_inputs(self, batch, t):
        """
        Input for state model.
        Shape: [batch_size, n_agents * (n_actions + input_shape) + state_shape]
        """
        # concatenate all partial observations
        bs = batch.batch_size
        inputs = []
        inputs.append(batch["obs_all"][:, t])  # b1av
        if self.args.obs_last_action:
            if t == 0:
                inputs.append(th.zeros_like(batch["actions_onehot"][:, t]))
            else:
                inputs.append(batch["actions_onehot"][:, t-1])
        if self.args.obs_agent_id:
            agent_index = th.eye(self.n_agents, device=batch.device).unsqueeze(0).expand(bs, -1, -1)
            inputs.append(agent_index)

        # concatenate all actions in timestep t. This can only be executed in learning process.
        inputs.append(batch["actions_all_onehot"][:, t])
        inputs = th.cat([x.reshape(bs*self.n_agents, -1) for x in inputs], dim=1)

        # concatenate global state
        inputs = th.cat([inputs.reshape(bs, -1), batch["state"][:, t]], dim=1)

        return inputs

    def _build_action_inputs(self, batch, t):
        """
        Input for action model.
        Shape: [batch_size * n_agents, input_shape]
        """
        bs = batch.batch_size
        inputs = []
        inputs.append(batch["obs_all"][:, t])  # b1av
        if self.args.obs_last_action:
            if t == 0:
                inputs.append(th.zeros_like(batch["actions_onehot"][:, t]))
            else:
                inputs.append(batch["actions_onehot"][:, t-1])
        if self.args.obs_agent_id:
            agent_index = th.eye(self.n_agents, device=batch.device).unsqueeze(0).expand(bs, -1, -1)
            inputs.append(agent_index)
        inputs = th.cat([x.reshape(bs, self.n_agents, -1) for x in inputs], dim=2)

        # remove inputs of adv agent
        inputs = inputs[:, self.args.victim_agent_ids]
        inputs = inputs.reshape(bs * (self.n_agents - self.n_adv_agents), -1)

        return inputs


    def _get_input_shape(self, scheme):
        input_shape = scheme["obs"]["vshape"]
        if self.args.obs_last_action:
            input_shape += scheme["actions_onehot"]["vshape"][0]
        if self.args.obs_agent_id:
            input_shape += self.n_agents

        return input_shape

    def _get_state_input_shape(self, scheme):
        return (self._get_input_shape(scheme) + scheme["actions_onehot"]["vshape"][0]) * self.args.n_agents + scheme["state"]["vshape"]
