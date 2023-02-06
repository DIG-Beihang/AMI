from modules.agents import REGISTRY as agent_REGISTRY
from components.action_selectors import REGISTRY as action_REGISTRY
from torch.distributions import Categorical
import torch as th


# This multi-agent controller contains a model of all agents
class STATEMAAMAC:
    def __init__(self, scheme, groups, args):
        self.n_agents = args.n_agents
        self.agents_num_for_model = 1
        self.n_adv_agents = args.n_adv_agents
        self.args = args
        input_shape = self._get_input_shape(scheme)
        self.state_shape = scheme["state"]["vshape"]
        maa_input_shape = self._get_maa_input_shape(scheme)
        oracle_input_shape = self._get_oracle_input_shape(scheme)
        self._build_agents(input_shape, maa_input_shape, oracle_input_shape)
        self.agent_output_type = args.agent_output_type

        self.action_selector = action_REGISTRY[args.action_selector](args)

        self.hidden_states = None

    def select_actions(self, ep_batch, t_ep, t_env, bs=slice(None), test_mode=False):
        # Only select actions for the selected batch elements in bs
        avail_actions = ep_batch["avail_actions"][:, t_ep]
        agent_outputs = self.forward(ep_batch, t_ep, test_mode=test_mode)
        chosen_actions = self.action_selector.select_action(agent_outputs[bs], avail_actions[bs], t_env, test_mode=test_mode)
        return chosen_actions

    def forward(self, ep_batch, t, test_mode=False):
        # output probability with softmax
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

    def forward_maa(self, ep_batch, t, test_mode=False):
        maa_inputs = self._build_maa_inputs(ep_batch, t)
        avail_actions = ep_batch["avail_actions_all"][:, t+1] # bav
        avail_actions = avail_actions.unsqueeze(1).expand(-1, self.agents_num_for_model, -1, -1) # bnav
        maa_outs, self.maa_hidden_states = self.maa(maa_inputs, self.maa_hidden_states)

        # Softmax the agent outputs if they're policy logits
        if self.agent_output_type == "pi_logits":
            if getattr(self.args, "mask_before_softmax", True):
                # Make the logits for unavailable actions very negative to minimise their affect on the softmax
                reshaped_avail_actions = avail_actions.reshape(ep_batch.batch_size * self.agents_num_for_model, -1)
                maa_outs[reshaped_avail_actions == 0] = -1e10
            maa_outs = maa_outs.reshape(ep_batch.batch_size * self.agents_num_for_model * self.n_agents, -1)
            maa_outs = th.nn.functional.softmax(maa_outs, dim=-1)
        return maa_outs.view(ep_batch.batch_size, self.agents_num_for_model, self.n_agents, -1)

    def forward_oracle_nosoftmax(self, ep_batch, t, test_mode=False):
        oracle_inputs = self._build_oracle_inputs(ep_batch, t)
        avail_actions = ep_batch["avail_actions_all"][:, t+1] # bav
        oracle_outs, self.oracle_hidden_states = self.oracle(oracle_inputs, self.oracle_hidden_states)
        # Softmax the agent outputs if they're policy logits
        if self.agent_output_type == "pi_logits":
            if getattr(self.args, "mask_before_softmax", True):
                # Make the logits for unavailable actions very negative to minimise their affect on the softmax
                reshaped_avail_actions = avail_actions.reshape(ep_batch.batch_size * self.agents_num_for_model, -1)
                oracle_outs[reshaped_avail_actions == 0] = -1e10
            oracle_outs = oracle_outs.reshape(ep_batch.batch_size * self.agents_num_for_model * self.n_agents, -1)
        return oracle_outs.view(ep_batch.batch_size, self.n_agents, -1)

    def forward_oracle_logsoftmax(self, ep_batch, t, test_mode=False):
        oracle_inputs = self._build_oracle_inputs(ep_batch, t)
        avail_actions = ep_batch["avail_actions_all"][:, t+1] # bav
        oracle_outs, self.oracle_hidden_states = self.oracle(oracle_inputs, self.oracle_hidden_states)
        # Softmax the agent outputs if they're policy logits
        if self.agent_output_type == "pi_logits":
            if getattr(self.args, "mask_before_softmax", True):
                # Make the logits for unavailable actions very negative to minimise their affect on the softmax
                reshaped_avail_actions = avail_actions.reshape(ep_batch.batch_size * self.agents_num_for_model, -1)
                oracle_outs[reshaped_avail_actions == 0] = -1e10
            oracle_outs = oracle_outs.reshape(ep_batch.batch_size * self.agents_num_for_model * self.n_agents, -1)
        oracle_outs = th.nn.functional.log_softmax(oracle_outs, dim=-1)
        return oracle_outs.view(ep_batch.batch_size, self.n_agents, -1)

    def forward_maa_inputs(self, maa_inputs, ep_batch, t, test_mode=False):
        avail_actions = ep_batch["avail_actions_all"][:, t+1] # bav
        avail_actions = avail_actions.unsqueeze(1).expand(-1, self.agents_num_for_model, -1, -1) # bnav
        maa_outs, self.maa_hidden_states = self.maa(maa_inputs, self.maa_hidden_states)

        # Softmax the agent outputs if they're policy logits
        if self.agent_output_type == "pi_logits":
            if getattr(self.args, "mask_before_softmax", True):
                # Make the logits for unavailable actions very negative to minimise their affect on the softmax
                reshaped_avail_actions = avail_actions.reshape(ep_batch.batch_size * self.agents_num_for_model, -1)
                maa_outs[reshaped_avail_actions == 0] = -1e10
            maa_outs = maa_outs.reshape(ep_batch.batch_size * self.agents_num_for_model * self.n_agents, -1)
            maa_outs = th.nn.functional.softmax(maa_outs, dim=-1)
        return maa_outs.view(ep_batch.batch_size, self.agents_num_for_model, self.n_agents, -1)

    def init_hidden(self, batch_size):
        self.hidden_states = self.agent.init_hidden().unsqueeze(0).expand(batch_size, self.n_adv_agents, -1)  # bav

    def init_hidden_maa(self, batch_size):
        self.maa_hidden_states = self.maa.init_hidden().unsqueeze(0).expand(batch_size, self.agents_num_for_model, -1)  # bav

    def init_hidden_oracle(self, batch_size):
        self.oracle_hidden_states = self.oracle.init_hidden().unsqueeze(0).expand(batch_size, self.agents_num_for_model, -1)  # bav

    def parameters(self):
        return list(self.agent.parameters())

    def maa_parameters(self):
        return list(self.maa.parameters())
    
    def oracle_parameters(self):
        return list(self.oracle.parameters())

    def load_state(self, other_mac):
        self.agent.load_state_dict(other_mac.agent.state_dict())
        self.maa.load_state_dict(other_mac.maa.state_dict())
        self.oracle.load_state_dict(other_mac.oracle.state_dict())

    def mac_eval(self):
        self.agent.eval()
        self.maa.eval()
        self.oracle.eval()

    def mac_train(self):
        self.agent.train()
        self.maa.train()
        self.oracle.train()

    def cuda(self):
        self.agent.cuda()
        self.maa.cuda()
        self.oracle.cuda()

    def save_models(self, path):
        th.save(self.agent.state_dict(), "{}/agent.th".format(path))
        th.save(self.maa.state_dict(), "{}/maa.th".format(path))
        th.save(self.oracle.state_dict(), "{}/oracle.th".format(path))

    def load_models(self, path):
        self.agent.load_state_dict(th.load("{}/agent.th".format(path), map_location=lambda storage, loc: storage))
        self.maa.load_state_dict(th.load("{}/maa.th".format(path), map_location=lambda storage, loc: storage))
        self.oracle.load_state_dict(th.load("{}/oracle.th".format(path), map_location=lambda storage, loc: storage))

    def _build_agents(self, input_shape, maa_input_shape, oracle_input_shape):
        self.agent = agent_REGISTRY[self.args.agent](input_shape, self.args)
        self.maa = agent_REGISTRY['maa'](maa_input_shape, self.args)
        self.oracle = agent_REGISTRY['policy_oracle'](oracle_input_shape, self.args)

    def _build_inputs(self, batch, t):
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

    def _build_maa_inputs(self, batch, t):
        bs = batch.batch_size

        inputs = batch["state"][:, t].unsqueeze(1).expand(-1, self.agents_num_for_model, -1)
        inputs = inputs.reshape(bs*self.agents_num_for_model, -1)

        actions_onehot = batch["actions_all_onehot"][:, t].reshape(bs, -1)
        actions_onehot = actions_onehot.unsqueeze(1).expand(-1, self.agents_num_for_model, -1).reshape(bs*self.agents_num_for_model, -1)
        return th.cat([inputs, actions_onehot], dim=1)

    def _build_oracle_inputs(self, batch, t):
        bs = batch.batch_size

        inputs = batch["state"][:, t].unsqueeze(1).expand(-1, self.agents_num_for_model, -1)
        inputs = inputs.reshape(bs*self.agents_num_for_model, -1)

        actions_onehot = batch["actions_all_onehot"][:, t].reshape(bs, -1)
        actions_onehot = actions_onehot.unsqueeze(1).expand(-1, self.agents_num_for_model, -1).reshape(bs*self.agents_num_for_model, -1)
        return th.cat([inputs, actions_onehot], dim=1)

    def _get_input_shape(self, scheme):
        input_shape = scheme["obs"]["vshape"]
        if self.args.obs_last_action:
            input_shape += scheme["actions_onehot"]["vshape"][0]
        if self.args.obs_agent_id:
            input_shape += self.n_agents
        return input_shape

    def _get_maa_input_shape(self, scheme):
        return scheme["state"]["vshape"] + scheme["actions_onehot"]["vshape"][0] * self.args.n_agents
        
    def _get_oracle_input_shape(self, scheme):
        return scheme["state"]["vshape"] + scheme["actions_onehot"]["vshape"][0] * self.args.n_agents

    def _build_inputs_adv(self, batch, t):
        bs = batch.batch_size
        inputs = []
        inputs.append(batch["obs"][:, t])  # b1av

        if self.args.obs_last_action:
            if t == 0:
                inputs.append(th.zeros_like(batch["actions_onehot"][:, t]))
            else:
                inputs.append(batch["actions_onehot"][:, t-1])
        if self.args.obs_agent_id:
            adv_index = []
            for i in range(batch["obs"].shape[0]):
                adv_index.append(th.eye(self.n_agents, device=batch.device)[:self.args.n_adv_agents])
            adv_index = th.stack(adv_index, dim=0)
            inputs.append(adv_index)
        inputs = th.cat([x.reshape(bs*self.n_adv_agents, -1) for x in inputs], dim=1)
        
        return batch["obs"][:, t], adv_index

    def forward_adv(self, ep_batch, t, test_mode=False):
        agent_inputs = self._build_inputs(ep_batch, t)
        avail_actions = ep_batch["avail_actions"][:, t]
        agent_outs, _ = self.agent(agent_inputs, self.hidden_states)
        if self.agent_output_type == "pi_logits":
            if getattr(self.args, "mask_before_softmax", True):
                reshaped_avail_actions = avail_actions.reshape(ep_batch.batch_size * self.n_adv_agents, -1)
                agent_outs[reshaped_avail_actions == 0] = -1e10
            agent_outs = th.nn.functional.softmax(agent_outs, dim=-1)
        return agent_outs

    def fgsm(self, batch, t, mac):
        celoss = th.nn.CrossEntropyLoss()
        target = th.argmax(self.forward_adv(batch, t), dim=-1)
        target = target.clone().detach().cuda()

        agent_inputs, adv_index = self._build_inputs_adv(batch, t)
        agent_inputs = agent_inputs.clone().detach().squeeze().cuda()
        adv_index = adv_index.clone().detach().squeeze().cuda()
        adv_inputs = agent_inputs.clone().detach()

        for i in range(self.args.iter):
            adv_inputs.requires_grad = True
            adv_in = th.cat([adv_inputs, adv_index], dim=-1).squeeze()
            outputs = mac.forward_fgsm(adv_in, batch, t)

            cost = - celoss(outputs, target)

            grad = th.autograd.grad(cost, adv_inputs, retain_graph=False, create_graph=False)[0]
            adv_inputs = adv_inputs.detach() + self.args.alpha * grad.sign()
            delta = th.clamp(adv_inputs - agent_inputs, min=-self.args.eps, max=self.args.eps)
            adv_inputs = th.clamp(agent_inputs + delta, min=-1, max=1).detach()
        
        return adv_inputs