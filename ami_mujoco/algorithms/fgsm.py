import torch
import torch.nn as nn
from algorithms.utils.util import check

class FGSM():
    def __init__(self, args, victim, attack, device=torch.device("cpu")):
        self.args = args
        self.eps = args.epsilon
        self.alpha = args.alpha
        self.iter = args.iteration
        self.attack = attack
        self.victim = victim
        self.device = device
        self.tpdv = dict(dtype=torch.float32, device=device)
        self.loss = nn.MSELoss()
        
    @torch.enable_grad()
    def forward(self, obs, rnn_states_actor, rnn_states_actor_victim, masks, available_actions=None, deterministic=False):
        obs = check(obs).to(**self.tpdv)
        input = obs.clone().detach()
        rnn_states_actor = check(rnn_states_actor).to(**self.tpdv)
        rnn_states_actor_victim = check(rnn_states_actor_victim).to(**self.tpdv)
        masks = check(masks).to(**self.tpdv)
        if available_actions is not None:
            available_actions = check(available_actions).to(**self.tpdv)

        target = self.attack.actor.get_logits(obs, rnn_states_actor, masks, available_actions)
        if self.args.env_name == "StarCraft2":
            target = target.argmax(dim=-1).clone().detach()

        for i in range(self.iter):    
            obs = obs.requires_grad_()
            output = self.victim.actor.get_logits(obs, rnn_states_actor_victim, masks, available_actions)

            cost = - self.loss(output, target)

            grad = torch.autograd.grad(cost, obs, retain_graph=False, create_graph=False)[0]

            obs = obs.clone().detach() + self.alpha * grad.sign()
            delta = torch.clamp(obs - input, min=-self.eps, max=self.eps)
            obs = torch.clamp(input + delta, min=-1, max=1).clone().detach()

        return obs.cpu().numpy()
