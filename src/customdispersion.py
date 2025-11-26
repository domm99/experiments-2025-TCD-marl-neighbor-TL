import torch
from vmas.scenarios.dispersion import Scenario as DispersionScenario


class DenseDispersionScenario(DispersionScenario):

    def __init__(self, n_agents, share_reward, penalise_by_time):
        super().__init__()

        self.saved_n_agents = n_agents
        self.saved_share_reward = share_reward
        self.saved_penalise_by_time = penalise_by_time

    def make_world(self, batch_dim, device, **kwargs):
        return super().make_world(
            batch_dim=batch_dim,
            device=device,
            n_agents=self.saved_n_agents,
            share_reward=self.saved_share_reward,
            penalise_by_time=self.saved_penalise_by_time,
            **kwargs
        )

    def reset_world_at(self, env_index: int = None):
        super().reset_world_at(env_index)

        for agent in self.world.agents:

            curr_dist = self._get_min_dist_to_targets(agent)

            if not hasattr(agent, "prev_min_dist"):
                agent.prev_min_dist = curr_dist.clone()
            else:
                if env_index is None:
                    agent.prev_min_dist = curr_dist.clone()
                else:
                    agent.prev_min_dist[env_index] = curr_dist[env_index]

    def reward(self, agent):
        base_reward = super().reward(agent)
        curr_dist = self._get_min_dist_to_targets(agent)
        delta_distance = agent.prev_min_dist - curr_dist
        shaping_reward = delta_distance * 100.0
        agent.prev_min_dist = curr_dist.detach()
        total_reward = base_reward + shaping_reward
        return total_reward


    def _get_min_dist_to_targets(self, agent):
        targets_pos = torch.stack([t.state.pos for t in self.world.landmarks], dim=1)
        agent_pos = agent.state.pos.unsqueeze(1)
        dists = torch.linalg.norm(agent_pos - targets_pos, dim=2)
        min_dist, _ = torch.min(dists, dim=1)
        return min_dist