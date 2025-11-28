import torch
from vmas.scenarios.dispersion import Scenario as DispersionScenario
from vmas.scenarios.discovery import Scenario as DiscoveryScenario

class DenseScenario:

    def __init__(self, n_agents, share_reward, penalise_by_time):
        super().__init__()

        self.n_agents = n_agents
        self.share_reward = share_reward
        self.penalise_by_time = penalise_by_time

    def make_world(self, batch_dim, device, **kwargs):
        return super().make_world(
            batch_dim=batch_dim,
            device=device,
            n_agents=self.n_agents,
            share_reward=self.share_reward,
            penalise_by_time=self.penalise_by_time,
            **kwargs
        )

    def reset_world_at(self, env_index: int = None):
        super().reset_world_at(env_index)

        noise_level = 0.1

        for agent in self.world.agents:

            base_pos = torch.zeros(
                self.world.dim_p,
                device=self.world.device,
                dtype=torch.float32,
            )

            noise = torch.randn(
                self.world.dim_p,
                device=self.world.device,
                dtype=torch.float32,
            ) * noise_level

            agent.set_pos(
                base_pos + noise,
                batch_index=env_index,
            )

            curr_dist = self._get_min_dist_to_targets(agent)

            if not hasattr(agent, "prev_min_dist"):
                agent.prev_min_dist = curr_dist.clone()
            else:
                if env_index is None:
                    agent.prev_min_dist = curr_dist.clone()
                else:
                    agent.prev_min_dist[env_index] = curr_dist[env_index]

    def reward(self, agent):
        base_reward = super().reward(agent) * 1000.0

        current_distance = self._get_min_dist_to_targets(agent)

        shaping_reward = - current_distance * 100.0

        # curr_dist = self._get_min_dist_to_targets(agent)
        delta_distance = agent.prev_min_dist - current_distance
        #
        if current_distance.item() < 0.1 and delta_distance < 0.05:
             penalize_stand_still = -100.0
        else:
            penalize_stand_still = 0.0
        # else:
        #     shaping_reward = delta_distance * 100.0
        # agent.prev_min_dist = curr_dist.detach()
        total_reward = shaping_reward + penalize_stand_still + base_reward
        return total_reward


    def _get_min_dist_to_targets(self, agent):
        targets_pos = torch.stack([t.state.pos for t in self.world.landmarks], dim=1)
        agent_pos = agent.state.pos.unsqueeze(1)
        dists = torch.linalg.norm(agent_pos - targets_pos, dim=2)
        min_dist, _ = torch.min(dists, dim=1)
        return min_dist


class DenseDispersionScenario(DenseScenario, DispersionScenario):
    pass

class DenseDiscoveryScenario(DenseScenario, DiscoveryScenario):
    pass