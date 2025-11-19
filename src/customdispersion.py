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
        base_reward = super().reward(agent) * 100.0

        curr_dist = self._get_min_dist_to_targets(agent)

        delta_distance = agent.prev_min_dist - curr_dist

        if delta_distance > 0:
            distance_reward = 10
        else:
            distance_reward = -10

        return distance_reward + base_reward

    def _get_min_dist_to_targets(self, agent):
        """Funzione helper per calcolare la distanza dal target più vicino"""
        # agent.state.pos è (num_envs, 2)
        # target.state.pos è (num_envs, 2)

        # Raccogliamo le posizioni di tutti i target (landmarks)
        # self.world.landmarks contiene i target nello scenario Dispersion
        targets_pos = torch.stack([t.state.pos for t in self.world.landmarks],
                                  dim=1)  # Shape: (num_envs, num_targets, 2)

        # Espandiamo la pos dell'agente per broadcasting
        agent_pos = agent.state.pos.unsqueeze(1)  # Shape: (num_envs, 1, 2)

        # Calcoliamo la distanza da TUTTI i target
        dists = torch.linalg.norm(agent_pos - targets_pos, dim=2)  # Shape: (num_envs, num_targets)

        # Prendiamo solo la distanza dal target più vicino (minimo)
        min_dist, _ = torch.min(dists, dim=1)  # Shape: (num_envs,)

        return min_dist