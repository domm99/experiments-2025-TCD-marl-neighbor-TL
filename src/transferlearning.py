from utils import *
from config import Config

def get_agent_position(env, agent_id):
    return env.agents[agent_id].state.pos.cpu().numpy().flatten()

def transfer_learning_with_restricted_communication(cfg: Config, current_agents, agents, env):
    prefix = current_agents[0].split('_')[0]
    current_agents_ids = {int(i.split('_')[-1]) for i in current_agents}
    if len(current_agents_ids) != len(env.agents):
        raise Exception(f'''
        Something is wrong with the number of agents, please check :)
        ------------
        Current agents: {current_agents_ids} 
        ------------
        Raw agents: {[a.name for a in env.agents]} 
        ''')

    for a in current_agents_ids:
        pos = get_agent_position(env, a)
        other_agents = current_agents_ids - {a}
        neighbors_positions = {
            neigh_id: get_agent_position(env, neigh_id)
            for neigh_id in other_agents
        }
        neighbors_distances = {k: euclidean_distance(pos, v) for k, v in neighbors_positions.items()}
        visible_neighbors = [f'{prefix}_{k}' for k, v in neighbors_distances.items() if v < cfg.communication_range]
        print(f'Visible neighbors: {visible_neighbors}')
        if len(visible_neighbors) > 0:
            teacher_id = ss_average_uncertainty(visible_neighbors, agents)
            exp = agents[teacher_id].experience
            agents[f'{prefix}_{a}'].learn_from_teacher(exp)

def transfer_learning_all_agents(current_agents, agents):
    teacher_id = ss_average_uncertainty(current_agents, agents)
    print(f'Selected teacher: {teacher_id}')
    exp = agents[teacher_id].experience
    for aid in current_agents:
        agents[aid].learn_from_teacher(exp)