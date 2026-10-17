from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / 'hackathon'))
from critic_agent import Agent as HeuristicAgent

class Agent(HeuristicAgent):

    def __init__(self):
        super().__init__(value_weights={'production': 59929672.754816875, 'reachable_production_0': 47717.88982758559, 'reachable_production_1': 14315.366948275678, 'buildable_nodes': 4771.7889827585595, 'army_size': 40975.41546982123, 'public_vps': 8029816805.37781})
