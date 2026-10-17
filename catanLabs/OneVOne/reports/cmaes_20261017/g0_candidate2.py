from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / 'hackathon'))
from critic_agent import Agent as HeuristicAgent

class Agent(HeuristicAgent):

    def __init__(self):
        super().__init__(value_weights={'production': 109330479.03797121, 'reachable_production_0': 76899.31825172789, 'reachable_production_1': 23069.795475518367, 'buildable_nodes': 7689.931825172789, 'army_size': 110386.09219671022, 'public_vps': 7243449062.292085})
