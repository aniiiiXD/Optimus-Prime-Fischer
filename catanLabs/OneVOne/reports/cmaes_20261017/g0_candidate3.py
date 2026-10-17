from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / 'hackathon'))
from critic_agent import Agent as HeuristicAgent

class Agent(HeuristicAgent):

    def __init__(self):
        super().__init__(value_weights={'production': 33578392.58929311, 'reachable_production_0': 92050.70311949417, 'reachable_production_1': 27615.21093584825, 'buildable_nodes': 9205.070311949416, 'army_size': 41475.76362796064, 'public_vps': 35751756.11021924})
