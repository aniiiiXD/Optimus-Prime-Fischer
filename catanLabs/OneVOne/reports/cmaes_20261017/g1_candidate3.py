from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / 'hackathon'))
from critic_agent import Agent as HeuristicAgent

class Agent(HeuristicAgent):

    def __init__(self):
        super().__init__(value_weights={'production': 145951474.56273726, 'reachable_production_0': 49769.88329838361, 'reachable_production_1': 14930.964989515083, 'buildable_nodes': 4976.988329838361, 'army_size': 62450.598005527536, 'public_vps': 284202101.80354553})
