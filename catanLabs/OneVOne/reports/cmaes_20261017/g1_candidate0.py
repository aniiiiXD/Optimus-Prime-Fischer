from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / 'hackathon'))
from critic_agent import Agent as HeuristicAgent

class Agent(HeuristicAgent):

    def __init__(self):
        super().__init__(value_weights={'production': 248742113.5912204, 'reachable_production_0': 53612.90010387112, 'reachable_production_1': 16083.870031161336, 'buildable_nodes': 5361.290010387112, 'army_size': 32929.747506897416, 'public_vps': 59990339832.20385})
