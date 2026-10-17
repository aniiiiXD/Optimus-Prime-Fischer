from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / 'hackathon'))
from critic_agent import Agent as HeuristicAgent

class Agent(HeuristicAgent):

    def __init__(self):
        super().__init__(value_weights={'production': 186323510.55316806, 'reachable_production_0': 310184.62779796077, 'reachable_production_1': 93055.38833938824, 'buildable_nodes': 31018.46277979608, 'army_size': 28055.276704649936, 'public_vps': 110301991.52206407})
