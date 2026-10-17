from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / 'hackathon'))
from critic_agent import Agent as HeuristicAgent

class Agent(HeuristicAgent):

    def __init__(self):
        super().__init__(value_weights={'production': 118028924.81768274, 'reachable_production_0': 32864.05242339693, 'reachable_production_1': 9859.21572701908, 'buildable_nodes': 3286.4052423396934, 'army_size': 43621.380434239916, 'public_vps': 7104709780.528846})
