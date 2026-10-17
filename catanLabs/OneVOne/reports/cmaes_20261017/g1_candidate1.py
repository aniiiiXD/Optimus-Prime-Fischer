from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / 'hackathon'))
from critic_agent import Agent as HeuristicAgent

class Agent(HeuristicAgent):

    def __init__(self):
        super().__init__(value_weights={'production': 108991437.887872, 'reachable_production_0': 68124.23844922753, 'reachable_production_1': 20437.27153476826, 'buildable_nodes': 6812.423844922753, 'army_size': 78673.20104464688, 'public_vps': 533256146.8908381})
