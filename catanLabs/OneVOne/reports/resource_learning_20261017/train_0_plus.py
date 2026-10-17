from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / 'hackathon'))
from resource_policy import Agent as ResourceAgent

class Agent(ResourceAgent):

    def __init__(self):
        super().__init__(exact_monopoly=True, params=[2.0, 0.0, 0.0])
