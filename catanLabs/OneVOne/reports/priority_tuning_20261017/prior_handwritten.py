from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / 'hackathon'))
from priority_policy import Agent as PriorityAgent

class Agent(PriorityAgent):

    def __init__(self):
        super().__init__(priorities=[0.9663865546218487, 0.9663865546218487, 0.7142857142857143, 1.2184873949579833, 1.1344537815126052])
