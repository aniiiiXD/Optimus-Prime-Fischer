from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / 'hackathon'))
from critic_agent import Agent as CriticAgent

class Agent(CriticAgent):

    def __init__(self):
        super().__init__(checkpoint=None, exploration=0.0, blend=0.25)
