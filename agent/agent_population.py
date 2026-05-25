import random
from agent.agent import Agent
from configs import load_config
from utils import get_random_position

config = load_config()


class AgentPopulation:
    def __init__(self, screen, goals):
        self.screen = screen
        self.population_size = config["population"]["agent_population_size"]
        start_position = get_random_position(300)
        self.population = [Agent(screen, start_position, goals, True) for _ in range(self.population_size)]
        self.goals = goals

    def forward(self, step):
        for player in self.population:
            player.update(step)
            
    def draw(self):
        for player in self.population:
            player.draw()

    def reproduce(self):

        new_population = []

        start_position = get_random_position(300)

        for _ in range(self.population_size - 5):
            sample = random.sample(self.population, 10)

            best_agent = max(sample, key=(lambda agent: agent.fitness))

            new_agent = Agent(self.screen, start_position, self.goals, False)
            new_agent.inherit_hidden_layers(best_agent.hidden_layers)

            new_agent.mutate()

            new_population.append(new_agent)

        top_agents = sorted(self.population, key=lambda agent: agent.fitness, reverse=True)[:5]

        for agent in top_agents:
            new_agent = Agent(self.screen, start_position, self.goals, False)
            new_agent.inherit_hidden_layers(agent.hidden_layers)

            new_population.append(new_agent)

        self.population = new_population

        return top_agents[0].fitness
