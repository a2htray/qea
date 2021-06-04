import unittest
import numpy as np
from qea import QDPSO, QDPSOParams


class TestQDPSO(unittest.TestCase):
    def test_one(self):
        def fitness_func(solution):
            return sum(np.power((solution + 0.5), 2))

        fitnesses = []
        def on_generation(gen, solutions, best_solution, params):
            fitnesses.append(params.fitness_func(best_solution))
                
        model = QDPSO(QDPSOParams(
            num_generations=500,
            fitness_func=fitness_func,
            num_solutions=10,
            num_dimensions=2,
            on_generation=on_generation,
            random_state=1,
            bound_low=-10,
            bound_high=10,
        ))
        model.run()

        print(fitnesses)


if __name__ == '__main__':
    unittest.main()