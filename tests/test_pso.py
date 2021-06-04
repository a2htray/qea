import unittest
from qea import PSO, PSOParams


class TestPSO(unittest.TestCase):
    def test_one(self):
        model = PSO(PSOParams(
            num_generations=100,
            num_solutions=10,
            num_dimensions=2,
            random_state=1,
        ))
        model.run()


if __name__ == '__main__':
    unittest.main()
