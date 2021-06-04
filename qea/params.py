import numpy as np

def default_fitness_func(solution)->float:
    return 1.0 / len(solution) * sum(np.power(solution, 2))


def default_on_initialization(initialized_solutions, params):
    pass
    # print("Model initialization")
    # for i, solution in enumerate(initialized_solutions):
    #     print("{i}th {solution} fitness value={fitness_value}".format(i=i, solution=solution, fitness_value=params.fitness_func(solution)))


def default_on_generation(gen: int, solutions, best_solution, params):
    pass
    # print("Gen {gen}th best {best_solution} fitness value={fitness_value}".format(gen=gen, best_solution=best_solution, fitness_value=params.fitness_func(best_solution)))

class Params(object):
    bound_types = [int, list]

    def __init__(self, 
        num_generations: int=200, 
        fitness_func=default_fitness_func,
        num_solutions: int=60,
        num_dimensions: int=8,
        bound_low=-4,
        bound_high=4,
        on_initialization=default_on_initialization,
        on_generation=default_on_generation,
        random_state=None) -> None:
        """
        进化算法参数基类

        num_generations: int 迭代数
        fitness_func: 适应值函数
        num_solutions: int 解的个数
        num_dimensions: int 解向量的维数
        bound_low: int|list 解向量最小限制
        bound_high: int|list 解向量最大限制
        random_state: 固定随机
        """
        self.num_generations = abs(num_generations)
        self.fitness_func = fitness_func
        self.num_solutions = num_solutions
        self.num_dimensions = num_dimensions

        # bound_low
        if type(bound_low) not in Params.bound_types:
            raise Exception('the type of given bound_low is not in [int, list]')

        if isinstance(bound_low, int):
            self.bound_lows = np.array([bound_low] * self.num_dimensions)
        else:
            if len(bound_low) != self.num_dimensions:
                raise Exception("the length of given bound_low does not match")
            else:
                self.bound_lows = np.array(bound_low)
        
        # bound_high
        if type(bound_high) not in Params.bound_types:
            raise Exception('the type of given bound_high is not in [int, list]')

        if isinstance(bound_high, int):
            self.bound_highs = np.array([bound_high] * self.num_dimensions)
        else:
            if len(bound_high) != self.num_dimensions:
                raise Exception("the length of given bound_high does not match")
            else:
                self.bound_highs = np.array(bound_high)


        self.on_initialization = on_initialization
        self.on_generation = on_generation
        self.rs = np.random.RandomState(random_state)
