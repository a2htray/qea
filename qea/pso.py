from .params import Params
from .utils import init_solution, init_velocity
import numpy as np
import copy


def default_on_velocity_change(gen_i, particle_i, velocity_before, velocity_after):
    pass


def default_on_position_change(gen_i, particle_i, position_before, position_after):
    pass


class PSOParams(Params):
    init_velocity_types = [int, list]

    def __init__(self, 
        init_velocity_low=0,
        init_velocity_high=1,
        inertia_weight=0.5,
        cognitive_c=2,
        social_c=2,
        on_velocity_change=default_on_velocity_change,
        on_position_change=default_on_position_change,
        **kargs) -> None:
        """
        PSO 参数类

        init_velocity_low: int|list 初始速度向量最小限制,
        init_velocity_high int|list 初始速度向量最大限制,
        inertia_weight: 参数1,
        cognitive_c: 参数2,
        social_c: 参数3,
        on_velocity_change=default_on_velocity_change,
        on_position_change=default_on_position_change,
        """
        super().__init__(**kargs)

        # init_velocity_low
        if type(init_velocity_low) not in PSOParams.bound_types:
            raise Exception('the type of given init_velocity_low is not in [int, list]')

        if isinstance(init_velocity_low, int):
            self.init_velocity_lows = np.array([init_velocity_low] * self.num_dimensions)
        else:
            if len(init_velocity_low) != self.num_dimensions:
                raise Exception("the length of given init_velocity_low does not match")
            else:
                self.init_velocity_lows = np.array(init_velocity_low)
        
        # init_velocity_high
        if type(init_velocity_high) not in PSOParams.bound_types:
            raise Exception('the type of given init_velocity_high is not in [int, list]')

        if isinstance(init_velocity_high, int):
            self.init_velocity_highs = np.array([init_velocity_high] * self.num_dimensions)
        else:
            if len(init_velocity_high) != self.num_dimensions:
                raise Exception("the length of given init_velocity_high does not match")
            else:
                self.init_velocity_highs = np.array(init_velocity_high)

        self.inertia_weight = inertia_weight
        self.cognitive_c = cognitive_c
        self.social_c = social_c
        self.on_velocity_change = on_velocity_change
        self.on_position_change = on_position_change


class BaseParticle:
    def __init__(self, rs, num_dimensions: int, bound_lows, bound_highs, fitness_func) -> None:
        # 当前的位置
        self.position = init_solution(rs, num_dimensions, bound_lows, bound_highs)
        # 粒子所有经过的位置，二维
        self.positions = [
            self.position.copy(),
        ]
        # 经过的最好位置的下标
        self.best_position_index = 0
        self._fitness_func = fitness_func

    def best_position(self):
        return self.positions[self.best_position_index]
    
    def evaluate(self):
        return self._fitness_func(self.position)


class Particle(BaseParticle):
    def __init__(self, params: PSOParams) -> None:
        super().__init__(
            params.rs, 
            params.num_dimensions,
            params.bound_lows,
            params.bound_highs,
            params.fitness_func)
        self._params = params
        # 当前速度
        self.velocity = init_velocity(
            self._params.rs,
            self._params.num_dimensions,
            self._params.init_velocity_lows,
            self._params.bound_highs)

    def update_velocity(self, gen_i, particle_i, global_best_particle):
        global_best_position = global_best_particle.position
        velocity_before = self.velocity

        for i in range(self._params.num_dimensions):
            r1, r2 = self._params.rs.uniform(0, 1, 2)

            cognitive_velocity = self._params.cognitive_c * r1 * (self.best_position()[i] - self.position[i])
            social_velocity = self._params.social_c * r2 * (global_best_position[i] - self.position[i])
            self.velocity[i] = self._params.inertia_weight * self.velocity[i] + cognitive_velocity + social_velocity

        self._params.on_velocity_change(
            gen_i=gen_i,
            particle_i=particle_i,
            velocity_before=velocity_before,
            velocity_after=self.velocity)

    def update_position(self, gen_i, particle_i):
        position_before = self.position

        self.position = self.position + self.velocity

        # 检查是否越界
        for i in range(self._params.num_dimensions):
            if self.position[i] > self._params.bound_highs[i]:
                self.position[i] = self._params.bound_highs[i]
            
            if self.position[i] < self._params.bound_lows[i]:
                self.position[i] = self._params.bound_lows[i]
        
        self.positions.append(self.position.copy())

        self._params.on_position_change(
            gen_i=gen_i,
            particle_i=particle_i,
            position_before=position_before,
            position_after=self.position)

        # 更新完速度后确定是否为该粒子历史最优
        if self.evaluate() < self._params.fitness_func(self.best_position()):
            self.best_position_index = len(self.positions) - 1


class PSO:
    def __init__(self, params: PSOParams) -> None:
        self.params = params

        # 当前代的种群
        self.particles = [
            Particle(self.params) for _ in range(self.params.num_solutions)
        ]
        # 全局最优的解
        self.global_best_particle = None
        self.params.on_initialization([particle.position for particle in self.particles], self.params)
    
    def run(self):
        for i in range(self.params.num_generations):
            fitness_values = [particle.evaluate() for particle in self.particles]
            best_i = np.argmin(fitness_values)

            if self.global_best_particle is None or self.global_best_particle.evaluate() > fitness_values[best_i]:
                self.global_best_particle = copy.deepcopy(self.particles[best_i])
            
            for j in range(self.params.num_solutions):
                self.particles[j].update_velocity(
                    gen_i=i,
                    particle_i=j,
                    global_best_particle=self.global_best_particle)
                self.particles[j].update_position(
                    gen_i=i, 
                    particle_i=j)
            
            self.params.on_generation(
                gen=i,
                solutions=[particle.position for particle in self.particles],
                best_solution=self.global_best_particle.position,
                params=self.params)
