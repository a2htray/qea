import numpy as np
from .params import Params
from .pso import BaseParticle, default_on_position_change
import copy


class QDPSOParams(Params):
    max_z = 1/np.log10(np.sqrt(2))

    def __init__(self, 
        z=2, 
        on_position_change=default_on_position_change, **kargs) -> None:
        super().__init__(**kargs)
        self.z = z
        self._check_z()
        self.on_position_change = on_position_change

    def _check_z(self):
        if not (0 < self.z < QDPSOParams.max_z):
            raise Exception('the given z is out of range, range=({low}, {hgih})'.format(low=0, high=QDPSOParams.max_z))


class QDPSOParticle(BaseParticle):
    def __init__(self, params: QDPSOParams) -> None:
        super().__init__(
            params.rs, 
            params.num_dimensions,
            params.bound_lows,
            params.bound_highs,
            params.fitness_func)
        self._params = params
    
    def update_position(self, gen_i, particle_i, global_best_particle):
        position_before = self.position

        for i in range(self._params.num_dimensions):
            r1, r2 = self._params.rs.uniform(0, 1, 2)
            pi = (r1 * self.best_position()[i] + r2 * global_best_particle.position[i]) / (r1 + r2)

            L = self._params.z * abs(self.position[i] - pi)
            U = self._params.rs.random()

            if U > 0.5:
                self.position[i] = pi - L * np.log10(1/U)
            else:
                self.position[i] = pi + L * np.log10(1/U)

            # 检查是否越界
            if self.position[i] > self._params.bound_highs[i]:
                self.position[i] = self._params.bound_highs[i]
            
            if self.position[i] < self._params.bound_lows[i]:
                self.position[i] = self._params.bound_lows[i]

        self._params.on_position_change(
            gen_i=gen_i,
            particle_i=particle_i,
            position_before=position_before,
            position_after=self.position)

        # 更新完速度后确定是否为该粒子历史最优
        if self.evaluate() < self._params.fitness_func(self.best_position()):
            self.best_position_index = len(self.positions) - 1

class QDPSO:
    def __init__(self, params: QDPSOParams) -> None:
        self.params = params
        self.particles = [
            QDPSOParticle(params) for _ in range(self.params.num_solutions)
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
                self.particles[j].update_position(
                    gen_i=i,
                    particle_i=j,
                    global_best_particle=self.global_best_particle,
                )


            self.params.on_generation(
                gen=i,
                solutions=[particle.position for particle in self.particles],
                best_solution=self.global_best_particle.position,
                params=self.params)