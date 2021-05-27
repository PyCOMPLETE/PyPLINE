from PyHEADTAIL.particles.generators import ParticleGenerator
from PyPLINEDParticles import PyPLINEDParticles

class PyPLINEDParticleGenerator(ParticleGenerator):

    def generate(self,name,rank,number):
        ''' Returns a particle  object with the parameters specified
        in the constructor of the Generator object
        '''
        coords = self._create_phase_space()
        particles = PyPLINEDParticles(name,rank,number,self.macroparticlenumber,
                              self.intensity/self.macroparticlenumber,
                              self.charge, self.mass, self.circumference,
                              self.gamma,
                              coords_n_momenta_dict=coords)
        if particles.isReal:
            self._linear_match_phase_space(particles)
        return particles
