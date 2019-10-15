import numpy as np
import random


tam_population = 4
max_layers = 5

individuals = []
for i in range(tam_population):
	individual_in = np.append(np.array([2]), np.random.randint(low=2, high=300, size=random.randint(1,(max_layers-1))))
	individual_out = np.append(np.copy(individual_in[1:]),np.array([2])) 
	
	individuals.append([individual_in, individual_out])

individuals = np.array(individuals)	

print('\n')

print(individuals)
print (type(individuals))
print('\n')
print(individuals[0])
print(individuals[0, 0])