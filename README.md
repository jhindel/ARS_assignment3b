# ARS_assignment3b

Collision-free navigation of mobile robot which covers as much area as possible.

Controller based on artificial neural network with two outputs, each output controls speed of one wheel. Robot has 12 infrared distance sensors (30°distance) as input to ANN.

EA functions: 
* Population size of 100
* Genetic representation: Bitstring 
* Truncated rank based selection 
* Genetical replacement
* One point crossover
* Mutation
* Fitness function based on removed dust, distance of sensors to the wall in addition to discouragement of negative velocity and spinning.

