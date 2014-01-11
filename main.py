## =================================================================== ##
#  this is file main.py, created at 11-Jan-2014                #
#  maintained by Gustavo Rabello dos Anjos                              #
#  e-mail: gustavo.rabello@gmail.com                                    #
## =================================================================== ##


import vonKarman

length = 20.0
npoints = 500
maxIter = 1000
corr    = 0.1
eps     = 1e-08

grid  = vonKarman.Grid(length,npoints)
cond  = vonKarman.InitConditions(grid)
bound = vonKarman.BoundaryConditions(grid)
sim = vonKarman.Simulator(grid,cond,bound,maxIter,corr,eps)

vonKarman.Plot(grid.x,sim.F,sim.G,sim.H)



