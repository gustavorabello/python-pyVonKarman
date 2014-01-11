## =================================================================== ##
#  this is file main.py, created at 11-Jan-2014                #
#  maintained by Gustavo Rabello dos Anjos                              #
#  e-mail: gustavo.rabello@gmail.com                                    #
## =================================================================== ##


import vonKarman

length  = 10.0
npoints = 800
maxIter = 1000
corr    = 0.1
eps     = 1e-08
k       = 0.040739 


grid  = vonKarman.Grid(length,npoints)
bound = vonKarman.BoundaryConditions(grid)
cond  = vonKarman.InitConditions(grid,bound,'1disk')
vonKarman.Plot(grid.x,cond.F,cond.G,cond.H)
sim = vonKarman.Simulator(grid,cond,bound,k,maxIter,corr,eps)

vonKarman.Plot(grid.x,sim.F,sim.G,sim.H)

