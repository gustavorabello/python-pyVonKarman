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
#k       = 0.0
bcType  = '1disk'


grid  = vonKarman.Grid(length,npoints)
init = vonKarman.InitConditions(grid,bcType)
vonKarman.Plot(grid.x,init.F,init.G,init.H,'init.pdf')
bound = vonKarman.BoundaryConditions(grid,init)
vonKarman.Plot(grid.x,bound.F,bound.G,bound.H,'bound.pdf')
sim = vonKarman.Simulator(grid,init,bound,k,maxIter,corr,eps)

vonKarman.Plot(grid.x,sim.F,sim.G,sim.H,'final.pdf')

