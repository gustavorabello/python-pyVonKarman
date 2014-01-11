###########################################################################
#       Solucao do sistema de equacoes diferenciais lineares              #
#                                                                         #
#   Esta funcao tem por objetivo solucionar o sistema de EDOs             #   
#   encontrado por Von Karman. Atraves do metodo numerico chamado         #
#   metodo de diferencas finitas.                                         #
#                                                                         #
#   Autores: Rachel Lucena                                                #
#            Jose Pontes                                                  #
#            Norberto Mangiavacchi                                        #
#            Gustavo Anjos                                                #
#                                                                         #
#  Data: 11 de janeiro de 2013                                            #
#                                                                         #
###########################################################################

import numpy as np
from scipy.sparse import lil_matrix,csr_matrix
from scipy.sparse import csc_matrix,coo_matrix
from scipy.sparse import dok_matrix
import scipy.linalg
import scipy.sparse.linalg
import matplotlib.pyplot as plt


class Grid:
 def __init__(_self,_L,_npoints):

  _self.L = _L
  _self.npoints = _npoints
  _self.dx = _self.L/(_self.npoints-2);   ### Variacao entre cada ponto

  _self.x=np.zeros((_self.npoints,1),dtype=float)   ###

  _self.x[0] = -_self.dx/2.0
  for i in xrange(1,_self.npoints):
   _self.x[i] = _self.x[i-1] + _self.dx

class InitConditions:
 def __init__(_self,_grid,_boundary,_type):

  grid = _grid            ## Recebe a classe Grid
  _self.F = _boundary.F
  _self.G = _boundary.G
  _self.H = _boundary.H

  ### Condicao para usar os perfis da Funcao_Sistema1 como c.i.
  if _type is '1disk':
   f1 = 0.4 
   f2 = 5*f1 
   g1 = 1.74
   g2 = grid.dx/g1
   h1 = 0.615
   h2 = 0.5

   _self.G[1] = 1.0-grid.dx/2.0;
   for i in xrange(2,grid.npoints-1):
    _self.F[i] = 0.4*(np.exp(-f1*grid.x[i])-np.exp(-f2*grid.x[i]))
    _self.G[i] = np.exp(-i*g2)
    _self.H[i] = np.exp(-h1*grid.x[i])-np.exp(-h2*grid.x[i])
  
  elif _type is 'vonKarman':
   f1 = 0.4 
   f2 = 5*f1 
   g1 = 1.74
   g2 = grid.dx/g1
   h1   = -0.887       
   h2  = h1/4.0
   h3   = 0.75*grid.dx

   _self.G[1] = 1.0-grid.dx/2.0;
   for i in xrange(2,grid.npoints-1):
    _self.F[i] = 0.4*(np.exp(-f1*grid.x[i])-np.exp(-f2*grid.x[i]))
    _self.G[i] = np.exp(-i*g2)
    _self.H[i] = h2*grid.x[i]
    if grid.x[i] > 4:
     _self.H[i] = h1
  
  else:
   for i in xrange(2,grid.npoints-1):
    _self.F[i] = 0.0
    _self.G[i] = 0.0
    _self.H[i] = 0.0


class BoundaryConditions:
 def __init__(_self,_grid):

  grid = _grid

  ### Building vectors F,G,H,b,r and matrix A
  _self.F=np.zeros((grid.npoints,1),dtype=float)   
  _self.G=np.zeros((grid.npoints,1),dtype=float) 
  _self.H=np.zeros((grid.npoints,1),dtype=float)  
  _self.A = lil_matrix((3*grid.npoints,3*grid.npoints), dtype='float32')
  _self.b = np.zeros((3*grid.npoints,1), dtype=float)  
  _self.r = np.zeros((3*grid.npoints,1), dtype=float) 

  # Dirichlet (first point)
  _self.F[0] = 0.0
  _self.G[0] = 1.0+grid.dx/2.0;
  _self.H[0] = -_self.H[1]

  # Neumann (last point)
  _self.F[grid.npoints-1] = -_self.F[grid.npoints-2]
  _self.G[grid.npoints-1] = -_self.G[grid.npoints-2]
  _self.H[grid.npoints-1] = -_self.H[grid.npoints-2]

  ## Condicoes de contorno do residuo 
  # r(0,0)=0; r(1,0)=0; r(2,0)=0;
  # r(3*grid.npoints-3,0)=0; 
  # r(3*grid.npoints-2,0)=0; 
  # r(3*grid.npoints-1,0)=0;
  
  ## Condicoes de contorno da matriz A 
  _self.A[0,0] = 1; _self.A[0,3] = 1;
  _self.A[1,1] = 1; _self.A[1,4] = 1;
  _self.A[2,2] = 1; _self.A[2,5] = 1;
  #       
  _self.A[3*grid.npoints-3,3*grid.npoints-6] =  1; _self.A[3*grid.npoints-3,3*grid.npoints-3] = 1;
  _self.A[3*grid.npoints-2,3*grid.npoints-5] =  1; _self.A[3*grid.npoints-2,3*grid.npoints-2] = 1;
  _self.A[3*grid.npoints-1,3*grid.npoints-4] = -1; _self.A[3*grid.npoints-1,3*grid.npoints-1] = 1;   
  #

  ## Calculo de b 
  for i in xrange(3,3*grid.npoints-5,3):
   _self.b[i]   = 0; #10^(-0);
   _self.b[i+1] = 0; #10^(-0);
   _self.b[i+2] = 0; #10^(-0);


class Simulator:
 def __init__(_self,_grid,_initConditions,_boundary,_konst,_iterMax,_p,_epsilon):

  grid = _grid
  initCond = _initConditions
  boundary = _boundary
  _self.A = boundary.A
  _self.b = boundary.b
  _self.r = boundary.r
  _self.F = _initConditions.F
  _self.G = _initConditions.G
  _self.H = _initConditions.H
  d1 = 1.0/(2*grid.dx);  ## Constante1 que multiplica as derivadas
  d2 = 1.0/(grid.dx*grid.dx); ## Constante2 que multiplica as derivadas

  X=np.zeros((3*grid.npoints,1),dtype=float)  ### Construcao do vetor X   

  for k in xrange(0,_iterMax):
   print "iteration: " + str(k)
   ############################ Calculo do Residuo #######################
   for i in xrange(1,grid.npoints-1):
    ti=3*(i+1);
    tj=3*(i-1); 

    _self.r[ti-3]=d1*(-_self.H[i-1]+_self.H[i+1])+2*_self.F[i];
    #
    _self.r[ti-2]=d2*(_self.F[i-1]-2*_self.F[i]+_self.F[i+1])-d1*_self.H[i]*(-_self.F[i-1]+_self.F[i+1])-_self.F[i]**2+_self.G[i]**2-_konst;
    #
    _self.r[ti-1]=d2*(_self.G[i-1]-2*_self.G[i]+_self.G[i+1])-d1*_self.H[i]*(-_self.G[i-1]+_self.G[i+1])-2*_self.F[i]*_self.G[i];
 
    ## 
	### Otimizar este trecho! -----------------------
	## 
    ## Primeira Equacao (contiduidade)
    _self.A[ti-3,tj+2]=-d1;
    _self.A[ti-3,tj+3]=2;
    _self.A[ti-3,tj+8]=d1;
    ## Segunda Equacao 
    _self.A[ti-2,tj]=d2+d1*_self.H[i];
    _self.A[ti-2,tj+3]=-2*d2-2*_self.F[i];
    _self.A[ti-2,tj+4]=2*_self.G[i]; 
    _self.A[ti-2,tj+5]=-d1*(-_self.F[i-1]+_self.F[i+1]);
    _self.A[ti-2,tj+6]=d2-d1*_self.H[i];
    ## Terceira Equacao 
    _self.A[ti-1,tj+1]=d2+d1*_self.H[i];
    _self.A[ti-1,tj+3]=-2*_self.G[i];
    _self.A[ti-1,tj+4]=-2*d2-2*_self.F[i];
    _self.A[ti-1,tj+5]=-d1*(-_self.G[i-1]+_self.G[i+1]); 
    _self.A[ti-1,tj+7]=d2-d1*_self.H[i];
    ## 
	### ---------------------------------------------
	## 

   ## Resolvendo o sistema AX=b ########################
   _self.X = scipy.linalg.solve(_self.A.todense(),_self.b-_self.r)
     
   ## Calculo da Correcao #######################
   for i in xrange(0,grid.npoints):
    ti=3*i;
    _self.F[i]=_self.F[i]+_p*_self.X[ti];
    _self.G[i]=_self.G[i]+_p*_self.X[ti+1];
    _self.H[i]=_self.H[i]+_p*_self.X[ti+2];
     
   ## Verificando se o residuo eh maior que Epsilon ###########
   q=0;
   for i in xrange(3,3*grid.npoints-3):
    if abs(_self.b[i]-_self.r[i]) > _epsilon:
     q=1;
     break
   if q==0:
    break


class PostProcessing:
 def __init__(_self,_grid,_simulator):

  grid = _grid
  sim = _simulator
  d1 = 1.0/(2*grid.dx);       ## Constante1 que multiplica as derivadas
  d2 = 1.0/(grid.dx*grid.dx); ## Constante2 que multiplica as derivadas
  
  ## Construcao dos vetores das derivadas de primeira ordem
  _self.D1F = np.zeros((grid.npoints,1),dtype=float) 
  _self.D1G = np.zeros((grid.npoints,1),dtype=float) 
  _self.D1H = np.zeros((grid.npoints,1),dtype=float) 

  ## Construcao dos vetores das derivadas de segunda ordem
  _self.D2F = np.zeros((grid.npoints,1),dtype=float) 
  _self.D2G = np.zeros((grid.npoints,1),dtype=float) 
  _self.D2H = np.zeros((grid.npoints,1),dtype=float) 

  ## Calculo das derivadas de primeira ordem: D1F, D1G, D1H 
  _self.D1F[0]=d1*(-3*sim.F[0]+4*sim.F[1]-sim.F[2]);
  _self.D1G[0]=d1*(-3*sim.G[0]+4*sim.G[1]-sim.G[2]);
  _self.D1H[0]=d1*(-3*sim.H[0]+4*sim.H[1]-sim.H[2]);

  ## Calculo das derivadas de segunda ordem: D2F, D2G, D2H ########
  _self.D2F[0]=d2*(2*sim.F[0]-5*sim.F[1]+4*sim.F[2]-sim.F[3]);
  _self.D2G[0]=d2*(2*sim.G[0]-5*sim.G[1]+4*sim.G[2]-sim.G[3]);
  _self.D2H[0]=d2*(2*sim.H[0]-5*sim.H[1]+4*sim.H[2]-sim.H[3]);

  _self.D1F[grid.npoints-1]=d1*(3*F[grid.npoints-1]-4*sim.F[grid.npoints-2]+sim.F[grid.npoints-3]);
  _self.D1G[grid.npoints-1]=d1*(3*sim.G[grid.npoints-1]-4*sim.G[grid.npoints-2]+sim.G[grid.npoints-3]);
  _self.D1H[grid.npoints-1]=d1*(3*sim.H[grid.npoints-1]-4*sim.H[grid.npoints-2]+sim.H[grid.npoints-3]);   
  #
  _self.D2F[grid.npoints-1]=d2*(2*sim.F[grid.npoints-1]-5*sim.F[grid.npoints-2]+4*sim.F[grid.npoints-3]-sim.F[grid.npoints-4]);
  _self.D2G[grid.npoints-1]=d2*(2*sim.G[grid.npoints-1]-5*sim.G[grid.npoints-2]+4*sim.G[grid.npoints-3]-sim.G[grid.npoints-4]);
  _self.D2H[grid.npoints-1]=d2*(2*sim.H[grid.npoints-1]-5*sim.H[grid.npoints-2]+4*sim.H[grid.npoints-3]-sim.H[grid.npoints-4]);  
  #    
  for i in xrange(1,grid.npoints-2):
   _self.D1F[i]=d1*(-sim.F[i-1]+sim.F[i+1]);
   _self.D1G[i]=d1*(-sim.G[i-1]+sim.G[i+1]);
   _self.D1H[i]=d1*(-sim.H[i-1]+sim.H[i+1]);
   _self.D2F[i]=d2*(sim.F[i-1]-2*F[i]+sim.F[i+1]);
   _self.D2G[i]=d2*(sim.G[i-1]-2*G[i]+sim.G[i+1]);
   _self.D2H[i]=d2*(sim.H[i-1]-2*H[i]+sim.H[i+1]);

class Plot:
 def __init__(_self,_x,_F,_G,_H):

 ## Plotando os perfis F, G e H ########################

  plt.plot(_x,_F,_x,_G,_x,-_H)
  plt.axis([-0.05,15,-0.05,1.2])

  plt.title('Partial result of F, G, -H')
  plt.xlabel('Domain Length')
  plt.ylabel('F, G, -H')
  plt.legend(('F','G','-H'))

  plt.show()
  #savefig('test.pdf')


