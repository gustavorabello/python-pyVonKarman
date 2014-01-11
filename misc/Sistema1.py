###########################################################################
#       Solucao do sistema de equacoes diferenciais lineares              #
#                                                                         #
#   Esta funcao tem por objetivo solucionar o sistema de EDOs             #       
#    encontrado por Von Karman. Atraves do metodo numerico chamado        #
#    metodo de diferencas finitas.                                        #
#                                                                         #
#   Autores: Rachel Lucena                                                #
#            Jose Pontes                                                  #
#                                                                         #
#  Data: 18 de maio de 2012.                                              #
#                                                                         #
###########################################################################

import numpy as np
from scipy.sparse import lil_matrix,csr_matrix
from scipy.sparse import csc_matrix,coo_matrix
from scipy.sparse import dok_matrix
import scipy.linalg
import scipy.sparse.linalg
import matplotlib.pyplot as plt

def Sistema1(_L,_nptos,_itemax,_epsilon,_p,_cont):

 # Dados de entrada #############################
    
 ### L eh o comprimento da malha.
 ### Recomendado: L=20;
 
 ### _nptos eh o numero de pontos da malha.
 ### Recomendado: _nptos=500;
 
 ### itemax eh numero maximo de iteracoes da aplicacao da correcao.
 ### Recomendado: itemax=1000;
 
 ### epsilon eh o valor do erro.
 ### Recomendado: epsilon=10^(-8);
 
 ### p eh o valor da porcentagem da correcao.
 ### Recomendado: p=0.1; 
 
 ### cont eh o uso dos dados de uma simulacao anterior
 ### Se cont==0 utiliza c.i. do programa, se nao utiliza da simulacao 
 ### anterior.
 
################### Calculo de algumas constantes ######################

 dx = _L/(_nptos-2);   ### Variacao entre cada ponto

 d1=1.0/(2*dx);        ### Constante1 que multiplica as derivadas
 d2=1.0/(dx*dx);       ### Constante2 que multiplica as derivadas
 
 af1=0.4;            ### Constante1 da c.i. de F1
 af2=5*af1;          ### Constante2 da c.i. de F1
 
 g0=1.74;            ### Constante1 da c.i. de G1           
 dxg=dx/g0;          ### Constante2 da c.i. de G1

 a=-0.887;           ### Constante1 da c.i. de H1
 b1=a/4;             ### Constante2 da c.i. de H1
 z=0.75*dx;          ### Constante3 da c.i. de H1
 
 #----x=-dx/2:dx:_L+dx/2;  ### Intervalo dos pontos para plotagem dos perfis
 #xt=transpose(x);    ### Vetor transposto de x

 # Construcao dos vetores: F1,G1,H1,r,X e da matriz A #############
 global F1,G1,H1,D1F1,D1G1,D1H1,D2F1,D2G1,D2H1
 
 x=np.zeros((_nptos,1),dtype=float)   ###
 F1=np.zeros((_nptos,1),dtype=float)   ###
 G1=np.zeros((_nptos,1),dtype=float)   ### Construcao dos vetores F1,G1,H1
 H1=np.zeros((_nptos,1),dtype=float)   ###
 
 b=np.zeros((3*_nptos,1),dtype=float)  ### Construcao do vetor b

 r=np.zeros((3*_nptos,1),dtype=float)  ### Construcao do vetor r
 
 A= lil_matrix((3*_nptos,3*_nptos), dtype='float32')
 
 X=np.zeros((3*_nptos,1),dtype=float)  ### Construcao do vetor X   
 
 D1F1=np.zeros((_nptos,1),dtype=float) ### Construcao dos vetores das
 D1G1=np.zeros((_nptos,1),dtype=float) ### derivadas de primeira ordem
 D1H1=np.zeros((_nptos,1),dtype=float) ###         F1, G1, H1

 D2F1=np.zeros((_nptos,1),dtype=float) ### Construcao dos vetores das
 D2G1=np.zeros((_nptos,1),dtype=float) ### derivadas de segunda ordem
 D2H1=np.zeros((_nptos,1),dtype=float) ###         F1, G1, H1
    
 # Condicoes iniciais das funcoes F1, G1 e H1 #################

 ######## Condicao para usar os perfis da Funcao_Sistema1 como c.i.#########
 if _cont==0:
  ############### Condicoes iniciais da funcao F1 ##################
  for i in range(2,_nptos-2):
   F1[i]=0.4*(np.exp(-af1*i*dx)-np.exp(-af2*i*dx));
 
  F1[_nptos-1]=-F1[_nptos-1]
  ############### Condicoes iniciais da funcao G1 ##################
  G1[0]=1+dx/2;
  G1[1]=1-dx/2;    
  for i in range(2,_nptos-2):
   G1[i]=np.exp(-i*dxg);
 
  G1[_nptos-1]=-G1[_nptos-2];
  ############### Condicoes iniciais da funcao H1 ################## 
  for jj in range(3,_nptos):
   if z<4:
    H1[jj]=b1*(jj-3)*dx;
    z=z+dx;
   else:
    H1[jj]=a;
 #---else:
  #---funcao1=load('DadosSistema1.mat');
  #---F1=funcao1.F1;      G1=funcao1.G1;      H1=funcao1.H1;

##################### Condicoes de contorno do residuo #################### 
#     r(1,1)=0;           r(2,1)=0;           r(3,1)=0;
#     r(3*_nptos-2,1)=0;   r(3*_nptos-1,1)=0;   r(3*_nptos,1)=0;
################### Condicoes de contorno da matriz A #####################

 A[0,0] = 1; A[0,3] = 1;
 A[1,1] = 1; A[1,4] = 1;
 A[2,2] = 1; A[2,5] = 1;
#       
 A[3*_nptos-3,3*_nptos-6] =  1; A[3*_nptos-3,3*_nptos-3] = 1;
 A[3*_nptos-2,3*_nptos-5] =  1; A[3*_nptos-2,3*_nptos-2] = 1;
 A[3*_nptos-1,3*_nptos-4] = -1; A[3*_nptos-1,3*_nptos-1] = 1;   
#
########################### Calculo de b ##################################
 for i in range(3,3*_nptos-5,3):
  b[i]= 0;#10^(-0);
  b[i+1]= 0;#10^(-0);
  b[i+2]= 0;#10^(-0);
########################## Inicio das iteracoes ###########################
 for k in range(0,_itemax):
  ############################ Calculo do Residuo #######################
  for i in range(1,_nptos-1):
   ti=3*(i+1);
   r[ti-3]=d1*(-H1[i-1]+H1[i+1])+2*F1[i];
   #
   r[ti-2]=d2*(F1[i-1]-2*F1[i]+F1[i+1])-d1*H1[i]*(-F1[i-1]+F1[i+1])-F1[i]**2+G1[i]**2;
   #
   r[ti-1]=d2*(G1[i-1]-2*G1[i]+G1[i+1])-d1*H1[i]*(-G1[i-1]+G1[i+1])-2*F1[i]*G1[i];

    ################ Calculando as entradas da Matriz A ###################
  for i in range(1,_nptos-1):
   ti=3*(i+1); 
   tj=3*(i-1); 
   ############### Primeira Equacao (contiduidade) ##############
   A[ti-3,tj+2]=-d1;
   A[ti-3,tj+3]=2;
   A[ti-3,tj+8]=d1;
   ############### Segunda Equacao ##############
   A[ti-2,tj]=d2+d1*H1[i];
   A[ti-2,tj+3]=-2*d2-2*F1[i];
   A[ti-2,tj+4]=2*G1[i]; 
   A[ti-2,tj+5]=-d1*(-F1[i-1]+F1[i+1]);
   A[ti-2,tj+6]=d2-d1*H1[i];
   ############### Terceira Equacao ##############
   A[ti-1,tj+1]=d2+d1*H1[i];
   A[ti-1,tj+3]=-2*G1[i];
   A[ti-1,tj+4]=-2*d2-2*F1[i];
   A[ti-1,tj+5]=-d1*(-G1[i-1]+G1[i+1]); 
   A[ti-1,tj+7]=d2-d1*H1[i];

    #################### Resolvendo o sistema AX=b ########################
  X = scipy.linalg.solve(A.todense(),b-r)
    
    ########################### Calculo da Correcao #######################
  for i in range(0,_nptos):
   ti=3*i;
   F1[i]=F1[i]+_p*X[ti];
   G1[i]=G1[i]+_p*X[ti+1];
   H1[i]=H1[i]+_p*X[ti+2];
    
  ############# Verificando se o residuo eh maior que Epsilon ###########
  q=0;
  for kk in range(3,3*_nptos-3):
   if abs(b[kk]-r[kk]) > _epsilon:
    q=1;
    break
  if q==0:
   break

################ Plotando os perfis F1, G1 e H1 ########################

 x[0] = -dx/2.0
 for i in range(1,_nptos):
  x[i] = x[i-1] + dx

 plt.plot(x,F1,x,G1,x,-H1)
 plt.axis([-0.05,15,-0.05,1.2])
 plt.show()

 title('Resultado parcial de F1, G1, -H1')
 xlabel('Comprimento da malha')
 ylabel('F1, G1, -H1')
 legend(('F1','G1','-H1'))

############################# Pos-processamento ###########################
######## Calculo das derivadas de primeira ordem: D1F1, D1G1, D1H1 ########
 D1F1[0]=d1*(-3*F1[0]+4*F1[1]-F1[2]);
 D1G1[0]=d1*(-3*G1[0]+4*G1[1]-G1[2]);
 D1H1[0]=d1*(-3*H1[0]+4*H1[1]-H1[2]);
 #
 D1F1[_nptos-1]=d1*(3*F1[_nptos-1]-4*F1[_nptos-2]+F1[_nptos-3]);
 D1G1[_nptos-1]=d1*(3*G1[_nptos-1]-4*G1[_nptos-2]+G1[_nptos-3]);
 D1H1[_nptos-1]=d1*(3*H1[_nptos-1]-4*H1[_nptos-2]+H1[_nptos-3]);   
 #    
 for i in range(1,_nptos-2):
  D1F1[i]=d1*(-F1[i-1]+F1[i+1]);
  D1G1[i]=d1*(-G1[i-1]+G1[i+1]);
  D1H1[i]=d1*(-H1[i-1]+H1[i+1]);
###### Calculo das derivadas de segunda ordem: D2F1, D2G1, D2H1 ########
 D2F1[0]=d2*(2*F1[0]-5*F1[1]+4*F1[2]-F1[3]);
 D2G1[0]=d2*(2*G1[0]-5*G1[1]+4*G1[2]-G1[3]);
 D2H1[0]=d2*(2*H1[0]-5*H1[1]+4*H1[2]-H1[3]);
 #
 D2F1[_nptos-1]=d2*(2*F1[_nptos-1]-5*F1[_nptos-2]+4*F1[_nptos-3]-F1[_nptos-4]);
 D2G1[_nptos-1]=d2*(2*G1[_nptos-1]-5*G1[_nptos-2]+4*G1[_nptos-3]-G1[_nptos-4]);
 D2H1[_nptos-1]=d2*(2*H1[_nptos-1]-5*H1[_nptos-2]+4*H1[_nptos-3]-H1[_nptos-4]);  
 #
 for i in range(1,_nptos-2):
  D2F1[i]=d2*(F1[i-1]-2*F1[i]+F1[i+1]);
  D2G1[i]=d2*(G1[i-1]-2*G1[i]+G1[i+1]);
  D2H1[i]=d2*(H1[i-1]-2*H1[i]+H1[i+1]);
 end
#--------------------------------------------------
# ############# Salvando os vetores como arquivo de saida ###################
#     fid = fopen('Resultados_sistema1.dat', 'wt');
#     fprintf(fid, 'x F1 G1 -H1 D1F1 D1G1 D1H1 D2F1 D2G1 D2H1\n');
#     for i=1:_nptos
#         fprintf(fid,'#4.10f #4.10f #4.10f #4.10f #4.10f #4.10f #4.10f #4.10f #4.10f #4.10f\n'...
#         ,x(1,i),F1(i,1),G1(i,1),-H1(i,1),D1F1(i,1),D1G1(i,1),D1H1(i,1)...
#         ,D2F1(i,1),D2G1(i,1),D2H1(i,1));
#     end
#     fclose(fid);
# ##################### Salvando os dados do workspace ######################
# save('DadosSistema1');
# tempo1=toc
#-------------------------------------------------- 

Sistema1(20.0,50,1000,1e-08,0.1,0)
