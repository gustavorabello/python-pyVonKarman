%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%       Solucao do sistema de equacoes diferenciais lineares              %
%                                                                         %
%   Esta funcao tem por objetivo solucionar o sistema de EDOs             %       
%    encontrado por Von Karman. Atraves do metodo numerico chamado        %
%    metodo de diferencas finitas.                                        %
%                                                                         %
%   Autores: Rachel Lucena                                                %
%            Jose Pontes                                                  %
%                                                                         %
%  Data: 18 de maio de 2012.                                              %
%                                                                         %
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

function Sistema1(L,nptos,itemax,epsilon,p,cont)

%%%%%%%%%%%%%%%%%%%%%%%%%%%% Dados de entrada %%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    
    %%% L eh o comprimento da malha.
    %%% Recomendado: L=20;
    
    %%% nptos eh o numero de pontos da malha.
    %%% Recomendado: nptos=500;
    
    %%% itemax eh numero maximo de iteracoes da aplicacao da correcao.
    %%% Recomendado: itemax=1000;
    
    %%% epsilon eh o valor do erro.
    %%% Recomendado: epsilon=10^(-8);
    
    %%% p eh o valor da porcentagem da correcao.
    %%% Recomendado: p=0.1; 
    
    %%% cont eh o uso dos dados de uma simulacao anterior
    %%% Se cont==0 utiliza c.i. do programa, se nao utiliza da simulacao 
    %%% anterior.
    
%%%%%%%%%%%%%%%%%%%%%% Calculo de algumas constantes %%%%%%%%%%%%%%%%%%%%%%
 
    dx = L/(nptos-2);   %%% Variacao entre cada ponto
 
    d1=1/(2*dx);        %%% Constante1 que multiplica as derivadas
    d2=1/(dx*dx);       %%% Constante2 que multiplica as derivadas
    
    af1=0.4;            %%% Constante1 da c.i. de F1
    af2=5*af1;          %%% Constante2 da c.i. de F1
    
    g0=1.74;            %%% Constante1 da c.i. de G1           
    dxg=dx/g0;          %%% Constante2 da c.i. de G1
   
    a=-0.887;           %%% Constante1 da c.i. de H1
    b1=a/4;             %%% Constante2 da c.i. de H1
    z=0.75*dx;          %%% Constante3 da c.i. de H1
    
    x=-dx/2:dx:L+dx/2;  %%% Intervalo dos pontos para plotagem dos perfis
    %xt=transpose(x);    %%% Vetor transposto de x

%%%%%%%%%% Construcao dos vetores: F1,G1,H1,r,X e da matriz A %%%%%%%%%%%%%
    global F1 G1 H1 D1F1 D1G1 D1H1 D2F1 D2G1 D2H1
    
    F1=zeros(nptos,1);          %%%
    G1=zeros(nptos,1);          %%% Construcao dos vetores F1,G1,H1
    H1=zeros(nptos,1);          %%%
    
    b=zeros(3*nptos,1);        %%% Construcao do vetor b
   
    r=zeros(3*nptos,1);        %%% Construcao do vetor r
    
    A=zeros(3*nptos,3*nptos);  %%% Construcao da matriz A
    
    X=zeros(3*nptos,1);        %%% Construcao do vetor X   
    
    D1F1=zeros(nptos,1);    %%% Construcao dos vetores das
    D1G1=zeros(nptos,1);    %%% derivadas de primeira ordem
    D1H1=zeros(nptos,1);    %%%         F1, G1, H1

    D2F1=zeros(nptos,1);    %%% Construcao dos vetores das
    D2G1=zeros(nptos,1);    %%% derivadas de segunda ordem
    D2H1=zeros(nptos,1);    %%%         F1, G1, H1
    
%%%%%%%%%%%%%% Condicoes iniciais das funcoes F1, G1 e H1 %%%%%%%%%%%%%%%%%
 tic
%%%%%%%% Condicao para usar os perfis da Funcao_Sistema1 como c.i.%%%%%%%%%
if cont==0
    %%%%%%%%%%%%%%% Condicoes iniciais da funcao F1 %%%%%%%%%%%%%%%%%%
    for i=3:nptos-2
        F1(i,1)=0.4*(exp(-af1*i*dx)-exp(-af2*i*dx));
    end
    F1(nptos,1)=-F1(nptos-1,1);
    %%%%%%%%%%%%%%% Condicoes iniciais da funcao G1 %%%%%%%%%%%%%%%%%%
    G1(1,1)=1+dx/2;
    G1(2,1)=1-dx/2;    
    for i=3:nptos-2
        G1(i,1)=exp(-i*dxg);
    end
    G1(nptos,1)=-G1(nptos-1,1);
    %%%%%%%%%%%%%%% Condicoes iniciais da funcao H1 %%%%%%%%%%%%%%%%%% 
    for jj=4:nptos
        if z<4
        H1(jj,1)=b1*(jj-3)*dx;
        z=z+dx;
        else
            H1(jj,1)=a;
        end
    end
else
    funcao1=load('DadosSistema1.mat');
    F1=funcao1.F1;      G1=funcao1.G1;      H1=funcao1.H1;
end
%%%%%%%%%%%%%%%%%%%%% Condicoes de contorno do residuo %%%%%%%%%%%%%%%%%%%% 
%     r(1,1)=0;           r(2,1)=0;           r(3,1)=0;
%     r(3*nptos-2,1)=0;   r(3*nptos-1,1)=0;   r(3*nptos,1)=0;
%%%%%%%%%%%%%%%%%%% Condicoes de contorno da matriz A %%%%%%%%%%%%%%%%%%%%%
    A(1,1)=1;       A(1,4)=1;
    A(2,2)=1;       A(2,5)=1;
	A(3,3)=1;       A(3,6)=1;
%       
	A(3*nptos-2,3*nptos-5)=1;   	A(3*nptos-2,3*nptos-2)=1;
	A(3*nptos-1,3*nptos-4)=1;   	A(3*nptos-1,3*nptos-1)=1;
	A(3*nptos,3*nptos-3)=-1;     	A(3*nptos,3*nptos)=1;   
%
%%%%%%%%%%%%%%%%%%%%%%%%%%% Calculo de b %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    j=2;
    for i=4:3:3*nptos-5 
        b(i,1)= 0;%10^(-0);
        b(i+1,1)= 0;%10^(-0);
        b(i+2,1)= 0;%10^(-0);
        j=j+1;
    end
%%%%%%%%%%%%%%%%%%%%%%%%%% Inicio das iteracoes %%%%%%%%%%%%%%%%%%%%%%%%%%%
for k=1:itemax
    %%%%%%%%%%%%%%%%%%%%%%%%%%%% Calculo do Residuo %%%%%%%%%%%%%%%%%%%%%%%
    for i=2:(nptos-1)
        ti=3*i;
        r(ti-2,1)=d1*(-H1(i-1,1)+H1(i+1,1))+2*F1(i,1);
        %
        r(ti-1,1)=d2*(F1(i-1,1)-2*F1(i,1)+F1(i+1,1))-d1*H1(i,1)*...
            (-F1(i-1,1)+F1(i+1,1))-F1(i,1)^2 + G1(i,1)^2;
        %
        r(ti,1)=d2*(G1(i-1,1)-2*G1(i,1)+G1(i+1,1))-d1*H1(i,1)*...
            (-G1(i-1,1)+G1(i+1,1))-2*F1(i,1)*G1(i,1);
    end
    %%%%%%%%%%%%%%%% Calculando as entradas da Matriz A %%%%%%%%%%%%%%%%%%%
    for i=2:nptos-1
        ti=3*i; 
        tj2=3*(i-2); 
            %%%%%%%%%%%%%%% Primeira Equacao (contiduidade) %%%%%%%%%%%%%%
            A(ti-2,tj2+3)=-d1;
            A(ti-2,tj2+4)=2;
            A(ti-2,tj2+9)=d1;
            %%%%%%%%%%%%%%% Segunda Equacao %%%%%%%%%%%%%%
            A(ti-1,tj2+1)=d2+d1*H1(i,1);
            A(ti-1,tj2+4)=-2*d2-2*F1(i,1);
            A(ti-1,tj2+5)=2*G1(i,1); 
            A(ti-1,tj2+6)=-d1*(-F1(i-1,1)+F1(i+1,1));
            A(ti-1,tj2+7)=d2-d1*H1(i,1);
            %%%%%%%%%%%%%%% Terceira Equacao %%%%%%%%%%%%%%
            A(ti,tj2+2)=d2+d1*H1(i,1);
            A(ti,tj2+4)=-2*G1(i,1);
            A(ti,tj2+5)=-2*d2-2*F1(i,1);
            A(ti,tj2+6)=-d1*(-G1(i-1,1)+G1(i+1,1)); 
            A(ti,tj2+8)=d2-d1*H1(i,1);
    end
    %%%%%%%%%%%%%%%%%%%% Resolvendo o sistema AX=b %%%%%%%%%%%%%%%%%%%%%%%%
    X=A\(b-r);      
    
    %%%%%%%%%%%%%%%%%%%%%%%%%%% Calculo da Correcao %%%%%%%%%%%%%%%%%%%%%%%
    for i=1:nptos
        ti1=3*(i-1);
        F1(i,1)=F1(i,1)+p*X(ti1+1,1);
        G1(i,1)=G1(i,1)+p*X(ti1+2,1);
        H1(i,1)=H1(i,1)+p*X(ti1+3,1);
    end
    
    %%%%%%%%%%%%% Verificando se o residuo eh maior que Epsilon %%%%%%%%%%%
    q=0;
    for kk=4:3*nptos-3
        if abs(b(kk,1)-r(kk,1))>epsilon 
            q=1;
            break
        end
    end
    if q==0
        break
    end
end
%%%%%%%%%%%%%%%%%%% Plotando os perfis F1, G1 e H1 %%%%%%%%%%%%%%%%%%%%%%%%
    figure
    plot(x,F1,'r',x,G1,'b',x,-H1,'m')
%    axis([-0.05,15,-0.05,1.2]);
    title('Resultado parcial de F1, G1, -H1')
    xlabel('Comprimento da malha')
    ylabel('F1, G1, -H1')
    legend('F1','G1','-H1')
    grid
%%%%%%%%%%%%%%%%%%%%%%%%%%%%% Pos-processamento %%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%% Calculo das derivadas de primeira ordem: D1F1, D1G1, D1H1 %%%%%%%%
    D1F1(1,1)=d1*(-3*F1(1,1)+4*F1(2,1)-F1(3,1));
    D1G1(1,1)=d1*(-3*G1(1,1)+4*G1(2,1)-G1(3,1));
    D1H1(1,1)=d1*(-3*H1(1,1)+4*H1(2,1)-H1(3,1));
    %
    D1F1(nptos,1)=d1*(3*F1(nptos,1)-4*F1(nptos-1,1)+F1(nptos-2,1));
    D1G1(nptos,1)=d1*(3*G1(nptos,1)-4*G1(nptos-1,1)+G1(nptos-2,1));
    D1H1(nptos,1)=d1*(3*H1(nptos,1)-4*H1(nptos-1,1)+H1(nptos-2,1));   
    %    
    for i=2:nptos-1
        D1F1(i,1)=d1*(-F1(i-1,1)+F1(i+1,1));
        D1G1(i,1)=d1*(-G1(i-1,1)+G1(i+1,1));
        D1H1(i,1)=d1*(-H1(i-1,1)+H1(i+1,1));
    end
%%%%%%%%% Calculo das derivadas de segunda ordem: D2F1, D2G1, D2H1 %%%%%%%%
    D2F1(1,1)=d2*(2*F1(1,1)-5*F1(2,1)+4*F1(3,1)-F1(4,1));
    D2G1(1,1)=d2*(2*G1(1,1)-5*G1(2,1)+4*G1(3,1)-G1(4,1));
    D2H1(1,1)=d2*(2*H1(1,1)-5*H1(2,1)+4*H1(3,1)-H1(4,1));
    %
    D2F1(nptos,1)=d2*(2*F1(nptos,1)-5*F1(nptos-1,1)+4*F1(nptos-2,1)...
                  -F1(nptos-3,1));
    D2G1(nptos,1)=d2*(2*G1(nptos,1)-5*G1(nptos-1,1)+4*G1(nptos-2,1)...
                  -G1(nptos-3,1));
    D2H1(nptos,1)=d2*(2*H1(nptos,1)-5*H1(nptos-1,1)+4*H1(nptos-2,1)...
                  -H1(nptos-3,1));  
    %
    for i=2:nptos-1
        D2F1(i,1)=d2*(F1(i-1,1)-2*F1(i,1)+F1(i+1,1));
        D2G1(i,1)=d2*(G1(i-1,1)-2*G1(i,1)+G1(i+1,1));
        D2H1(i,1)=d2*(H1(i-1,1)-2*H1(i,1)+H1(i+1,1));
    end
%%%%%%%%%%%%% Salvando os vetores como arquivo de saida %%%%%%%%%%%%%%%%%%%
    fid = fopen('Resultados_sistema1.dat', 'wt');
    fprintf(fid, 'x F1 G1 -H1 D1F1 D1G1 D1H1 D2F1 D2G1 D2H1\n');
    for i=1:nptos
        fprintf(fid,'%4.10f %4.10f %4.10f %4.10f %4.10f %4.10f %4.10f %4.10f %4.10f %4.10f\n'...
        ,x(1,i),F1(i,1),G1(i,1),-H1(i,1),D1F1(i,1),D1G1(i,1),D1H1(i,1)...
        ,D2F1(i,1),D2G1(i,1),D2H1(i,1));
    end
    fclose(fid);
%%%%%%%%%%%%%%%%%%%%% Salvando os dados do workspace %%%%%%%%%%%%%%%%%%%%%%
save('DadosSistema1');
tempo1=toc