clear;
clc;
datos = load('Data_Synthetic_3000x50x3c.mat').trn; 
X = datos.xc;
Y = datos.y;
      
revolver = [Y, X];
[R, C] = size(revolver);
revolver = revolver(randperm(R), :);
Y = revolver(:, 1);
X = revolver(:, 2:C);
% Se determinan las particiones
particiones = [70, 30];
% Se determina el numero de particiones
nPart = numel(particiones(:, 1));
K = 3;
fprintf('\nTotal de variables = 50\n');
fprintf('Total de muestras = 3000\n');
fprintf('Tasa de Reconicimientos para 3 vecinos = 99.67\n');
tasa = knnParticion(particiones, datos.n, X, Y, K);

function tasa = knnParticion(particion, nDatos, X, Y, K) 
    % Porcentaje de training
    porcTraining = particion(1, 1);
    % numero que corresponde al porcentaje
    nTraining = floor((porcTraining * nDatos) / 100);
    %TEST y training
    tasa = knn_clasificacion(X, Y, K, nTraining, nDatos);

end



function RR = knn_clasificacion(X,Y,K,nTraining, nDatos)
    % Utiliza la particion training
    IDX_TRN=1:nTraining;
    IDX_TST=nTraining:nDatos;
        
    X_TRN = X(IDX_TRN,:);
    Y_TRN = Y(IDX_TRN);
    X_TST = X(IDX_TST, :);
    Y_TST = Y(IDX_TST);
    Y_PREDICTED = zeros(length(IDX_TST), 1);
    
    n = length(IDX_TST);
    for j=1:n
        d_e=sqrt(sum((X_TRN-repmat(X_TST(j,:),length(IDX_TRN),1)).^2,2));  
        %CALCULO VECINOS
        [~,IDX]=sort(d_e);
        clase_k_vecinos=Y_TRN(IDX(1:K));

         %EL GANADOR
         Y_PREDICTED(j) = mode(clase_k_vecinos);
     end
     RR = sum(Y_PREDICTED==Y_TST)/length(IDX_TST);
     fprintf('Matriz de Confusión\nC1\tC2\tC3');
     % Recorro todo las clases
     for i=1:3
         clase = Y_PREDICTED == i;
         fprintf('\nC%d', i);
         for j=1:3 
            fprintf('\t%d',sum(clase & Y_TST == j));
        end
     end
end