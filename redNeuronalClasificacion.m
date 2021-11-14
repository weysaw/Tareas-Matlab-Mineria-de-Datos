% Axel Leonardo Ornelas Munguía 
% 13 de Noviembre del 2021
% Tarea 11 Redes Neuronales
clear; clc;


%load datos_clasificacion.mat
X = [6.1358, 8.8427, 1; 
    -0.1113, 2.9378, 2;
     7.8209, 9.4302, 1;
     1.0552, 2.7683, 2; 
     7.4538, 11.2482, 1];
[datos, etiquetas] = separarDatos(X);
% Preprocesar etiquetas
etiquetas = etiquetas - 1;

MU = mean(datos);
sigma = std(datos);

% Preprocesar datos
datos = mediaCeroVarianza(datos);


theta = 0.1; % Error maximo
epsilon = 0.500; % Se usa para determinar la funcion de salida
tasa = 0.100; % Tasa de aprendizaje 
EPOCAS = 10; % Epocas maximas

W = redNeuronalClasificar(datos, etiquetas, theta, tasa, epsilon, EPOCAS);
fprintf("\n\nPesos Finales"); disp(W);
fprintf("\n\nEcuación de la Red Neuronal\tY = ");
for i = 1:numel(W)
    fprintf("%6.3fx%d ", W(i), i - 1);
end
fprintf("\nMU: "); disp(MU); fprintf("\nSIGMA: ");
disp(sigma);
fprintf("\nw: "); disp(W);
fprintf("\n");


function W = redNeuronalClasificar(X, etiquetas, theta, tasa, epsilon, epocas)
    % Esto se debe de calcular aletoriamente
    W = [ 0.688, 0.614, 2.334 ];
    %pesosIniciales = rand([1, 3], 1, 5);

    fprintf("::: theta = %.4f\n:::epsilon = %.4f\n::: tasa = %.4f\nMAX-EPOCH = %d\n", theta, epsilon, tasa, epocas);
    fprintf("Pesos Iniciales");
    disp(W);
    n = numel(X(:, 1));
    
    for i=1:n
       W = calcularEpoca(X, W, etiquetas, epsilon, tasa, n, i);
    end
end

function W = calcularEpoca(X, W, etiquetas, epsilon, tasa, n, i)
    nCol = numel(X(1, :));
    % Menos datos que los pesos se significa que se debe de hacer x = 1
    nPesos = numel(W);
    % Indica cuantas columnas con unos deben de agregarse
    nuevaCol = ones(n, (nPesos - nCol));
    X = [nuevaCol, X]; 
    errores = zeros(n, 1);
    fprintf("**********Epoca %d**********\n\ni\t\t", i);
    for j = 0:nPesos-1
        fprintf("x%d\t\t", j);
    end
    for j = 0:nPesos-1
        fprintf("w%d\t\t", j);
    end
    fprintf("Z\tY\tt\t\te\t");
    for j = 0:nPesos-1
        fprintf("n*e*x%d\t", j);
    end
    fprintf("\n-------------------------------------------------------------------------------------------------\n");
    for j = 1:n
        % Se calcula Z(X)
        Z = sum(W.*X(j, :));
        % Se obtiene la salida 
        Y = Z >= epsilon;
        t = etiquetas(j);
        % Se calcula el error
        errores(j) = Y - t;
        % Determina cuanto se va modificar un peso
        modificacionesPeso = tasa * errores(j) .* X(j, :);
        
        fprintf("%d\t", j);
        for k = 1:nPesos
            fprintf("%6.3f\t", X(j, k));
        end
        for k = 1:nPesos
            fprintf("%6.3f\t", W(k));
        end
        fprintf("%6.3f\t%d\t%d\t%6.3f", Z, Y, t, errores(j));
        for k = 1:nPesos
            fprintf("\t%6.3f", modificacionesPeso(k));
        end
        fprintf("\n");
        % Se modifica el peso con las modficiaciones
        W = W - modificacionesPeso;
    end
    rmse = sqrt(sum(errores.^2));
    fprintf("-------------------------------------------------------------------------------------------------\nRMSE = %6.3f\n", rmse);
end

% Separa los datos si las etiquetas vienen en conjunto
function [datos, etiquetas] = separarDatos(X)
    nCol = numel(X(1, :));
    datos = X(:, 1:(nCol - 1));
    etiquetas = X(:, nCol);
end

% Estandarización: Xi - ux -> Media Cero
function Xu = mediaCero(X)
    Xu = X - mean(X); 
end

% Estandarización: (Xi - ux) / σx -> Media cero varianza 1
function XuVar = mediaCeroVarianza(X)
    Xu = mediaCero(X);
    XuVar = Xu./std(X);
end
