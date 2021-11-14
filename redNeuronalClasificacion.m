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
% Preprocesar datos
datos = mediaCeroVarianza(datos);

MU = mean(datos);
sigma = std(datos);

theta = 0.1; % Error maximo
tasa = 0.100; % Tasa de aprendizaje 
EPOCAS = 10; % Epocas maximas

pesosFinales = redNeuronal(datos, etiquetas, theta, tasa, EPOCAS);

function salida = MSE(errores)
    salida = sqrt(sum(errores.^2));
end


function pesos = redNeuronal(X, etiquetas, theta, tasa, epocas)
    % Esto se debe de calcular aletoriamente
    pesos = [ 0.388, 0.614, 2.334 ];
    %pesosIniciales = rand([1, 3], 1, 5);

    fprintf("::: theta = %.4f\n::: tasa = %.4f\nMAX-EPOCH = %d\n", theta, tasa, EPOCAS);
    fprintf("Pesos Iniciales");
    disp(pesos);
    n = numel(X(:, 1));
    
    for i=1:epocas
       [pesosFinales, mse ] = calculaEpoca(X, etiquetas, n);
    end
end

function [ pesosFinales, mse ] = calcularEpoca(X, etiquetas, n)
    for j=1:n
        
    end
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
