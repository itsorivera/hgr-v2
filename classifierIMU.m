function [proccesingTime, gestoResultKNN] = classifierIMU(quat, datasetIMUs,kNNimu,thresholdkNNimu)

nameDynamicGestures={'relax';'up';'down';'left';'right';'forward';'backward'};
numMoves=7;
numTry=15;
% Default values
% freq=50; %Hz
% timeSeriesIMU = 1.25;   % tiempo de reconocimiento
% 
% %windowTimeIMU=0.95;      % tiempo de ventana deslizante
% strideSamplesIMU=50;            % samples del paso, stride
% 
% windowTimeIMU=0.95;
% 
% 
% strideTimeIMU=strideSamplesIMU/freq;   % tiempo del paso, stride

%hiper parameters
%kNNimu=7;
%thresholdkNNimu=0.8;

% Depend varaibles
% samplesWindowIMU = round(windowTimeIMU*freq);
% numExecutionsTimerIMU = ceil(timeSeriesIMU/strideTimeIMU);
%numExecutionsTimer = 6;

%% Initializing variables
[sizeDatasetIMUs,~]=size(datasetIMUs);
% tClassificacionVectorIMU=zeros(numExecutionsTimerIMU,1);
% % Gesto resultante finales, al aplicar filtro y umbral.
% probabilidadVectorIMU=zeros(numTry*numMoves,numExecutionsTimerIMU);
% 
% 
% %resultados por cada ventana 180x5
% resultadosGestoVectorIMU=zeros(180,numExecutionsTimerIMU);
% 
% %resultado final usando la moda de la ventana
% resultadosGestoVectorModaIMU=zeros(180,1);
% 
% testSampleIMU = testSample;
DTWUnknownGestureIMU=zeros(numTry*numMoves,4);
% unknownGestureIMU=zeros(samplesWindowIMU,4);%por cada posicion que la ventana vaya cambiando me dira que gesto o no gesto ha capturado
%% Recognition loop

% kExecutions=1;
% samplesObtained=0;
% VectorParaModaIMU=zeros(1,numExecutionsTimerIMU);


%ventanaIMU(idxUser, testSampleIMU, samplesWindowIMU, timeShiftIMU);
%% ############# Recibiendo la señal IMU a la variable ventana #############################################################

%[shiftSamples,~]=size(quat);% cuantas filas del trozo de EMG que voy a trabjar, 200?
%samplesObtained=samplesObtained+shiftSamples;%total de samples, 200 + 200 + 200 ...

%preparo la matriz window 200x8 para almacenar las muestras
%unknownGestureIMU = circshift(unknownGestureIMU,-shiftSamples);
%unknownGestureIMU(end-shiftSamples+1:end,:) = quat;
%unknownGestureIMU = quat;

tClassification=tic;
%% ##############           DTW-EMG         ######################################################################################

parfor kDatasetIMU=1:sizeDatasetIMUs
    for kChannel=1:4
        DTWUnknownGestureIMU(kDatasetIMU,kChannel)=...
            dtw_c(datasetIMUs{kDatasetIMU,kChannel}, quat(:, kChannel), 50);
    end
end

%% ##############           KNN-EMG         #######################################################################################
DTWsumChannelIMU=sum(DTWUnknownGestureIMU,2);

% kNNresults contiene la posición de los resultados con menor suma de distancias DTWs.
[~,kNNresults]=sort(DTWsumChannelIMU);

% Esta división cambia el significado de kNNresults.% Ahora contiene el número/ID del gesto correspondiente a los valores% menores de DTW calculados en la linea anterior
kNNresults=ceil(kNNresults/numTry);

% Escogiendo los k más cercanos
kNNresults=kNNresults(1:kNNimu,:);

% % Encontrando el más común entre los vecinos más cercanos
[gestoResultKNN,probGestureKNN]=mode(kNNresults);
%gestoResultKNN = me devuelve el valor que mas se repite
%probGestureKNN = me devuelve el numero de veces que gesTesultKNN se repite

% probabilidad por unidad
probGestureKNN=probGestureKNN/kNNimu;
% numero de reps del gestoResult / KNN

% Asignando nombre al gesto resultante
moveString=char(nameDynamicGestures{gestoResultKNN});


%% Umbral y filtro



if probGestureKNN>=thresholdkNNimu
    fprintf('%s,          %4.2f %%...\n', moveString , probGestureKNN*100);
    if gestoResultKNN~=1
        gestoResultKNN=gestoResultKNN+5;
    end
else
    fprintf('relax %4.2f %%...\n',probGestureKNN);
    gestoResultKNN=1;
end
proccesingTime = toc(tClassification);
end