function [proccesingTime, gestoResultKNN] = classifierEMG(emg, datasetGestures, kNNemg,probabilidadkNNUmbral)

nameStaticGestures={'relax';'waveIn';'waveOut';'fist';'open';'pinch'};
numGestures=6;
numTry=15;

ordenFiltro=4;
freqFiltro=0.05;
[Fb, Fa] = butter(ordenFiltro, freqFiltro, 'low'); % creating filter

% Default values

% freq=200; %Hz
% timeSeries = 5; %time to run the script
% windowTime=1; % time of the shift window, 200 samples
% timeShiftWindow=0.25; %0.2 seg para 40 samples de shiftWindow, stride
% timeShift=50;
%hiper parameters
%kNNemg=5;
%probabilidadkNNUmbral=0.8;

% Depend varaibles
% samplesWindow = round(windowTime*freq);
% numExecutionsTimer = ceil(timeSeries/timeShiftWindow);
%numExecutionsTimer = 6;

%% Starting parpool


%% Initializing variables
[sizeDatasetGestures,~]=size(datasetGestures);
% kExecutions=1;

DTWUnknownGesture=zeros(numTry*numGestures,8);
%unknownGesture=zeros(samplesWindow,8);%por cada posicion que la ventana vaya cambiando me dira que gesto o no gesto ha capturado
%% Recognition loop

%% ############# Recibiendo la señal emg a la variable ventana #############################################################

%preparo la matriz window 200x8 para almacenar las muestras
%unknownGesture = circshift(unknownGesture,-shiftSamples);
%unknownGesture(end-shiftSamples+1:end,:) = emg;

%unknownGesture =emg;
%unknownGesture = filtfilt(Fb, Fa,abs(unknownGesture));
tClassification=tic;
emg = filtfilt(Fb, Fa,abs(emg));
%% ##############           DTW-EMG         ######################################################################################

parfor kDatasetGesures=1:sizeDatasetGestures
    for kChannel=1:8
        DTWUnknownGesture(kDatasetGesures,kChannel)=...
            dtw_c(datasetGestures{kDatasetGesures,kChannel}, emg(:, kChannel), 50);
    end
end

%% ##############           KNN-EMG         #######################################################################################
DTWsumChannelEMG=sum(DTWUnknownGesture,2);

% kNNresults contiene la posición de los resultados con menor suma de distancias DTWs.
[~,kNNresults]=sort(DTWsumChannelEMG);

% Esta división cambia el significado de kNNresults.% Ahora contiene el número/ID del gesto correspondiente a los valores% menores de DTW calculados en la linea anterior
kNNresults=ceil(kNNresults/numTry);

% Escogiendo los k más cercanos
kNNresults=kNNresults(1:kNNemg,:);

% % Encontrando el más común entre los vecinos más cercanos
[gestoResultKNN,probGestureKNN]=mode(kNNresults);
%gestoResultKNN = me devuelve el valor que mas se repite
%probGestureKNN = me devuelve el numero de veces que gesTesultKNN se repite

% probabilidad por unidad
probGestureKNN=probGestureKNN/kNNemg;
% numero de reps del gestoResult / KNN

% Asignando nombre al gesto resultante
gestoString=char(nameStaticGestures{gestoResultKNN});


%% Umbral y filtro
if probGestureKNN>=probabilidadkNNUmbral
    fprintf('%s,          %4.2f %%...\n', gestoString , probGestureKNN*100);

else
    fprintf('relax %4.2f %%...\n',probGestureKNN);
    gestoResultKNN=1;
end
proccesingTime = toc(tClassification);
% fprintf('\n');
end