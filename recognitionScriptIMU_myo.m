%reconocimiento de solo sEMG
close all
clear
clc
 
global  imu kAux soIMU sfIMU

nameSignals={'relax';'waveIn';'waveOut';'fist';'open';'pinch';'up';'down';'left';'right';'forward';'backward'};
%usuarios =[24, 25, 26, 27, 32, 33, 35, 36, 37, 39, 40, 41, 42];
usuarios =[24, 25, 26, 27, 32, 33, 36];
resultadoUsuarios=zeros(length(usuarios),1);
noUser=1;

resultadosHiperParametrosIMU=zeros(length(usuarios),2);

a=1;
b=1;

addpath(genpath(pwd));
for i=1:length(usuarios)
    
    numeroUser=num2str(usuarios(i));
    nameUser= ['user_0' numeroUser];
    idxUser=usuarios(i);
    %% Llamada a creacion del dataset del usaurio
    %nameUser= 'user_024';
    %idxUser=24;
    
    nameMoves={'relax';'up';'down';'left';'right';'forward';'backward'};
    numMoves=7;
    numTry=15;
    
    [datasetIMUs]=datasetConstructionIMU(nameUser,idxUser,numTry,numMoves);
    
    % Default values
    %timeSeriesIMU = 1.25;   % tiempo de reconocimiento
    %freq=200; %Hz
    timeSeriesIMU = 5;   % tiempo de reconocimiento
    freq=50; %Hz
    
    
    %windowTimeIMU=0.95;      % tiempo de ventana deslizante
    timeShiftIMU=50;            % samples del paso, stride
    b=1;
    superMatriz=zeros(8,8);
    for windowTimeIMU=1.9:0.1:3.3
        %for timeShiftIMU=15:5:50
        
        timeShiftWindowIMU=timeShiftIMU/freq;   % tiempo del paso, stride
        
        %hiper parameters
        kNNimu=7;
        thresholdkNNimu=0.8;
        
        % Depend varaibles
        samplesWindowIMU = round(windowTimeIMU*freq);
        numExecutionsTimerIMU = ceil(timeSeriesIMU/timeShiftWindowIMU);
        %numExecutionsTimer = 6;
        
        %% Starting parpool
        fprintf('Por favor espere mientras se carga la pool parallel.\n')
        
        if isempty(gcp)
            parpool;
            beep
        end
        
        fprintf('Listo.\n')
        
        %% Initializing variables
        [sizeDatasetIMUs,~]=size(datasetIMUs);
        tClassificacionVectorIMU=zeros(numExecutionsTimerIMU,1);
        % Gesto resultante finales, al aplicar filtro y umbral.
        probabilidadVectorIMU=zeros(numTry*numMoves,numExecutionsTimerIMU);
        
        
        %resultados por cada ventana 180x5
        resultadosGestoVectorIMU=zeros(180,numExecutionsTimerIMU);
        
        %resultado final usando la moda de la ventana
        resultadosGestoVectorModaIMU=zeros(180,1);
        
        numAciertosIMU=0;
        predictedLabelsIMU=cell(180,1);
        
        %% n de samples que se van a testear
        tsoIMU=91;
        tsfIMU=180;
        %testSampleIMU = 91;
        
        
        %fprintf('ejecutando clasificador...\n\n')
        %for kNNemg=1:10
        %for probabilidadkNNUmbral=0.1:0.1:1%
        %numAciertosIMU=0;
        for testSampleIMU=tsoIMU:tsfIMU
            DTWUnknownGestureIMU=zeros(numTry*numMoves,4);
            unknownGestureIMU=zeros(samplesWindowIMU,4);%por cada posicion que la ventana vaya cambiando me dira que gesto o no gesto ha capturado
            %% Recognition loop
            kAux=0;     % timer loops counter
            kExecutions=1;
            samplesObtained=0;
            VectorParaModaIMU=zeros(1,numExecutionsTimerIMU);
            %inicio de valores de ventana para EMG
            soIMU=1;
            sfIMU=samplesWindowIMU;
            fprintf('--------- KNNimu: %d ---- PROB:%4.2f%% -----\n',kNNimu, thresholdkNNimu*100);
            while kAux<numExecutionsTimerIMU
                
                tClassification=tic;
                ventanaIMU(idxUser, testSampleIMU, samplesWindowIMU, timeShiftIMU);
                %% ############# Recibiendo la señal IMU a la variable ventana #############################################################
                
                [shiftSamples,~]=size(imu);% cuantas filas del trozo de EMG que voy a trabjar, 200?
                samplesObtained=samplesObtained+shiftSamples;%total de samples, 200 + 200 + 200 ...
                
                %preparo la matriz window 200x8 para almacenar las muestras
                unknownGestureIMU = circshift(unknownGestureIMU,-shiftSamples);
                unknownGestureIMU(end-shiftSamples+1:end,:) = imu;
                
                %% ##############           DTW-EMG         ######################################################################################
                
                parfor kDatasetIMU=1:sizeDatasetIMUs
                    for kChannel=1:4
                        DTWUnknownGestureIMU(kDatasetIMU,kChannel)=...
                            dtw_c(datasetIMUs{kDatasetIMU,kChannel}, unknownGestureIMU(:, kChannel), 50);
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
                moveString=char(nameMoves{gestoResultKNN});
                
                
                %% Umbral y filtro
                
                if probGestureKNN>=thresholdkNNimu
                    fprintf('%s_ts-%d. w-%d. %s,          %4.2f %%...\n', nameUser, testSampleIMU, kExecutions,moveString , probGestureKNN*100);
                    %resultadosGestoVectorIMU(testSampleIMU, kExecutions)=gestoResultKNN;
                    VectorParaModaIMU(1, kExecutions)=gestoResultKNN;
                    probabilidadVectorIMU(testSampleIMU,kExecutions)=probGestureKNN*100;
                    %para insertar etiqueta globalmente con 12 gestos
                    if gestoResultKNN~=1
                        gestoResultKNN=gestoResultKNN+5;
                        resultadosGestoVectorIMU(testSampleIMU, kExecutions)=gestoResultKNN;
                        VectorParaModaIMU(1, kExecutions)=gestoResultKNN;
                    end
                    resultadosGestoVectorIMU(testSampleIMU, kExecutions)=gestoResultKNN;
                    VectorParaModaIMU(1, kExecutions)=gestoResultKNN;
                else
                    fprintf('EMG noGesto-noUmbral %4.2f %%...\n',probGestureKNN);
                end
                
                
                % Cont de num de aciertos usando criterio de votacion a la moda
                if kExecutions==numExecutionsTimerIMU
                    %quito los relax
                    VectorParaModaIMU(VectorParaModaIMU == 1) = [];
                    VectorParaModaIMU(VectorParaModaIMU == 0) = [];
                    if isnan(mode(VectorParaModaIMU)) || (mode(VectorParaModaIMU(1,:))==0)
                        resultadosGestoVectorModaIMU(testSampleIMU, 1) = 1;
                        predictedLabelsIMU{testSampleIMU,1} = char(nameSignals{1});
                        if 1==ceil(testSampleIMU/numTry)
                            numAciertosIMU=numAciertosIMU+1;
                            fprintf('\nrelax\n');
                        end
                    else
                        resultadosGestoVectorModaIMU(testSampleIMU, 1) = mode(VectorParaModaIMU(1,:));
                        predictedLabelsIMU{testSampleIMU,1} = char(nameSignals{mode(VectorParaModaIMU(1,:))});
                    end
                    
                    %cuento los correctos, correspondientes a sus 15
                    if mode(VectorParaModaIMU(1,:))==ceil(testSampleIMU/numTry)
                        numAciertosIMU=numAciertosIMU+1;
                    end
                end
                
                tClassificacionVectorIMU(testSampleIMU,kExecutions) = toc(tClassification);
                
                kExecutions=kExecutions+1;
            end
            fprintf('\n');
            
        end
        
        beep
        
        testSamples=(tsfIMU-tsoIMU)+1;
        acc=(numAciertosIMU*100)/testSamples;
        fprintf('%s ACC %d:%d test-samples: %4.2f %%\nNum. Aciertos: %d\n', nameUser, tsoIMU, tsfIMU, acc, numAciertosIMU);
        fprintf('KNNemg: %d Prob:%4.2f%% -----\n',kNNimu, thresholdkNNimu*100);
        fprintf('samplesWindows:%d, TimeShift:%d\n', samplesWindowIMU, timeShiftIMU);
        %guardo acc de knn y umbral
        %resultadosHiperParametrosIMU(kNNimu,fix(thresholdkNNimu*10))=acc;
        
        resultadosHiperParametrosIMU(i,b)=acc;
        b=b+1;
    end
    
end %para for de usuarios