%reconocimiento de solo sEMG
close all
clear
clc

global  emg  leido kAux so sf gesto

%usuarios =[24, 25, 26, 27, 32, 33, 35, 36, 37, 39, 40, 41, 42];
usuarios =[24, 25, 26, 27, 32, 33, 35];
resultadoUsuarios=zeros(length(usuarios),10);
noUser=1;

resultadosHiperParametrosEMG=zeros(10,10);

for i=1:length(usuarios)
    
    addpath(genpath(pwd));
    
    %% Llamada a creacion del dataset del usaurio
    % nameUser= 'user_035';
    % idxUser=35;
    numeroUser=num2str(usuarios(i));
    nameUser= ['user_0' numeroUser];
    idxUser=usuarios(i);
    
    ordenFiltro=4;
    freqFiltro=0.05;
    [Fb, Fa] = butter(ordenFiltro, freqFiltro, 'low'); % creating filter
    nameGestures={'relax';'waveIn';'waveOut';'fist';'open';'pinch'};
    numGestures=6;
    numTry=15;
    
    [datasetGestures,umbralMax]=datasetConstructionEMG(nameUser,idxUser,Fb,Fa,numTry,numGestures);
    
    %hiper parameters
    kNNemg=7;
    probabilidadkNNUmbral=0.8;
    
    % Default values
    freq=500; %Hz
    timeSeries = 5; %time to run the script
    
    
    
    strideSamples=500;
    strideTime=strideSamples/freq; %0.2 seg para 40 samples de shiftWindow, stride
    numExecutionsTimer = ceil(timeSeries/strideTime);
    
    b=1;
    for windowTime=1.75:0.05:1.9 % time of the shift window, 200 samples
        % Depend varaibles
        samplesWindow = round(windowTime*freq);
        
        
        %% Starting parpool
        fprintf('Por favor espere mientras se carga la pool parallel.\n')
        
        if isempty(gcp)
            parpool;
            beep
        end
        
        fprintf('Listo.\n')
        
        %% Initializing variables
        [sizeDatasetGestures,~]=size(datasetGestures);
        tClassificacionVector=zeros(numExecutionsTimer,1);
        % Gesto resultante finales, al aplicar filtro y umbral.
        probabilidadVector=zeros(numTry*numGestures,numExecutionsTimer);
        
        
        %resultados por cada ventana 180x5
        resultadosGestoVectorEMG=zeros(180,numExecutionsTimer);
        
        %resultado final usando la moda de la ventana
        resultadosGestoVectorModa=zeros(180,1);
        
        numAciertos=0;
        predictedLabels=cell(180,1);
        
        %% n de samples que se van a testear
        testSampleInicio=16;
        testSampleFinal=90;
        %testSample = 17;
        
        
        %fprintf('ejecutando clasificador...\n\n')
        %for kNNemg=1:10
        %for probabilidadkNNUmbral=0.1:0.1:1%
        %numAciertos=0;
        for testSample=testSampleInicio:testSampleFinal
            DTWUnknownGesture=zeros(numTry*numGestures,8);
            unknownGesture=zeros(samplesWindow,8);%por cada posicion que la ventana vaya cambiando me dira que gesto o no gesto ha capturado
            %% Recognition loop
            leido=0;    % flag to know when new data is ready
            kAux=0;     % timer loops counter
            kExecutions=1;
            samplesObtained=0;
            VectorParaModa=zeros(1,numExecutionsTimer);
            %inicio de valores de ventana para EMG
            so=1;
            sf=samplesWindow;
            fprintf('--------- KNNemg: %d ---- PROB:%4.2f%% -----\n',kNNemg, probabilidadkNNUmbral*100);
            while kAux<numExecutionsTimer
                
                tClassification=tic;
                ventanaEMG(idxUser, testSample, samplesWindow, strideSamples);
                %% ############# Recibiendo la señal emg a la variable ventana #############################################################
                
                %[shiftSamples,~]=size(emg);% cuantas filas del trozo de EMG que voy a trabjar, 200?
                %samplesObtained=samplesObtained+shiftSamples;%total de samples, 200 + 200 + 200 ...
                
                %preparo la matriz window 200x8 para almacenar las muestras
                %unknownGesture = circshift(unknownGesture,-shiftSamples);
                %unknownGesture(end-shiftSamples+1:end,:) = emg;
                
                emg = filtfilt(Fb, Fa,abs(unknownGesture));
                
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
                gestoString=char(nameGestures{gestoResultKNN});
                
                
                %% Umbral y filtro
                gesto = 0;
                if probGestureKNN>=probabilidadkNNUmbral
                    fprintf('%s_ts-%d. w-%d. %s,          %4.2f %%...\n', nameUser, testSample, kExecutions,gestoString , probGestureKNN*100);
                    resultadosGestoVectorEMG(testSample, kExecutions)=gestoResultKNN;
                    VectorParaModa(1, kExecutions)=gestoResultKNN;
                    probabilidadVector(testSample,kExecutions)=probGestureKNN*100;
                else
                    fprintf('EMG noGesto-noUmbral %4.2f %%...\n',probGestureKNN);
                    gestoResultKNN=1;
                end
                
                
                % Cont de num de aciertos usando criterio de votacion a la moda
                if kExecutions==numExecutionsTimer
                    %quito los relax
                    VectorParaModa(VectorParaModa == 1) = [];
                    VectorParaModa(VectorParaModa == 0) = [];
                    if isnan(mode(VectorParaModa)) || (mode(VectorParaModa(1,:))==0)
                        resultadosGestoVectorModa(testSample, 1) = 1;
                        predictedLabels{testSample,1} = char(nameGestures{1});
                        if 1==ceil(testSample/numTry)
                            numAciertos=numAciertos+1;
                            fprintf('\nrelax\n');
                        end
                    else
                        resultadosGestoVectorModa(testSample, 1) = mode(VectorParaModa(1,:));
                        predictedLabels{testSample,1} = char(nameGestures{mode(VectorParaModa(1,:))});
                    end
                    
                    %cuento los correctos, correspondientes a sus 15
                    if mode(VectorParaModa(1,:))==ceil(testSample/numTry)
                        numAciertos=numAciertos+1;
                    end
                end
                
                tClassificacionVector(testSample,kExecutions) = toc(tClassification);
                
                kExecutions=kExecutions+1;
            end
            fprintf('\n');
            
        end
        
        beep
        
        testSamples=(testSampleFinal-testSampleInicio)+1;
        acc=(numAciertos*100)/testSamples;
        fprintf('%s ACC %d:%d test-samples: %4.2f %%\nNum. Aciertos: %d\n', nameUser, testSampleInicio, testSampleFinal, acc, numAciertos);
        fprintf('KNNemg: %d Prob:%4.2f%% -----\n',kNNemg, probabilidadkNNUmbral*100);
        fprintf('samplesWindows:%d, TimeShift:%d\n', samplesWindow, strideSamples);
        %guardo acc de knn y umbral
        %resultadosHiperParametrosEMG(kNNemg,fix(probabilidadkNNUmbral*10))=acc;
        resultadosHiperParametrosEMG(i,b)=acc;
        b=b+1;
    end
end