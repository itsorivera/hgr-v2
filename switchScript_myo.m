% este script es para guardar los resultados con el formato del inge jtzea.
% Sirve para gforce como para myo

clear all
clc

addpath(genpath(pwd));

response = struct;
users =[25,0];
resultsUsers=zeros(length(users),1);
for u=1:length(users)
    
    idxUser=users(u);
    userDataTest = loadUser(idxUser, 'training', '.\DATASET_85\');
    nameUser = userDataTest.userInfo.username;

    nameAllGestures={'relax';'waveIn';'waveOut';'fist';'open';'pinch';'up';'down';'left';'right';'forward';'backward'};
    numTry=15;

    nameDynamicGestures={'relax';'up';'down';'left';'right';'forward';'backward'};
    numMoves=7;
    datasetIMUs=datasetConstructionIMU(nameUser,numTry,numMoves, userDataTest);

    nameStaticGestures={'relax';'waveIn';'waveOut';'fist';'open';'pinch'};
    numGestures=6;
    %-----
    ordenFiltro=4;
    freqFiltro=0.05;
    [Fb, Fa] = butter(ordenFiltro, freqFiltro, 'low'); % creating filter
    %-----
    datasetGestures=datasetConstructionEMG(nameUser,Fb,Fa,numTry,numGestures, userDataTest);
    
    

    %aaa = struct;
    load MdlSwitch.mat
    %load switchCompleteSignal.mat

    if isempty(gcp)
        parpool;
        beep
    end

    timeSeries = 5; %time to run the script
    freqEMG = userDataTest.deviceInfo.emgSamplingRate;
    freqIMU=50;
    
    if freqEMG==200
        %% -------------------M   Y   O--------------------------------------
        %-------------------E   M   G---------------------------------------------
        %tam de ventana para EMG
        windowTimeEMG=2.5; % time of the shift window, 500 samples
        %windowTimeEMG=3.4;
        windowSamplesEMG = round(windowTimeEMG*freqEMG);

        %compas EMG
        strideSamplesEMG=1*freqEMG;       % samples del paso, stride
        strideTimeEMG=strideSamplesEMG/freqEMG;          % tiempo del paso del stride
        %strideTimeEMG =1;
        numExecutionsTimer = ceil(timeSeries/strideTimeEMG);
        
        %hiper param knn
        kNNemg=7;
        probabilidadkNNUmbral=0.8;

        %-------------------I   M   U---------------------------------------------
        %tam de ventana para IMU
        windowTimeIMU=3.3;
        windowSamplesIMU=round(windowTimeIMU*freqIMU);

        %compas IMU
        strideSamplesIMU=50;
        strideTimeIMU=strideSamplesIMU/freqIMU;          % tiempo del paso del stride
        numExecutionsTimerIMU = ceil(timeSeries/strideTimeIMU);

        %hiper param knn
        kNNimu=7;
        thresholdkNNimu=0.8;
    else
        %% -------------------G F O R C E--------------------------------------
        stride=1.1;
        %-------------------E   M   G---------------------------------------------
        %tam de ventana para EMG
        windowTimeEMG=2.5; % time of the shift window, 500 samples
        windowSamplesEMG = round(windowTimeEMG*freqEMG);

        %compas EMG
        strideTimeEMG=stride;         % tiempo del paso del stride
        strideSamplesEMG=strideTimeEMG*freqEMG;                     % samples del paso, stride
        %strideTimeEMG =1;
        numExecutionsTimer = ceil(timeSeries/strideTimeEMG);
        %hiper param knn
        kNNemg=7;
        probabilidadkNNUmbral=0.8;
        %-------------------I   M   U---------------------------------------------
        %tam de ventana para IMU
        windowTimeIMU=2.4;
        windowSamplesIMU=round(windowTimeIMU*freqIMU);

        %compas IMU
        strideTimeIMU=stride;          % tiempo del paso del stride
        strideSamplesIMU=strideTimeIMU*freqIMU;
        numExecutionsTimerIMU = ceil(timeSeries/strideTimeIMU);
        %hiper param knn
        kNNimu=7;
        thresholdkNNimu=0.8;
    end


    %-------------------R   E   S   U   L   T   S------------------------------
    resultsMatrix=zeros(12*numTry,numExecutionsTimer);
    resultsModeMatrix=zeros(12*numTry,1);
    resultsLabels=categorical(12*numTry,1);
    resultsTime=zeros(12*numTry,numExecutionsTimer);
    %posprocesam
    MatrixVectorPosPro=zeros(12*numTry, numExecutionsTimer);
    MatrixVectorPosProLabels=categorical(12*numTry, numExecutionsTimer);
    %resultsTimePoints=zeros(12*numTry,numExecutionsTimer);
    %resultVectorOfLabels=categorical(12*numTry,numExecutionsTimer);
    %--------------------------------------------------------------------------
    %For user
    tso=1;
    tsf=180;
    nClassifHits=0;

    %--------------------------------------------------------------------------
    %For sample
    for testSample=tso:tsf
        %testSample = 119;

        kExecutions=1;
        VectorParaModa=zeros(1,numExecutionsTimer);
        resultsTimePoints=zeros(1,numExecutionsTimer);
        vectorProccesingTime=zeros(1,numExecutionsTimer);
        resultVectorOfLabels=categorical(1,numExecutionsTimer);
        resultVectorOfLabelsPosPro=categorical(1,numExecutionsTimer);

        soEMG=1;
        sfEMG=windowSamplesEMG;

        soIMU=1;
        sfIMU=windowSamplesIMU;


        for i=1:numExecutionsTimer

            try
                quat = userDataTest.testing{testSample}.quaternions(soIMU:sfIMU,:);

                emg = userDataTest.testing{testSample}.emg(soEMG:sfEMG,:);
                emg = preprocessingSignal(emg);
            catch
                quat = userDataTest.testing{testSample}.quaternions(soIMU:end,:);

                emg = userDataTest.testing{testSample}.emg(soEMG:end,:);
                emg = preprocessingSignal(emg);
            end

            emgRms = energy(emg);
            quatRms = energy(quat);
            rmsValues = [emgRms quatRms];

            ypred = predict(model,rmsValues);
            ypred = string(ypred);
            fprintf('%s, ts: %d w-%d. >>> %s gesture ', nameUser,testSample, kExecutions, ypred);

            if strcmp(ypred,'d')
                [proccesingTime, gestoResultKNN] = classifierIMU(quat, datasetIMUs,kNNimu,thresholdkNNimu);

                resultsMatrix(testSample,kExecutions)=gestoResultKNN;
                resultsTime(testSample, kExecutions)=proccesingTime;
                resultVectorOfLabels(1, kExecutions)=char(nameAllGestures{gestoResultKNN});
                VectorParaModa(1, kExecutions)=gestoResultKNN;
            else
                [proccesingTime, gestoResultKNN] = classifierEMG(emg, datasetGestures, kNNemg,probabilidadkNNUmbral);
                
                resultsMatrix(testSample,kExecutions)=gestoResultKNN;
                resultsTime(testSample, kExecutions)=proccesingTime;
                resultVectorOfLabels(1, kExecutions)=char(nameAllGestures{gestoResultKNN});
                VectorParaModa(1, kExecutions)=gestoResultKNN;
            end
            
            %   posprocesamiento para pulir el % de reconocimiento
            %   limpiar las etiquetas erradas
            %if gestoResultKNN ~= gestosKNNVector(kExecutions-1)  && gestoResultKNN > 1
            %        gestoResultKNN=1;
            %end


            if kExecutions==numExecutionsTimer
                VectorPosPro=VectorParaModa;
                %quito los relax
                VectorParaModa(VectorParaModa == 1) = [];
                VectorParaModa(VectorParaModa == 0) = [];
                if isnan(mode(VectorParaModa)) || (mode(VectorParaModa(1,:))==0)
                    resultsModeMatrix(testSample, 1) = 1;
                    resultsLabels(testSample,1) = char(nameAllGestures{1});
                    if 1==ceil(testSample/numTry)
                        nClassifHits=nClassifHits+1;
                        fprintf('\nrelax\n');
                    end
                else
                    resultsModeMatrix(testSample, 1) = mode(VectorParaModa(1,:));
                    resultsLabels(testSample,1) = char(nameAllGestures{mode(VectorParaModa(1,:))});
                end

                %cuento los correctos, correspondientes a sus 15
                if mode(VectorParaModa(1,:))==ceil(testSample/numTry)
                    nClassifHits=nClassifHits+1;
                end
            end
            
            

            resultsTimePoints(1,kExecutions)=sfEMG;
            soEMG=soEMG+strideSamplesEMG;
            vectorProccesingTime(1, kExecutions) = (soEMG-1)/freqEMG;
            sfEMG=soEMG+windowSamplesEMG-1;

            soIMU=soIMU+strideSamplesIMU;
            sfIMU=soIMU+windowSamplesIMU-1;
            kExecutions=kExecutions+1;

            
        end
        %% pos procesamiento 
        VectorPosPro(VectorPosPro ~= 1 ) = resultsModeMatrix(testSample, 1);
        %VectorPosPro(VectorPosPro ~=  resultsModeMatrix(testSample, 1)) = 1;
        MatrixVectorPosPro(testSample, :) = VectorPosPro;
        for t=1:5
            resultVectorOfLabelsPosPro(1, t) = char(nameAllGestures{MatrixVectorPosPro(testSample,t)});
            MatrixVectorPosProLabels(testSample, t) = char(nameAllGestures{MatrixVectorPosPro(testSample,t)});
        end
        %% 
        fprintf('\n');
        response.testing.(nameUser).class{testSample,1} = resultsLabels(testSample,1);
        response.testing.(nameUser).vectorOfTimePoints(testSample,1) = {resultsTimePoints};
        response.testing.(nameUser).vectorOfProcessingTime(testSample,1) = {vectorProccesingTime};
        response.testing.(nameUser).vectorOfLabels(testSample,1) = {resultVectorOfLabelsPosPro};
    end
    % Results all samples
    beep
    testSamples=(tsf-tso)+1;
    acc=(nClassifHits*100)/testSamples;
    fprintf('%s, test-samples: %d:%d \nACC for classification: %4.2f %%\nNum. Aciertos: %d\n', nameUser, tso, tsf, acc, nClassifHits);
    resultsUsers(u,1)=acc;
    % Save format results
     
end


myCluster = parcluster('local');
delete(myCluster.Jobs)

%% energy function
function e = energy(signal)
%signal es una matriz que puede er de n*8 [EMG] o m*4 [IMU] 
    e = sum(abs((signal(2:end,:).*abs(signal(2:end,:))) - (signal(1:end-1,:).*abs(signal(1:end-1,:)))));
end