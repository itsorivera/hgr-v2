clear all
clc

addpath(genpath(pwd));

testing = struct;
users =[21, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23];
resultsUsers=zeros(length(users),1);
for u=1:length(users)
    %idxUser = 1;
    
    idxUser=users(u);
    userDataTest = loadUser(idxUser, 'testing', '.\DATASET_85\');
    nameUser = userDataTest.userInfo.username;

    nameAllGestures={'relax';'waveIn';'waveOut';'fist';'open';'pinch';'up';'down';'left';'right';'forward';'backward'};
    numTry=15;

    nameDynamicGestures={'relax';'up';'down';'left';'right';'forward';'backward'};
    numMoves=7;
    datasetIMUs=datasetConstructionIMU(nameUser,idxUser,numTry,numMoves);

    nameStaticGestures={'relax';'waveIn';'waveOut';'fist';'open';'pinch'};
    numGestures=6;
    %-----
    ordenFiltro=4;
    freqFiltro=0.05;
    [Fb, Fa] = butter(ordenFiltro, freqFiltro, 'low'); % creating filter
    %-----
    datasetGestures=datasetConstructionEMG(nameUser,idxUser,Fb,Fa,numTry,numGestures);
    
    

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

    %-------------------E   M   G---------------------------------------------
    %tam de ventana para EMG
    windowTimeEMG=2.5; % time of the shift window, 500 samples
    windowSamplesEMG = round(windowTimeEMG*freqEMG);

    %compas EMG
    strideSamplesEMG=1*freqEMG;       % samples del paso, stride
    strideTimeEMG=strideSamplesEMG/freqEMG;          % tiempo del paso del stride
    %strideTimeEMG =1;
    numExecutionsTimer = ceil(timeSeries/strideTimeEMG);

    %-------------------I   M   U---------------------------------------------
    %tam de ventana para IMU
    windowTimeIMU=2.4;
    windowSamplesIMU=round(windowTimeIMU*freqIMU);

    %compas IMU
    strideSamplesIMU=50;
    strideTimeIMU=strideSamplesIMU/freqIMU;          % tiempo del paso del stride
    numExecutionsTimerIMU = ceil(timeSeries/strideTimeIMU);

    %-------------------R   E   S   U   L   T   S------------------------------
    resultsMatrix=zeros(12*numTry,numExecutionsTimer);
    resultsModeMatrix=zeros(12*numTry,1);
    resultsLabels=categorical(12*numTry,1);
    resultsTime=zeros(12*numTry,numExecutionsTimer);
    %resultsTimePoints=zeros(12*numTry,numExecutionsTimer);
    %resultVectorOfLabels=categorical(12*numTry,numExecutionsTimer);
    %--------------------------------------------------------------------------
    %For user
    tso=34;
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
            fprintf('ts: %d w-%d. >>> %s gesture ',testSample, kExecutions, ypred);

            if strcmp(ypred,'d')
                [proccesingTime, gestoResultKNN] = classifierIMU(quat, datasetIMUs);
                resultsMatrix(testSample,kExecutions)=gestoResultKNN;
                resultsTime(testSample, kExecutions)=proccesingTime;
                resultVectorOfLabels(1, kExecutions)=char(nameAllGestures{gestoResultKNN});
                VectorParaModa(1, kExecutions)=gestoResultKNN;
            else
                [proccesingTime, gestoResultKNN] = classifierEMG(emg, datasetGestures);
                resultsMatrix(testSample,kExecutions)=gestoResultKNN;
                resultsTime(testSample, kExecutions)=proccesingTime;
                resultVectorOfLabels(1, kExecutions)=char(nameAllGestures{gestoResultKNN});
                VectorParaModa(1, kExecutions)=gestoResultKNN;
            end

            if kExecutions==numExecutionsTimer
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
        fprintf('\n');
        testing.(nameUser).class{testSample,1} = resultsLabels(testSample,1);
        testing.(nameUser).vectorOfTimePoints(testSample,1) = {resultsTimePoints};
        testing.(nameUser).vectorOfProcessingTime(testSample,1) = {vectorProccesingTime};
        testing.(nameUser).vectorOfLabels(testSample,1) = {resultVectorOfLabels};
    end
    % Results all samples
    beep
    testSamples=(tsf-tso)+1;
    acc=(nClassifHits*100)/testSamples;
    fprintf('%s, test-samples: %d:%d \nACC for classification: %4.2f %%\nNum. Aciertos: %d\n', nameUser, tso, tsf, acc, nClassifHits);
    resultsUsers(u,1)=acc;
    % Save format results
     
end

%% energy function
function e = energy(signal)
%signal es una matriz que puede er de n*8 [EMG] o m*4 [IMU] 
    e = sum(abs((signal(2:end,:).*abs(signal(2:end,:))) - (signal(1:end-1,:).*abs(signal(1:end-1,:)))));
end