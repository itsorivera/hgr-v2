% con umbral de actividad de quaterniones para gestos estaticos
function [datasetGestures, umbralMax]=datasetConstructionEMG_gforce(nameUser,Fb,Fa,numTry,numGestures, userDataTest)

datasetGestures=cell(numGestures*numTry,8);

% Variables for limit calculation
umbralMax=zeros(numTry*numGestures,1);



for j=1:180
    signal=userDataTest.training{j}.emg;
    sumaPerChSignal=sum(signal,1);
    meanSumPerChSignal=mean(sumaPerChSignal);
    for k=1:8
        datasetEMG_gforce{j,k}=signal(:,k) + meanSumPerChSignal;
    end
end



%idG=1;%indice inicial de las muestras gestures
% for i=1:numGestures
%     % Loop per gesture
%     
%     idGf=idG+15-1;%indice final de muestra gesture
%     for j=idG:idGf
%         % Loop per number of repetitions
%         signal=userDataTest.training{j}.emg;
%         %% building dataset
%         for k=1:8
%             % Loop per channel
%             unknownGesture = filtfilt(Fb, Fa,abs(signal(:,k)) ); % filtered absolute value
%             datasetGestures{(i-1)*numTry+(j-idG+1),k}=unknownGesture;
% 
%         end
%     end
%     idG=idG+15;
% end
save (['usersData_v2\' nameUser  'DatasetEMG.mat'],'datasetGestures')
end