% con umbral de actividad de quaterniones para gestos estaticos
function [datasetGestures, umbralMax]=datasetConstructionEMG(nameUser,Fb,Fa,numTry,numGestures, userDataTest)


%% Predefined values DATASET CONSTRUCTION MODEL V2
%fprintf('\nConstruyendo dataset (datasetConstructionEMG)...\n')
%con defaultFlag=1 los valores estan ya predefinidos

%userData = loadUser(idxUser, 'training', '.\DATASET_85\');
datasetGestures=cell(numGestures*numTry,8);

% Variables for limit calculation
umbralMax=zeros(numTry*numGestures,1);
timeShift=40; % so it can be simulated the shift window. (0.2 sec)


idG=1;%indice inicial de las muestras gestures
for i=1:numGestures
    % Loop per gesture
    
    idGf=idG+15-1;%indice final de muestra gesture
    for j=idG:idGf
        % Loop per number of repetitions
        signal=userDataTest.training{j}.emg;
        [samples,~]=size(signal);
        
        %% Algorithm for gesture detection on signals recorded
        diffSignal=abs([zeros(1,8);diff(signal)])/8; % derivation
        
        % Mean values per window
        for w=1:timeShift:samples-timeShift
            signalf(w:w+timeShift,:)= mean(sum(diffSignal(w:w+timeShift,:),2)); % mean values
        end        
        signalf(w:w+timeShift,:)= mean(sum(diffSignal(w:w+timeShift,:),2)); % mean values last vector part         
        
        % Getting max values
        %umbralMax((i-1)*numTry+j)=max(signalf);
        umbralMax((i-1)*numTry+(j-idG+1))=max(signalf);
        
        %% building dataset
        for k=1:8
            % Loop per channel
            unknownGesture = filtfilt(Fb, Fa,abs(signal(:,k)) ); % filtered absolute value
            datasetGestures{(i-1)*numTry+(j-idG+1),k}=unknownGesture;
        end
    end
    idG=idG+15;
    %fprintf("final gesture %d \n",idG);
end

save (['usersData_v2\' nameUser  'DatasetEMG.mat'],'datasetGestures')
%save (['usersData_v2\' nameUser  'DataSetMoves.mat'],'dataset')
%beep
end