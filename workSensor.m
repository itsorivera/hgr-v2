addpath(genpath(pwd));
%compas EMG
freqEMG=200;
timeSeries=30;
strideSamplesEMG=200;                      % 200 samples del paso, stride
strideTimeEMG=strideSamplesEMG/freqEMG;          % 1   segundo del paso del stride
numExecutionsTimer = ceil(timeSeries/strideTimeEMG);
load switchCompleteSignal.mat


% Connecting MYO
global myoObject kAux ypred
isConnected =  connectMyo; % isConnected is a flag, connectMyo is a function that connects the MYO
drawnow



kExecutions=1;
%% Recognition loop
leido=0;    % flag to know when new data is ready
kAux=0;     % timer loops counter


uiwait(msgbox('PLEASE, PRESS THE BUTTON TO START.','Instructions','modal'));

% setting timer
strideTimeEMG=strideTimeEMG;
tmr = timer('ExecutionMode','fixedRate', ...
    'TasksToExecute',numExecutionsTimer,...
    'TimerFcn',@(~,~)myoTimerFunction(model), ...
    'StartDelay',strideTimeEMG, ...
    'Period',strideTimeEMG);
myoObject.myoData.clearLogs();
start(tmr)

while kAux<numExecutionsTimer
    fprintf('kAux: %d, gesture type:%s \n',kAux, ypred);

    %     if kExecutions>1
    %         % filtro a la salida. Compara que el gesto sea igual al
    %         % resultante anterior. El gesto resultante anterior es
    %         % aquel sin cosiderar el umbral.
    %         if gestosKNNVector(kExecutions-1)==gestoResultKNN
    %             gesto=gestoResultKNN;
    %         end
    %
    %
    %         % cambio de estado
    %         if resultadosGestoVector(kExecutions-1)==gesto % resultado debe ser igual al gesto KNN anterior
    %             flagFilterGesture=gesto;
    %             gesto=0; % filtro
    %         elseif flagFilterGesture==gestoResultKNN % si el gesto de KNN es respuesta del filtro
    %             %                     elseif flagFilterGesture==gesto % si el gesto resultante es el valor del filtro
    %             gesto=0;
    %         end
    %     end
    kExecutions=kExecutions+1;
end



isConnected=terminateMyo;