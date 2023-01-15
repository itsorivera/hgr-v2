clear all
clc

timeSeries=30; % tiempo de reconocimiento

timeShiftWindow=0.2;


global leido  myoObject kAux 
isConnected =  connectMyo; % isConnected is a flag, connectMyo is a function that connects the MYO
drawnow

parallelFlag=1; % 0 for not parfor, 1 for parfor in DTW database

numExecutionsTimer=ceil(timeSeries/timeShiftWindow);% 30/0.2=150 cuantas veces la ventana se va a mover


%% Starting parpool
if parallelFlag==1
    fprintf('Por favor espere.\n')
    
    if isempty(gcp)
        parpool;
        beep
    end
    
    fprintf('Listo.\n')
end

kExecutions=1;


if isConnected == 1

    %% Recognition loop
    leido=0;    % flag to know when new data is ready
    kAux=0;     % timer loops counter

    drawnow
    uiwait(msgbox('PLEASE, PRESS THE BUTTON TO START.','Instructions','modal'));

    % setting timer
    timeShiftWindow=timeShiftWindow;
    tmr = timer('ExecutionMode','fixedRate', ...
        'TasksToExecute',numExecutionsTimer,...
        'TimerFcn',@(~,~)A_myoTimerFunction, ...
        'StartDelay',timeShiftWindow, ...
        'Period',timeShiftWindow);
    myoObject.myoData.clearLogs();
    start(tmr)


    while kAux<numExecutionsTimer

        fprintf('kAux: %d, kexecutions:%d\n',kAux, kExecutions);


        kExecutions=kExecutions+1;
    end

end

isConnected=terminateMyo;

