function myoTimerFunction(model)

% timer function for reading MYO. Data is transfered as global variables

global leido myoObject kAux rmsValues ypred

emg= myoObject.myoData.emg_log;% coge un trozo de señal EMG, establecer el trozo de 200 samples
quat = myoObject.myoData.quat_log;

prepEmg = preprocessingSignal(emg);
emgRms = energy(prepEmg);

quatRms = energy(quat);
rmsValues = [emgRms quatRms];

ypred = predict(model,rmsValues);


myoObject.myoData.clearLogs();
leido=1;
kAux=kAux+1;
end

%% energy function
function e = energy(signal)
%signal es una matriz que puede er de n*8 [EMG] o m*4 [IMU]
e = sum(abs((signal(2:end,:).*abs(signal(2:end,:))) - (signal(1:end-1,:).*abs(signal(1:end-1,:)))));
end