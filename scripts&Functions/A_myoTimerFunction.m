function A_myoTimerFunction()
% timer function for reading MYO. Data is transfered as global variables

global  emg  leido myoObject kAux

emg= myoObject.myoData.emg_log;% coge un trozo de señal EMG, establecer el trozo de 200 samples
myoObject.myoData.clearLogs();
leido=1;
kAux=kAux+1;
end