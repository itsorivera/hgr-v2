function emg = preprocessingSignal(emg)
    
    unknownGesture=zeros(1000,8);
    unknownGesture = emg;
    ordenFiltro=4;
    freqFiltro=0.05;
    [Fb, Fa] = butter(ordenFiltro, freqFiltro, 'low'); % creating filter
    emg = filtfilt(Fb, Fa, abs(unknownGesture));
end