function ventanaEMG(idxUser,testSample, windowSamples, strideSamples)

global  emg  leido kAux so sf
userDataTest = loadUser(idxUser, 'training', '.\DATASET_85\');
%% EMG
if kAux > 0
    so=so+strideSamples;
    sf=so+windowSamples-1;

    try
        signal = userDataTest.testing{testSample}.emg(so:sf,:);
    catch
        signal = userDataTest.testing{testSample}.emg(so:end,:);
    end
    emg = signal;
else
    try
        signal = userDataTest.testing{testSample}.emg(so:sf,:);
    catch
        signal = userDataTest.testing{testSample}.emg(so:end,:);
    end
    emg = signal;
end

leido=1;
kAux=kAux+1;
end