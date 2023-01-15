function [imux, soIMU, sfIMU] = ventanaIMUgforce(idxUser, testSampleIMU, samplesWindowIMU, timeShiftIMU, soIMU, sfIMU)

global   kAux

%% Cargo el gesto a clasificar. Move o Gesture
userDataTest = loadUser(idxUser, 'training', '.\DATASET_85\');

%% QUATERNIONS
if kAux>0
    soIMU=soIMU+timeShiftIMU;
    sfIMU=soIMU+samplesWindowIMU-1;
    try
        signalq = userDataTest.testing{testSampleIMU}.quaternions(soIMU:sfIMU,:);
    catch
        signalq = userDataTest.testing{testSampleIMU}.quaternions(soIMU:end,:);
    end
    %signalq=(signalq+1)/2;
    for k=1:4
        unknownGestureQuaternions{:,k}=signalq(:,k);
    end
    numericVectorQ = cell2mat(unknownGestureQuaternions);
    imux = numericVectorQ;
else
    try
        signalq = userDataTest.testing{testSampleIMU}.quaternions(soIMU:sfIMU,:);
    catch
        signalq = userDataTest.testing{testSampleIMU}.quaternions(soIMU:end,:);
    end
    %signalq=(signalq+1)/2;
    for k=1:4
        unknownGestureQuaternions{:,k}=signalq(:,k);
    end
    numericVectorQ = cell2mat(unknownGestureQuaternions);
    imux = numericVectorQ;
end

    kAux=kAux+1;
end