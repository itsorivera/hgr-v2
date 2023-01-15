function imux = ventanaIMU(idxUser, testSampleIMU, samplesWindowIMU, timeShiftIMU)

global  leido kAux imu soIMU sfIMU 

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
    imu = numericVectorQ;
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
    imu = numericVectorQ;
    imux = numericVectorQ;
end

    leido=1;
    kAux=kAux+1;
end