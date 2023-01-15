%% getting users data
%userList = [1 24 25 27 32 33 35 37 39 40 41 42 26 34 3 5 7 10 13 15 17 23 30 28 4 6 8 9 11 12 14 16 18 19 20 21 22 29 31];
userList = [1 3];
window = 1.5;
numGuesses = length(userList) * (165); %165 from 11 gestures * 15 repetitions
guessMatrix = [];
labels = [];
testingGuessMat = [];
testingLabels = [];

for idx = 1:length(userList)
    user = userList(idx);
    %userData = loadUser(user,"training",fullfile("DATASET_85","DATASET_85"));
    userData = loadUser(user, 'training', '.\DATASET_85\');
    samplingF = userData.deviceInfo.emgSamplingRate;
    factorConversion = 50/ samplingF;
    reps = getAllRepetitions(userData);
    
    trainingRep = reps.training;
    datosEntrenamiento = prepareData(trainingRep,samplingF);
    guessMatrix = [guessMatrix ; datosEntrenamiento.guesses];
    labels = [labels ; datosEntrenamiento.labels];

    testingRep = reps.testing;
    datosTesting = prepareData(testingRep, samplingF);
    testingGuessMat = [testingGuessMat; datosTesting.guesses];
    testingLabels = [testingLabels; datosTesting.labels];
end
% randomize matrices
seed = randperm(length(guessMatrix));
guessMatrixR = guessMatrix(seed,:);
labelsR = labels(seed,:);

seed2 = randperm(length(testingGuessMat));
testingGuessMatR = testingGuessMat(seed2,:);
testingLabelsR = testingLabels(seed2,:);


%% creating the model
model = fitclinear(guessMatrixR,labelsR,'Learner','logistic');
save MdlSwitch model
%load('switch.mat');

%% testing 
%total errors
errTrain = loss(model, guessMatrix, labels);
errTest = loss(model, testingGuessMat, testingLabels);
%error for each user
userIndices = 1:165:length(labels);
errTraineUser = [];
errTesteUser = [];
for j = 1:length(userIndices)
    inicio = userIndices(j);
    finish = inicio + 164;
    errTraineUser = [errTraineUser ; loss(model, guessMatrix(inicio:finish,:), labels(inicio:finish,:))];
    errTesteUser = [errTesteUser ; loss(model, testingGuessMat(inicio:finish,:), testingLabels(inicio:finish,:))];
end

%compute validation values
negation = ones(length(errTesteUser),1);
trainingClassf = (negation - errTraineUser) *100;
testingClassf = (negation - errTesteUser) *100;
trainClassfTotal = (1 - errTrain)*100;
testingClassfTotal = (1 - errTest)*100;
results = table(transpose(userList), trainingClassf, testingClassf);
%% Function to prepare data
function data = prepareData(repetitionData, samplingF)
    numGestureRepetitions=15;
    factorConversion = 50/ samplingF;
    window = 1.5;
    data = struct;
    guessMat = zeros(length(repetitionData)-numGestureRepetitions, 12);
    labels = cell(length(repetitionData)-numGestureRepetitions, 1);
    for i = 1:length(repetitionData)-numGestureRepetitions
        rep = repetitionData{i+numGestureRepetitions,1};%empiezo desde el la muestra 16
        %saco y proceso EMG de muestra training 16
        emg = preprocessingSignal(rep.emg);
        labels{i,1} = convertStringsToChars(string(rep.gestureName));
        %saco IMU de muestra training 16
        quat = rep.quaternions;
        %inicio del groundtruth
        inicio = rep.groundTruthIndex(1);
        %compute the start and finish indices for the emg window and quat
        %window 
        emgFinishIdx = inicio+window*samplingF;
        qStart = ceil(inicio*factorConversion);
        qFinish = floor(inicio*factorConversion)+75;

        if emgFinishIdx > length(emg) || qFinish > length(quat)
            emgRms = energy(emg(inicio:end,:));
            quatRms = energy(quat(qStart:end,:));
        else
            emgRms = energy(emg(inicio:emgFinishIdx,:));
            quatRms = energy(quat(qStart:qFinish,:));
        end
        rmsValues = [emgRms quatRms];
        guessMat(i,:) = rmsValues;
    end
    cats = categorical(labels,{'waveIn' 'waveOut' 'open' 'pinch' 'fist' 'forward' 'backward' 'up' 'down' 'left' 'right'}, {'s' 's' 's' 's' 's' 'd' 'd' 'd' 'd' 'd' 'd'});
    data.guesses = guessMat;
    data.labels = cats;
end

%% energy function
function e = energy(signal)
%signal es una matriz que puede er de n*8 [EMG] o m*4 [IMU] 
    e = sum(abs((signal(2:end,:).*abs(signal(2:end,:))) - (signal(1:end-1,:).*abs(signal(1:end-1,:)))));
end