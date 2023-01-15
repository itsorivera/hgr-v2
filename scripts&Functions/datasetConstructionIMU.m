% con umbral de actividad de quaterniones para gestos estaticos
function [datasetIMU]=datasetConstructionIMU(nameUser,numTry,numMoves, userDataTest)


%% Predefined values DATASET CONSTRUCTION MODEL V2
%fprintf('\nConstruyendo dataset (datasetConstructionIMU)...\n')
datasetIMU=cell(numMoves*numTry,4);
%userData = loadUser(idxUser, 'training', '.\DATASET_85\');
%% DATASET MOVES
idx=1;
flag=1;
for i=1:numMoves
    idxf=idx+15-1;%indice final de muestra move
    for j=idx:idxf
        signal=userDataTest.training{j}.quaternions;
        %% building dataset
        %signal=(signal+1)/2;
        for k=1:4
            datasetIMU{(i-1)*numTry+(j-idx+1),k}=signal(:,k);
        end
    end
    idx=idx+15;%actualizacion para el siguiente indice inicial del move
    %fprintf("final %d \n",idx);
        if flag==1
        idx=idx+75;
        flag=0;
        end
end
save (['usersData_v2\' nameUser  'DatasetIMU.mat'],'datasetIMU')
end