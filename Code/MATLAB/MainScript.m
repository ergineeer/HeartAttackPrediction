% % % % % 
% Heart Attack Prediction Study
% Exploratory Data Analysis, Feature Engineering, Binary Classification
% Ensemble Algorithm - Gentle Adaptive Boosting
% 18/10/2024
% 

% Housekeeping
cd(fileparts(matlab.desktop.editor.getActiveFilename))
clear
clc

% Set Random Number Generator Seed for repeatability
rng(1);

% Import data into workspace
heartData = importDataset("Dataset/heart.csv");
head(heartData)


%% Data Analysis
summary(heartData)
size(heartData)
varTypes = varfun(@class, heartData, 'OutputFormat', 'cell')'
missingValues = sum(sum(ismissing(heartData)))


%% Feature Engineering 
heartData.CholToAge = heartData.chol./heartData.age;
gscatter(heartData.age,heartData.CholToAge,heartData.output)
legend(["Female", "Male"]); grid on; xlabel("Age"); ylabel("Cholestrol to Age")

heartData.AgeToMaxRate = heartData.age./heartData.thalachh;
gscatter(heartData.age,heartData.AgeToMaxRate,heartData.output)
legend(["Female", "Male"]); grid on; xlabel("Age"); ylabel("Age to Max Rate")

heartData.MaxRateToResting = heartData.thalachh./heartData.trtbps;
gscatter(heartData.age,heartData.MaxRateToResting,heartData.output)
legend(["Female", "Male"]); grid on; xlabel("Age"); ylabel("Max Rate To Resting Blood Pressure")

heartData.MajorVesselsToRestingBP = double(string(heartData.caa))./heartData.trtbps;
gscatter(heartData.age,heartData.MajorVesselsToRestingBP,heartData.output)
legend(["Female", "Male"]); grid on; xlabel("Age"); ylabel("Resting Blood Pressure To Number of Major Vessels")

heartData.MajorVesselsToMaxRate = double(string(heartData.caa))./heartData.thalachh;
gscatter(heartData.age,heartData.MajorVesselsToMaxRate,heartData.output)
legend(["Female", "Male"]); grid on; xlabel("Age"); ylabel("Max Rate To Number of Major Vessels"); 

heartData.STDepressionToMaxRate = heartData.oldpeak./heartData.thalachh;
gscatter(heartData.age,heartData.STDepressionToMaxRate,heartData.output)
legend(["Female", "Male"]); grid on; xlabel("Age"); ylabel("Resting Blood Pressure To Number of Major Vessels")


%% Datatype Conversion
heartDataIntact = heartData;
heartData = movevars(heartData, 'output', 'After', 'STDepressionToMaxRate');
for i = 1:width(heartData)
    if iscategorical(heartData.(i))
        heartData.(i) = double(heartData.(i));
    end
end


%% Correlation Analysis
heartDataArray = table2array(heartData);
corrCoefMatrix = corrcoef(heartDataArray);
outputCorrelationVector = abs(corrCoefMatrix(:,width(corrCoefMatrix)));
[featImpCorelation, idx] = sortrows(outputCorrelationVector,"descend");

featImpLabelsbyCorrelation = string(heartData.Properties.VariableNames(idx))';
featImpLabelsbyCorrelation(1) = []; % Remove "Output" Column
featImpCorelation(1) = []; % Remove "Output" Column

% Visualize Results
% 1. Corelation Matrix Visualization
figure; hold on
imagesc(corrCoefMatrix)
colormap("Pink")
axis off
hold off

% 2. Output Corelation Array
figure; hold on
title("Correlation")
plot(featImpCorelation)
xticks(1:length(featImpCorelation));
tickLabels = featImpLabelsbyCorrelation;
xticklabels(tickLabels);
grid on


%% Feature Analysis by TreeBagger
heartData_Response = heartData.output;
heartData_Predictors = removevars(heartData, 'output');
heartData_Predictors = table2array(heartData_Predictors);

nTrees = 100;
rf = TreeBagger(nTrees, ...
    heartData_Predictors, heartData_Response, ... 
    'Method', 'classification', ...
    'OOBPrediction', 'on', ...
    'OOBPredictorImportance','on');

% Feature importance
importance = rf.OOBPermutedVarDeltaError;
[featImpTreeBagger,idx] = sortrows(importance',"descend");
fprintf('\nFeature importance:\n');
featImpLabelsbyTreeBagger = strings(length(featImpTreeBagger),1);
for i = 1:size(heartData_Predictors,2)
    ii = idx(i);
    fprintf('%s: %f\n', heartData.Properties.VariableNames{ii}, featImpTreeBagger(i));
    featImpLabelsbyTreeBagger(i,1) = heartData.Properties.VariableNames{ii}; 
end


%% Feature Analysis by Cross-Tabulation
pValues = zeros(length(heartData.Properties.VariableNames)-1,1);
fprintf('\nFeature importance:\n');
for i=1:width(heartData)
    varName = heartData.Properties.VariableNames{i};
    [~, ~, pVal] = crosstab(heartDataIntact.output, heartDataIntact.(varName));
    fprintf("%s - %.4f\n", varName, pVal)
    pValues(i,1) = pVal;
end
featImpLabelsbypVals = string(heartData.Properties.VariableNames)';
[featImp_pValuesOrdered,idx] = sortrows(pValues,"ascend");
featImpLabelsbypVals = featImpLabelsbypVals(idx,:);
 
featImp_pValuesOrdered(1) = []; % Remove "Output" Column
featImpLabelsbypVals(1) = []; 


%% Show Prominent Features
maxLength = max(cellfun(@length, string(heartData.Properties.VariableNames)));

for i=1:15
    if i==1
        inputStr = "                                          Important Features\n";
        inputStr = inputStr + "     Corelation                           Tree Bagger                          Cross-Tabulation\n";
        inputStr = inputStr + "%.2d - %-*s: %.3f   |   %-*s: %.3f   |   %-*s: %.3f\n";
    else
        inputStr = "%.2d - %-*s: %.3f   |   %-*s: %.3f   |   %-*s: %.3f\n";
    end
    fprintf(inputStr,i, maxLength, featImpLabelsbyCorrelation(i,1), featImpCorelation(i,1), ...
        maxLength, featImpLabelsbyTreeBagger(i,1), featImpTreeBagger(i,1), ...
        maxLength, featImpLabelsbypVals(i,1), featImp_pValuesOrdered(i,1));
end

trainingFeatures = intersect(featImpLabelsbyTreeBagger, featImpLabelsbyCorrelation)
trainingFeatures = [trainingFeatures; "output"];

currentColumns = heartDataIntact.Properties.VariableNames;
columnsToDelete = setdiff(currentColumns, trainingFeatures);
heartDataIntact(:, columnsToDelete) = [];


%% Preparing Train and Test Partitions
HoldOutRatio = 0.2;
heartDataPartitions = cvpartition(height(heartDataIntact), "HoldOut", HoldOutRatio)

heartDataTrainingIdx = training(heartDataPartitions);
heartDataTraining = heartDataIntact(heartDataTrainingIdx,:)

heartDataTestIdx = test(heartDataPartitions);
heartDataTest = heartDataIntact(heartDataTestIdx,:)


