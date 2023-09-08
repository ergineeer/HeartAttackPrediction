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
