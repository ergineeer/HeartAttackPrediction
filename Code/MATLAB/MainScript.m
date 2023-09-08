% Housekeeping
cd(fileparts(matlab.desktop.editor.getActiveFilename))
clear
clc

% Set Random Number Generator Seed for repeatability
rng(1);

% Import data into workspace
heartData = importDataset("Dataset/heart.csv");
head(heartData)
