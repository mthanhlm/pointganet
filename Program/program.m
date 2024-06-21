clc
%%Load DATA
dataFolder = fullfile('D:\Study\PointGANet\DALES\Dataset\DALES');
trainDataFolder = fullfile(dataFolder,'dales_las','train');
testDataFolder = fullfile(dataFolder,'dales_las','test');

lasReader = lasFileReader(fullfile(trainDataFolder,'5080_54435.las'));
[pc,attr] = readPointCloud(lasReader,'Attributes','Classification');
labels = attr.Classification;

% Select only labeled data.
pc = select(pc,labels~=0);
labels = labels(labels~=0);
classNames = [
    "ground"
    "vegetation"
    "cars"    "trucks"
    "powerlines"
    "fences"
    "poles"
    "buildings"
    ];

%% Preprocess Data
blocksize = [51 51 Inf];
fs = matlab.io.datastore.FileSet(trainDataFolder);
bpc = blockedPointCloud(fs,blocksize);
numClasses = numel(classNames);
[weights,maxLabel,maxWeight] = helperCalculateClassWeights(fs,numClasses);
%% Create Datastore Object for Training
ldsTrain = blockedPointCloudDatastore(bpc);
labelIDs = 1 : numClasses;
numPoints = 8192;
ldsTransformed = transform(ldsTrain,@(x,info) helperTransformToTrainData(x, ...
    numPoints,info,labelIDs,classNames),'IncludeInfo',true);
read(ldsTransformed)

%% Define Model
load('multiply_en4_de16.mat')
lgraph = multiply_en4_de16;
% Replace the FocalLoss layer with pixelClassificationLayf
% larray = pixelClassificationLayer('Name','SegmentationLayer','ClassWeights', ...
%     weights,'Classes',classNames);
% lgraph = replaceLayer(lgraph,'FocalLoss',larray);
%analyzeNetwork(lgraph)

%% Training options
learningRate = 0.0005;
l2Regularization = 0.01;
numEpochs = 20;
miniBatchSize = 64;
learnRateDropFactor = 0.1;
learnRateDropPeriod = 10;
gradientDecayFactor = 0.9;
squaredGradientDecayFactor = 0.999;

options = trainingOptions('adam', ...
    'InitialLearnRate',learningRate, ...
    'L2Regularization',l2Regularization, ...
    'MaxEpochs',numEpochs, ...
    'MiniBatchSize',miniBatchSize, ...
    'LearnRateSchedule','piecewise', ...
    'LearnRateDropFactor',learnRateDropFactor, ...
    'LearnRateDropPeriod',learnRateDropPeriod, ...
    'GradientDecayFactor',gradientDecayFactor, ...
    'SquaredGradientDecayFactor',squaredGradientDecayFactor, ...
    'ExecutionEnvironment','gpu');

%% Train Model

doTraining = false;
if doTraining
    % Train the network on the ldsTransformed datastore using 
    % the trainNetwork function.
    [net,info] = trainNetwork(ldsTransformed,lgraph,options);
else
    % Load the pretrained network.
    load('pretrained_pointganet.mat','trainednetInfo');
    net = trainednetInfo{1};
end

%% Segment Aerial Point Cloud

tbpc = blockedPointCloud(fullfile(testDataFolder,'5100_54490.las'),blocksize);
tbpcds = blockedPointCloudDatastore(tbpc);
numNearestNeighbors = 20;
radius = 0.05;
labelsDensePred = [];
labelsDenseTarget = [];
while hasdata(tbpcds)
    
    % Read the block along with block information.
    [ptCloudDense,infoDense] = read(tbpcds);

    % Extract the labels from the block information.
    labelsDense = infoDense.PointAttributes.Classification;
    
    % Select only labeled data.
    ptCloudDense = select(ptCloudDense{1},labelsDense~=0);
    labelsDense = labelsDense(labelsDense~=0);

    % Use the helperDownsamplePoints function, attached to this example as a
    % supporting file, to extract a downsampled point cloud from the
    % dense point cloud.
    ptCloudSparse = helperDownsamplePoints(ptCloudDense, ...
        labelsDense,numPoints);

    % Make the spatial extent of the dense point cloud and the sparse point
    % cloud same.
    limits = [ptCloudDense.XLimits;ptCloudDense.YLimits;ptCloudDense.ZLimits];
    ptCloudSparseLocation = ptCloudSparse.Location;
    ptCloudSparseLocation(1:2,:) = limits(:,1:2)';
    ptCloudSparse = pointCloud(ptCloudSparseLocation,'Color',ptCloudSparse.Color, ...
        'Intensity',ptCloudSparse.Intensity, ...
        'Normal',ptCloudSparse.Normal);

    % Use the helperNormalizePointCloud function, attached to this example as
    % a supporting file, to normalize the point cloud between 0 and 1.
    ptCloudSparseNormalized = helperNormalizePointCloud(ptCloudSparse);
    ptCloudDenseNormalized = helperNormalizePointCloud(ptCloudDense);

    % Use the helperTransformToTestData function, defined at the end of this
    % example, to convert the point cloud to a cell array and to permute the
    % dimensions of the point cloud to make it compatible with the input layer
    % of the network.
    ptCloudSparseForPrediction = helperTransformToTestData(ptCloudSparseNormalized);

    % Get the output predictions.
    labelsSparsePred = pcsemanticseg(ptCloudSparseForPrediction{1,1}, ...
        net,'OutputType','uint8');

    % Use the helperInterpolate function, attached to this example as a
    % supporting file, to calculate labels for the dense point cloud,
    % using the sparse point cloud and labels predicted on the sparse point cloud.
    interpolatedLabels = helperInterpolate(ptCloudDenseNormalized, ...
        ptCloudSparseNormalized,labelsSparsePred,numNearestNeighbors, ...
        radius,maxLabel,numClasses);

    % Concatenate the predicted and target labels from the blocks.
    labelsDensePred = vertcat(labelsDensePred,interpolatedLabels);
    labelsDenseTarget = vertcat(labelsDenseTarget,labelsDense);
end

%% Evaluate Network

confusionMatrix = segmentationConfusionMatrix(double(labelsDensePred), ...
    double(labelsDenseTarget),'Classes',1:numClasses);
metrics = evaluateSemanticSegmentation({confusionMatrix},classNames,'Verbose',true);

metrics.DataSetMetrics
metrics.ClassMetrics
if doTraining
    trainednetInfo{1}=net;
    trainednetInfo{2}=info;
    trainednetInfo{3}=options;
    trainednetInfo{4}=metrics;
    avoidOverwrite('D:\Study\PointGANet\Save_info\DALES','dales_train_info.mat','trainednetInfo')
    avoidOverwrite('D:\Study\PointGANet\Save_net\DALES','pointganet_pretrained.mat','net')
else
    disp('Error to save file!')
end
