% Load MNIST Training Data
data_images = loadMNISTImages('data/train-images.idx3-ubyte');
data_labels = loadMNISTLabels('data/train-labels.idx1-ubyte');

data_images = transpose(data_images);

size_training_set = 60000;
size_testing_set = 20000;

boRandomizeTrainingSet = 0;

% Create training set and testing set
X_train = zeros(size_training_set, size(data_images, 2));
Y_train = zeros(size_training_set, 10);

if boRandomizeTrainingSet == 1
    for i=1:size_training_set

        index = rand(1) * size(data_images, 1);
        index = int32(index);

        X_train(i, :) = data_images(index, :);
        Y_train(i, int32(data_labels(index)) + 1) = 1;

    end
else
    for i=1:size_training_set

        X_train(i, :) = data_images(i, :);
        Y_train(i, int32(data_labels(i)) + 1) = 1;

    end
end

X_test = zeros(size_testing_set, size(data_images, 2));
Y_test_digit = zeros(size_testing_set, 1);

for i=1:size_testing_set
    
    index = rand(1) * size(data_images, 1);
    index = int32(index);
    
    X_test(i, :) = data_images(index, :);
    Y_test_digit(i) = data_labels(index);
    
end

% Define Neural Network structure
NNStructDefinition.nbLayers                     = 3;

NNStructDefinition.layers                       = [];

NNStructDefinition.layers{1}.type               = 'input';
NNStructDefinition.layers{1}.nbNeurons          = 784;

NNStructDefinition.layers{2}.type               = 'hidden';
NNStructDefinition.layers{2}.activation_type    = 'relu';
NNStructDefinition.layers{2}.nbNeurons          = 392;

NNStructDefinition.layers{3}.type               = 'output';
NNStructDefinition.layers{3}.activation_type    = 'sigmoid';
NNStructDefinition.layers{3}.nbNeurons          = 10;

NNStructDefinition.learning_rate                = 0.1;

% Train NN (the last argument is the maximum number of epochs to be run)
[ W, B ] = NN_train( NNStructDefinition, X_train, Y_train, 50 );

% Predictions based on input and trained weights and biases
[ Y_pred ] = NN_predict(NNStructDefinition, X_test, W, B);

Y_pred_digit = zeros(size(Y_pred, 1), 1);
for i=1:size(Y_pred, 1)
    [~, indices] = max(Y_pred(i, :));
    Y_pred_digit(i) = indices(1) - 1;
end

nb_errors = 0;
for i=1:size(Y_pred_digit, 1)
    if Y_pred_digit(i) ~= Y_test_digit(i)
        nb_errors = nb_errors + 1;
    end
end

fprintf('%f percent of errors\n', (nb_errors / size_testing_set) * 100);
fprintf('Accuracy :', 1 - (nb_errors / size_testing_set));
    
