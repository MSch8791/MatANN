function [ W, B ] = NN_train( NNStructDefinition, X, Y, max_iteration_number )

	% Train the ANN
    % Takes as inputs :
    %           - NNStructDefinition    --> The neural network architecture definition
    %           - X             		--> The input matrix
	%			- Y						--> The targeted ouput matrix
    %           - max_iteration_number	--> The maximum number of epochs to be run

    %%%% Create the neural network based on the given structure definition
    nbFoundInputLayer  = 0;
    nbFoundOutputLayer = 0;

    W = cell(1, NNStructDefinition.nbLayers-1);
    B = cell(1, NNStructDefinition.nbLayers-1);
    
    if strcmp(NNStructDefinition.layers{1}.type, 'input') ~= 1
       error('Error : The neural network must begin by an input layer.'); 
    end
    
    if strcmp(NNStructDefinition.layers{end}.type, 'output') ~= 1
       error('Error : The neural network must end by an output layer.'); 
    end
    
    nbFoundInputLayer   = 1;

    for i=2:NNStructDefinition.nbLayers
        
        boCreateW = 0;
        
        LayerDef = NNStructDefinition.layers{i};
        
        if strcmp(LayerDef.type, 'input') == 1
            nbFoundInputLayer  = nbFoundInputLayer + 1;
        elseif strcmp(LayerDef.type, 'output') == 1
            nbFoundOutputLayer = nbFoundOutputLayer + 1;
            boCreateW = 1;
        else
            boCreateW = 1;
        end
        
        if boCreateW == 1
            w = zeros(LayerDef.nbNeurons, NNStructDefinition.layers{i-1}.nbNeurons);
            b = zeros(LayerDef.nbNeurons, 1);
            
            W{i-1} = w;
            B{i-1} = b; 
        end
        
    end
    
    if nbFoundInputLayer == 0
        error('Error : No input layer found.'); 
    elseif nbFoundInputLayer > 1
        error('Error : More than one input layer found.'); 
    elseif nbFoundOutputLayer == 0
        error('Error : No output layer found.'); 
    elseif nbFoundOutputLayer > 1
        error('Error : More than one output layer found.'); 
    end
    
    
    %%%% Initialize the weights and biases
    [W, B] = Initialize_Weights_And_Biases(W, B);
    
    %%%% Train neural network
    it = 0;
    
    while it <= max_iteration_number
        
        nbErrors = 0;
        
        for i=1:size(X, 1)

            [H, Z] = NN_feedforward(NNStructDefinition, transpose(X(i, :)), W, B);

            [W, B] = NN_backpropagate(NNStructDefinition, transpose(X(i, :)), transpose(Y(i, :)), W, B, H, Z, NNStructDefinition.learning_rate);

            if isequal(H{end}, transpose(Y(i, :))) == 0
                nbErrors = nbErrors + 1;
            end
        end
        
        if nbErrors == 0
            break;
        end
        
        it = it + 1;
        
    end
    
end

function [ W, B ] = Initialize_Weights_And_Biases( W, B )

    for i=1:length(W)
        W{i} = -0.001 + (rand(size(W{i}, 1), size(W{i}, 2)) * (0.002));
        B{i} = -0.001 + (rand(size(B{i}, 1), size(B{i}, 2)) * (0.002));
    end

end
    

