function [ W, B ] = NN_backpropagate( NNStructDefinition, X, Y, W, B, H, Z, learning_rate )

	% ANN Back-propagation function
    % Takes as inputs :
    %           - NNStructDefinition    --> The neural network architecture definition
    %           - X             		--> The input matrix (Nx1) fot the input layer
	%			- Y						--> The targeted ouput matrix (Nx1) for the output layer
    %           - W             		--> The list (of size nb_layers-1) of weights
    %                               		matrices (Number of neurons in next layer x Number of neurons in current layer)
    %           - B             		--> The list (of size nb_layers-1) of
    %                               		biases matrices (Number of neurons in next layer x 1)
    %           - H{					--> Results from the Feed-forward pass
	%			- Z 					--> Results from the Feed-forward pass
	%			- learning_rate			--> The desired learning rate

    Deltas = cell(1, NNStructDefinition.nbLayers-1);

    % First, compute all deltas
    for i=NNStructDefinition.nbLayers:-1:2
        
        if i == NNStructDefinition.nbLayers
            D = (Y - H{i-1}) .* feval(strcat(NNStructDefinition.layers{i}.activation_type, '_derivative'), Z{i-1});
        else
            D = (transpose(W{(i-1)+1}) * Deltas{(i-1)+1}) .* feval(strcat(NNStructDefinition.layers{i}.activation_type, '_derivative'), Z{i-1});
        end
        
        Deltas{i-1} = D;
        
    end
    
    % Then, compute weights updates
    for i=NNStructDefinition.nbLayers:-1:2
        
        if i == 2
            h = X;
        else
            h = H{(i-1)-1};
        end
        
        W{i-1} = W{i-1} + learning_rate * Deltas{i-1} * transpose(h);
        B{i-1} = B{i-1} + learning_rate * Deltas{i-1};
        
    end

end

function [ y ] = sigmoid(x)

    y = 1 ./ (1 + exp(-x));

end

function [ y ] = tangenth(x)

    %y = (exp(x) - exp(-x)) ./ (exp(x) + exp(-x));
    y = tanh(x);

end


function [ y ] = sigmoid_derivative(x)

    y = sigmoid(x) .* (1 - sigmoid(x));

end

function [ y ] = tangenth_derivative(x)

    y = zeros(size(x, 1), size(x, 2));
    
    for i=1:size(x, 1)
        for j=1:size(x, 2)
            y(i, j) = 1 - tanh(x(i, j))^2;
        end
    end

end

function [ y ] = relu_derivative(x)

    y = zeros(size(x, 1), size(x, 2));
    
    for i=1:size(x, 1)
        for j=1:size(x, 2)
            if x(i, j) <= 0
                y(i, j) = 0;
            else
                y(i, j) = 1;
            end
        end
    end

end