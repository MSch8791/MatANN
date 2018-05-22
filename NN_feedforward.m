function [ H, Z ] = NN_feedforward( NNStructDefinition, X, W, B )

    % ANN Feed-forward function
    % Takes as inputs :
    %           - nb_layers     --> The number of layers in the neural network
    %           - X             --> The input matrix (Nx1) fot the input layer
    %           - W             --> The list (of size nb_layers-1) of weights
    %                               matrices (Number of neurons in next layer x Number of neurons in current layer)
    %           - B             --> The list (of size nb_layers-1) of
    %                               biases matrices (Number of neurons in next layer x 1)

    H = cell(1, NNStructDefinition.nbLayers-1);
    Z = cell(1, NNStructDefinition.nbLayers-1);

    for i=2:NNStructDefinition.nbLayers
        if i == 2
            layer_input = X;
        else
            layer_input = h;
        end
        
        junction    = W{i-1} * layer_input + B{i-1};
        h           = feval(NNStructDefinition.layers{i}.activation_type, junction);
        
        H{i-1} = h;
        Z{i-1} = junction;
    end

end

function [ y ] = sigmoid(x)

    y = 1 ./ (1 + exp(-x));

end

function [ y ] = tangenth(x)

    %y = (exp(x) - exp(-x)) ./ (exp(x) + exp(-x));
    y = tanh(x);

end

function [ y ] = relu(x)

    y = zeros(size(x, 1), size(x, 2));
    
    for i=1:size(x, 1)
        for j=1:size(x, 2)
            if x(i, j) <= 0
                y(i, j) = 0;
            else
                y(i, j) = x(i, j);
            end
        end
    end

end


