function [ Y_pred ] = NN_predict( NNStructDefinition, X, W, B )

	% Predict outputs given the input data and the previously trained weights and biases by NN_train
    % Takes as inputs :
    %           - NNStructDefinition    --> The neural network architecture definition
    %           - X             		--> The input matrix fot the input layer
	%			- W						--> The pre-trained weights of the neural network
    %           - B						--> The pre-trained biases of the neural network

    Y_pred = zeros(size(X, 1), NNStructDefinition.layers{end}.nbNeurons);

    for i=1:size(X, 1)
        
        [H, ~] = NN_feedforward(NNStructDefinition, transpose(X(i, :)), W, B);
        
        Y_pred(i, :) = transpose(H{end});
        
    end

end

