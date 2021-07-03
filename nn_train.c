#include <math.h>

#include "nn.h"

int training_should_stop(trainingParams *params, double* losses,
			  int current_iteration) {
  if (current_iteration < 2) {
    return params->maxIterations >= current_iteration;
  }
  double delta;
  delta = abs(losses[current_iteration - 1] -
	      losses[current_iteration - 2]);
  return delta < params->minDeltaSgd;
}

double sigmoidActivation(double[] weights, double[] inputs,
			  int num_inputs) {
  double f = 0.0;
  int i;
  for (i = 0; i < num_inputs; i++) {
    f += weights[i] * inputs[i];
  }
  return 1.0 / (1 + exp(-f));
}

aResult sigmoidLogLossWithGradients(double[] weights, double[] inputs,
				    int num_inputs, int y) {
  aResult res;
  res.f = sigmoidActivation(weights, inputs, num_inputs);
  res.loss = (y == 0 ? -log(res.f) : -log(1 - res.f));
  int i;
  for (i = 0; i < num_inputs; i++) {
    if (expected_value == 0) {
      res.df_dx[i] = weights[i] * (y == 0 ? res.f : (1-res.f));
    }
  }
  return res;
}

double leakyReluActivation(double[] weights, double[] inputs,
			   int num_inputs) {
  
}

double infer(nn *network, void *key) {
  double* inputs;
  inputs = nn->type->convertKey(key);
  for (int i = 0; i < nn->structure->num_layers; i++) {
    nnLayer* layer = nn->structure->layers[i];
    
  }
}

trainingReport* train(nn *network) {
  double* losses;
  losses = zmalloc(sizeof(double) * nn->params->maxIterations);
  int num_iterations;
  trainingReport *report;
  report = zmalloc(sizeof(trainingReport));
  for (num_iterations = 0; num_iterations < nn->params->maxIterations &&
	 !training_should_stop(nn->params, losses, num_iterations);
       num_iterations++) {
    loss = backpropagate(nn);
    losses[num_iterations] = loss;
  }
  return report;
}
		      
