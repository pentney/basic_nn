#include "nn.h"

#include <math.h>
#include <assert.h>
#include <memory>
#include <iostream>
#include <vector>

using std::pair;
using std::vector;

void NNLayer::updateWeights(const vector<aResult>& lossesAndGrads,
			    const GDOptimizerParams& opt_params) {
  float grad;
  for (size_t i = 0; i < inWeights.row_size; i++) {
    for (size_t j = 0; j < inWeights.col_size; j++) {
      grad = lossesAndGrads[i].dloss_df *
	lossesAndGrads[i].dloss_dx[j];
      inWeights.at(i, j) -= grad * opt_params.learning_rate;
    }
  }
  for (size_t i = 0; i < bias.size(); i++) {
    bias[i] = lossesAndGrads[i].dloss_df *
      opt_params.learning_rate;
  }
}


float SigmoidNNLayer::activation(size_t unit,
				  const vector<float>& inputs)
  const {
  float f = 0.0;
  for (size_t i = 0; i < inputs.size(); i++) {
    f += inWeights.at(unit, i) * inputs[i];
  }
  f += bias[unit];
  return 1.0 / (1 + exp(-f));
}

void SigmoidNNLayer::lossWithGradients(size_t unit,
				       const vector<float>& inputs,
				       const vector<aResult>* next_layer_loss,
				       const vector2d<float>* next_layer_weights,
				       float y,
				       aResult* res) const {
  res->f = activation(unit, inputs);
  double activ_deriv = (res->f)*(1-res->f);
  if (next_layer_loss == nullptr) {
    res->loss = -(y * log(res->f) + (1-y) * log(1 - res->f));
    res->dloss_df = (res->f - y) * activ_deriv;
   } else {
    res->dloss_df = 0.0;
    for (size_t i = 0; i < next_layer_weights->row_size; i++) {
      res->dloss_df += (*next_layer_loss)[i].dloss_df *
	next_layer_weights->at(i, unit);
    }
    res->dloss_df *= activ_deriv;
  }
  for (size_t i = 0; i < inputs.size(); i++) {
    res->dloss_dx[i] = inputs[i];
  }
}

int SigmoidNNLayer::interpretOutput(float output) const {
  return (output >= threshold ? 1 : 0);
}

float PReluNNLayer::activation(size_t unit,
			       const vector<float>& inputs,
			       bool* nonneg) const {
  float f = 0.0;
  //std::cout << " unit: " << unit << std::endl;
  for (size_t i = 0; i < inputs.size(); i++) {
    //std::cout << " - weights: " << inWeights.at(unit, i) << " input: " << inputs[i] << std::endl;
    f += inWeights.at(unit, i) * inputs[i];
  }
  f += bias[unit];
  *nonneg = f >= 0;
  return (f >= 0 ? f : slope * f);
}

float PReluNNLayer::activation(size_t unit,
			       const vector<float>& inputs) const {
  bool ignore;
  return activation(unit, inputs, &ignore);
}

void PReluNNLayer::lossWithGradients(size_t unit,
				     const vector<float>& inputs,
				     const vector<aResult>* next_layer_loss,
				     const vector2d<float>* next_layer_weights,
				     float y, aResult* res) const {
  bool nonneg;
  res->f = activation(unit, inputs, &nonneg);
  double activ_deriv = (nonneg ? 1.0 : slope);
  if (next_layer_loss == nullptr) {
    res->loss = (y - res->f);
    res->dloss_df = -2 * res->loss * activ_deriv;
    res->loss *= res->loss;
  } else {
    res->dloss_df = 0.0;
    for (size_t i = 0; i < next_layer_weights->row_size; i++) {
      res->dloss_df += (*next_layer_loss)[i].dloss_df *
	next_layer_weights->at(i, unit);
    }
    res->dloss_df *= activ_deriv;
  }
  for (size_t i = 0; i < inputs.size(); i++) {
    res->dloss_dx[i] = inputs[i];
  }
}

int PReluNNLayer::interpretOutput(float output) const {
  return (int) output + 0.5f;
}

void NN::initializeWeights(float (*init)(size_t, size_t, size_t),
			   float (*init_bias)(size_t, size_t)) {
  for (size_t i = 0; i < layers.size(); i++) {
    for (size_t j = 0; j < layers[i]->inWeights.row_size; j++) {
      for (size_t k = 0; k < layers[i]->inWeights.col_size; k++) {
	layers[i]->inWeights.at(j, k) = init(i, j, k);
      }
      layers[i]->bias[j] = init_bias(i, j);
    }
  }
}

bool NN::trainingShouldStop(const TrainingReport* report) const {
  if (report->losses.size() < 2 + params->patience) {
    return params->maxIterations <= report->losses.size();
  }
  size_t i;
  for (i = 0; i < params->patience &&
	 (2 + i) <= report->losses.size(); i++) {
    float delta = report->losses[report->losses.size() -
			   (2 + i)] -
      report->losses[report->losses.size() -
		     (1 + i)];
    if (delta > params->minDeltaSgd) {
      return false;
    }
  }
  return i == params->patience;
}

bool NN::addLayer(LayerType type, size_t num_units) {
  size_t num_inputs = (layers.size() > 0 ?
		       layers.back()->inWeights.row_size :
		       params->numInputs);
  switch(type) {
  case RELU:
    layers.emplace_back(new PReluNNLayer(num_inputs, num_units, 0.01));
    break;
  case SIGMOID:
    layers.emplace_back(new SigmoidNNLayer(num_inputs, num_units));
    break;
  }
  return true;
}

bool NN::addOutputLayer(LayerType type) {
  assert(layers.size() > 0);
  int num_inputs = (layers.size() > 0 ?
		    layers.back()->inWeights.row_size :
		    params->numInputs);
  switch(type) {
  case RELU:
    layers.emplace_back(new PReluNNLayer(num_inputs, 1, 0.01));
    break;
  case SIGMOID:
    layers.emplace_back(new SigmoidNNLayer(num_inputs, 1));
    break;
  }
  return true;  
}

void NN::submitForAdd(const pair<vector<float>, float>& example) {
  examples.push_back(example);
}

vector<vector<float>> *NN::makeOutputVector() const {
  vector<vector<float>> *outputs =
    new vector<vector<float>>;
  outputs->push_back(vector<float>(params->numInputs));
  for (auto& layer_ptr : layers) {
    NNLayer* layer = layer_ptr.get();
    outputs->push_back(vector<float>(layer->inWeights.row_size));
  }
  return outputs;
}

float NN::backpropagate(const vector<pair<vector<float>, float>>& examples,
			 const GDOptimizerParams& opt_params) {
  float total_loss = 0.0;
  vector<vector<aResult>> output_gradient_results(layers.size());
  for (size_t i = 0; i < layers.size(); i++) {
    output_gradient_results[i].resize(layers[i]->inWeights.row_size,
				      aResult(layers[i]->inWeights.col_size));
  }
  std::unique_ptr<vector<vector<float>>> outputs(makeOutputVector());
  for (const pair<vector<float>, float>& example : examples) {
    /*
    std::cout << " backpropagating: ";
    for (float f : example.first) {
      std::cout << f << " ";
    }
    std::cout << " - label: " << example.second << std::endl;
    */
    inference(example.first, outputs.get());
    //std::cout << " inferred value: " << outputs->back()[0] << std::endl;
    for (size_t i = layers.size() - 1; i < layers.size(); i--) {
      NNLayer* layer = layers[i].get();
      const vector2d<float>* next_layer_weights =
	(i < layers.size() - 1 ? &(layers[i+1]->inWeights) : nullptr);
      const vector<aResult>* next_layer_loss =
	(i < layers.size() - 1 ? &(output_gradient_results[i+1]) : nullptr);
      for (size_t j = 0; j < layer->inWeights.row_size; j++) {
	layer->lossWithGradients(j, (*outputs)[i],
	                         next_layer_loss, next_layer_weights,
				 example.second,
				 &output_gradient_results[i][j]);
      }
      layer->updateWeights(output_gradient_results[i], opt_params);
      /*
      for (size_t i = 0; i < layer->inWeights.row_size; i++) {
	std::cout << "{";
	for (size_t j = 0; j < layer->inWeights.col_size; j++) {
	  std::cout << layer->inWeights.at(i, j) <<
	    (j == layer->inWeights.col_size - 1 ? "}, " : ", ");
	}
      }
      std::cout << "bias: {";
      for (size_t i = 0; i < layer->bias.size(); i++) {
	std::cout << layer->bias[i] <<
	  (i == layer->bias.size() - 1 ? "} " : ", ");
      }
      std::cout << endl;
      */
    }
    aResult res = output_gradient_results.back()[0];
    /*
    if (rand()%20000 == 0) {
      std::cout << "estimated: " << res.f << " actual: " <<
	example.second << " loss: " << res.loss << std::endl;
    }
    */
    total_loss += res.loss;
  }

  if (examples.size() > 0) {
    total_loss /= examples.size();
  }
  return total_loss;
}

float NN::inference(const vector<float>& inputs,
		     vector<vector<float>>* outputs) const {
  (*outputs)[0] = inputs;
  for (size_t i = 0; i < layers.size(); i++) {
    const NNLayer* layer = layers[i].get();
    for (size_t j = 0; j < layer->inWeights.row_size; ++j) {
      (*outputs)[i+1][j] = layer->activation(j, (*outputs)[i]);
    }
  }
  return outputs->back()[0];
}

float NN::inference(const vector<float>& inputs) const {
  std::unique_ptr<vector<vector<float>>> outputs(makeOutputVector());
  return inference(inputs, outputs.get());
}

bool NN::train(TrainingReport* report) {
  report->losses.clear();
  GDOptimizerParams opt_params;
  opt_params.learning_rate = params->learningRate;
  for (size_t num_iterations = 0; num_iterations < params->maxIterations &&
	 !trainingShouldStop(report); num_iterations++) {
    std::cout << " iteration: " << num_iterations;
    float loss = 0.0;
    loss = backpropagate(examples, opt_params);
    std::cout << " loss: " << loss << std::endl;
    report->losses.push_back(loss);
  }
  return true;
}

int NN::lookup(const vector<float>& inputs) const {
  return layers.back()->interpretOutput(inference(inputs));
}

