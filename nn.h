#ifndef __NN_H_
#define __NN_H_

#include <iostream>
#include <memory>
#include <sstream>
#include <string>
#include <vector>
#include <utility>

using std::endl;
using std::pair;
using std::vector;
using std::ostringstream;
using std::string;

#define NN_OK 0
#define NN_ERR 1

#define DEFAULT_MIN_DELTA 1e-6

typedef enum {
  SIGMOID,
  RELU
} LayerType;

// A more cache-friendly version of a two-dimensional vector (it
// keeps all rows in the 2D structure closer together in memory.)
// Currently assumes the same dimension for every inner vector, 
// so it's more like a matrix.
template <typename T>
class vector2d {
 public:
  vector<T> data;
  size_t row_size, col_size;

  vector2d<T>() { col_size = row_size = 0; }
  vector2d<T>(int ic, int jc) { resize(ic, jc); }
  void resize(size_t ic, size_t jc, T def) {
    row_size = ic;
    col_size = jc;
    data.resize(ic * jc, def);
  }
  void resize(size_t ic, size_t jc) {
    row_size = ic;
    col_size = jc;
    data.resize(ic * jc);
  }
  T& at(size_t i, size_t j) { return data[i * col_size + j]; }
  const T& at(size_t i, size_t j) const {
    return data[i * col_size + j];
  }
};

// Result with derivatives for each input variable.
struct aResult {
  float f;
  float loss, dloss_df;  // derivative w.r.t. activation
  vector<float> dloss_dx;  // derivative w.r.t. previous layer

  aResult(int num_vars) { dloss_dx.resize(num_vars, 0.0); }
  string toString() const {
    std::ostringstream out;
    out << "f: " << f << " loss: " << loss << " dloss/df: " <<
      dloss_df << " grad (prev): ";
    for (float val : dloss_dx) {
      out << val << " ";
    }
    return out.str();
  }
};

struct GDOptimizerParams {
  float learning_rate;
};

struct TrainingReport {
  vector<float> losses;  // loss at each iteration
  float timeElapsed;     // total training time

  string toString() {
    ostringstream out;
    out << "Training report: " << endl;
    for (size_t i = 0; i < losses.size(); i++) {
      out << " - iteration " << i << ": loss " << losses[i] << endl;
    }
    out << "--- training completed" << endl;
    return out.str();
  }
};

struct NNLayer {
  vector2d<float> inWeights;
  vector<float> bias;

  void Init(unsigned int num_inputs, unsigned int num_outputs) {
    inWeights.resize(num_outputs, num_inputs);
    bias.resize(num_outputs);
  }
  
  virtual float activation(size_t unit,
			    const vector<float>& inputs) const = 0;
  virtual void lossWithGradients(size_t unit,
				 const vector<float>& inputs,
				 const vector<aResult>* next_layer_loss,
				 const vector2d<float>* next_layer_weights,
				 float y, aResult* res) const = 0;
  virtual int interpretOutput(float output) const = 0;
  void updateWeights(const vector<aResult>& lossesAndGrads,
		     const GDOptimizerParams& opt_params);
  string toString() const {
    std::ostringstream out;
    out << "wts: ";
    for (size_t i = 0; i < inWeights.row_size; i++) {
      out << "{";
      for (size_t j = 0; j < inWeights.col_size; j++) {
	out << inWeights.at(i, j) << (j == inWeights.col_size - 1 ?
				      "}" : ", ");
      }
    }
    out << " bias: {";
    for (size_t i = 0; i < bias.size(); i++) {
      out << bias[i] << (i == bias.size() - 1 ?
			 "}" : ", ");
    }
    return out.str();
  }
};

struct PReluNNLayer: public NNLayer {
  float slope;
  PReluNNLayer() {}
  PReluNNLayer(unsigned int num_inputs, unsigned int num_outputs,
	       float sl) {
    NNLayer::Init(num_inputs, num_outputs);
    slope = sl;
  }

  float activation(size_t unit,
		   const vector<float>& inputs) const;
  float activation(size_t unit,
		   const vector<float>& inputs,
		   bool* ignore) const;
  void lossWithGradients(size_t unit,
			 const vector<float>& inputs,
			 const vector<aResult>* next_layer_loss,
			 const vector2d<float>* next_layer_weights,
			 float y, aResult* res) const;
  int interpretOutput(float output) const;
};

struct SigmoidNNLayer: public NNLayer {
  float threshold;
  SigmoidNNLayer() {}
  SigmoidNNLayer(unsigned int num_inputs,
	  	 unsigned int num_outputs,
		 float th=0.5) {
   NNLayer::Init(num_inputs, num_outputs);
   threshold = th;
 }
  float activation(size_t unit,
    		    const vector<float>& inputs) const;
  void lossWithGradients(size_t unit,
			 const vector<float>& inputs,
			 const vector<aResult>* next_layer_loss,
			 const vector2d<float>* next_layer_weights,
			 float y, aResult* res) const;
  int interpretOutput(float output) const;
};

struct NNParams {
  unsigned int numInputs;
  unsigned int maxIterations;
  float minDeltaSgd;
  size_t patience;
  unsigned int sgdBatchSize;
  float learningRate;
  NNParams(unsigned int numinputs,
	   unsigned int maxiter,
	   float mindeltasgd,
	   size_t patience,
	   unsigned int sgdbatchsize,
	   float learningrate) :
    numInputs(numinputs),
    maxIterations(maxiter),
    minDeltaSgd(mindeltasgd),
    patience(patience),
    sgdBatchSize(sgdbatchsize),
    learningRate(learningrate) {}
  NNParams(const NNParams& p) :
      NNParams(p.numInputs, p.maxIterations, p.minDeltaSgd,
	       p.patience, p.sgdBatchSize, p.learningRate) {}
};

class NN {
 public:
  vector<std::unique_ptr<NNLayer>> layers;
  vector<pair<vector<float>, float>> examples;

  std::unique_ptr<NNParams> params;
  NN(const NNParams& nn_params) {
    params.reset(new NNParams(nn_params));
  }

  bool addLayer(LayerType type, size_t num_units);
  bool addOutputLayer(LayerType type);
  void submitForAdd(const pair<vector<float>, float>& example);
  float inference(const vector<float>& inputs,
		   vector<vector<float>>* outputs) const;
  float inference(const vector<float>& inputs) const;
  int lookup(const vector<float>& inputs) const;
  bool train(TrainingReport* report);
  void initializeWeights(float (*init)(size_t, size_t, size_t),
			 float( *init_bias)(size_t, size_t));
  vector<vector<float>>* makeOutputVector() const;

  string toString() const {
    ostringstream out;
    for (size_t i = 0; i < layers.size(); i++) {
      out << "layer " << i << ": " <<
	layers[i]->toString() << "\n";
    }
    return out.str();
  }  
  bool trainingShouldStop(const TrainingReport* report) const;
  float backpropagate(const vector<pair<vector<float>, float>>& examples,
		       const GDOptimizerParams& opt_params);
};

#endif
