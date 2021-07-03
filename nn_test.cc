#include "nn.h"

#include "gtest/gtest.h"
#include "gradient_test.h"

TEST(SigmoidNNLayerTest, TestActivation) {
  SigmoidNNLayer layer(3, 2);
  layer.inWeights.data = { 0.5, 0.2, -1.0, 0.1,
			   0.5, 0.0, 0.0 -0.1 };
  EXPECT_FLOAT_EQ(0.56217653, layer.activation(0, { 1.0, 0.0, 0.25 }));
  EXPECT_FLOAT_EQ(0.7310586, layer.activation(1, { 2.0, 0.0, 3.0 }));
  EXPECT_FLOAT_EQ(0.37754068, layer.activation(1, { -1.0, 0.0, -5.0 }));
}

TEST(SigmoidNNLayerTest, TestLossWithGradients) {
  SigmoidNNLayer layer(3, 1);
  layer.inWeights.data = { 0.5, 0.2, -1.0, 0.0 };
  aResult expected_res(3), res(3);
  expected_res.f = 0.56217653;
  expected_res.loss = 0.57593942;
  expected_res.dloss_df = -0.10776328;
  expected_res.dloss_dx[0] = 1.0;
  expected_res.dloss_dx[1] = 0.0;
  expected_res.dloss_dx[2] = 0.25;
  layer.lossWithGradients(0, { 1.0, 0.0, 0.25 },
			  nullptr, nullptr,
			  1.0, &res);
  EXPECT_FLOAT_EQ(expected_res.f, res.f);
  EXPECT_FLOAT_EQ(expected_res.loss, res.loss);
  EXPECT_FLOAT_EQ(expected_res.dloss_df, res.dloss_df);
  EXPECT_FLOAT_EQ(expected_res.dloss_dx[0], res.dloss_dx[0]);
  EXPECT_FLOAT_EQ(expected_res.dloss_dx[1], res.dloss_dx[1]);
  EXPECT_FLOAT_EQ(expected_res.dloss_dx[2], res.dloss_dx[2]);
}

TEST(SigmoidNNLayerTest, TestGradientAccuracy) {
  GradientTest gradient_test;
  SigmoidNNLayer layer(3, 1);
  layer.inWeights.data = { 1.2, 0.0, 0.5, 0.0 };
  vector<float> inputs { 1.0, 0.2, 0.4 };
  aResult res(3);
  for (float i = -5.0; i < 5.0; i += 0.0005) {
    layer.inWeights.at(0, 1) = i;
    layer.lossWithGradients(0, inputs,
			    nullptr, nullptr,
			    1.0, &res);
    gradient_test.add_point(i, res.loss, res.dloss_df*res.dloss_dx[1]);
  }
  EXPECT_TRUE(gradient_test.gradientsMatchWithin(1e-4));
}

TEST(SigmoidNNLayerTest, TestLossWithGradientsAndNextLayer) {
  SigmoidNNLayer layer(3, 1);
  layer.inWeights.data = { 0.5, 0.2, -1.0, 0.0 };
  vector<aResult> next_layer(2, aResult(2));
  next_layer[0].dloss_df = 1.0;
  next_layer[1].dloss_df = -0.5;
  vector2d<float> next_layer_weights(2, 1);
  next_layer_weights.at(0, 0) = 0.2;
  next_layer_weights.at(1, 0) = 0.5;
  aResult expected_res(3), res(3);
  expected_res.f = 0.56217653;
  expected_res.loss = -0.57593942;
  expected_res.dloss_df = 0.25;
  expected_res.dloss_dx[0] = 1.0;
  expected_res.dloss_dx[1] = 0.0;
  expected_res.dloss_dx[2] = -0.01;
  layer.lossWithGradients(0, { 1.0, 0.0, 0.25 },
			  &next_layer, &next_layer_weights,
			  1.0, &res);
  EXPECT_FLOAT_EQ(expected_res.f, res.f);
  EXPECT_FLOAT_EQ(expected_res.dloss_df, res.dloss_df);
  EXPECT_FLOAT_EQ(expected_res.dloss_dx[0], res.dloss_dx[0]);
  EXPECT_FLOAT_EQ(expected_res.dloss_dx[1], res.dloss_dx[1]);
  EXPECT_FLOAT_EQ(expected_res.dloss_dx[2], res.dloss_dx[2]);
}

TEST(PReluNNLayerTest, TestActivation) {
  PReluNNLayer layer(3, 2, 0.01);
  layer.inWeights.data = { 0.5, 0.2, -1.0, 0.0,
			   0.5, 0.0, 0.0, 0.0 };
  EXPECT_FLOAT_EQ(0.25, layer.activation(0, { 1.0, 0.0, 0.25 }));
  EXPECT_FLOAT_EQ(1, layer.activation(1, { 2.0, 0.0, 3.0 }));
  EXPECT_FLOAT_EQ(-0.005, layer.activation(1, { -1.0, 0.0, -4.5 }));
}

TEST(PReluNNLayerTest, TestLossWithGradients) {
  PReluNNLayer layer(3, 2, 0.01);
  layer.inWeights.data = { 0.5, 0.2, 1.5, 0.0,
			   0.0, 1.0, -0.2, 0.0 };
  aResult expected_res(3), res(3);
  expected_res.f = 0.875;
  expected_res.loss = 0.015625;
  expected_res.dloss_df = -0.25;
  expected_res.dloss_dx[0] = 1.0;
  expected_res.dloss_dx[1] = 0.0;
  expected_res.dloss_dx[2] = 0.25;
  layer.lossWithGradients(0, { 1.0, 0.0, 0.25 },
			  nullptr, nullptr,
			  1.0, &res);
  EXPECT_FLOAT_EQ(expected_res.f, res.f);
  EXPECT_FLOAT_EQ(expected_res.loss, res.loss);
  EXPECT_FLOAT_EQ(expected_res.dloss_df, res.dloss_df);
  EXPECT_FLOAT_EQ(expected_res.dloss_dx[0], res.dloss_dx[0]);
  EXPECT_FLOAT_EQ(expected_res.dloss_dx[1], res.dloss_dx[1]);
  EXPECT_FLOAT_EQ(expected_res.dloss_dx[2], res.dloss_dx[2]);

  expected_res.f = 0.05;
  expected_res.loss = 0.0025;
  expected_res.dloss_df = 0.1;
  expected_res.dloss_dx[0] = -1;
  expected_res.dloss_dx[1] = 0.0;
  expected_res.dloss_dx[2] = -0.25;
  layer.lossWithGradients(1, { -1.0, 0.0, -0.25 },
			  nullptr, nullptr,
			  0.0, &res);
  EXPECT_FLOAT_EQ(expected_res.f, res.f);
  EXPECT_FLOAT_EQ(expected_res.loss, res.loss);
  EXPECT_FLOAT_EQ(expected_res.dloss_df, res.dloss_df);
  EXPECT_FLOAT_EQ(expected_res.dloss_dx[0], res.dloss_dx[0]);
  EXPECT_FLOAT_EQ(expected_res.dloss_dx[1], res.dloss_dx[1]);
  EXPECT_FLOAT_EQ(expected_res.dloss_dx[2], res.dloss_dx[2]);
}

TEST(PReluNNLayerTest, TestLossWithGradientsAndNextLayer) {
  PReluNNLayer layer(3, 2, 0.01);
  layer.inWeights.data = { 0.5, 0.2, 1.5, 0.0,
			   0.0, 1.0, -0.2, 0.0 };
  vector<aResult> next_layer(2, aResult(2));
  next_layer[0].dloss_df = 1.0;
  next_layer[1].dloss_df = -0.5;
  vector2d<float> next_layer_weights(2, 2);
  next_layer_weights.data = { 0.5, 0.5, 0.2, 0.3 };
  aResult expected_res(3), res(3);
  expected_res.f = 0.875;
  expected_res.dloss_df = 0.4;
  expected_res.dloss_dx[0] = 1.0;
  expected_res.dloss_dx[1] = 0.0;
  expected_res.dloss_dx[2] = 0.25;
  layer.lossWithGradients(0, { 1.0, 0.0, 0.25 },
			  &next_layer, &next_layer_weights,
			  1.0, &res);
  EXPECT_FLOAT_EQ(expected_res.f, res.f);
  EXPECT_FLOAT_EQ(expected_res.dloss_df, res.dloss_df);
  EXPECT_FLOAT_EQ(expected_res.dloss_dx[0], res.dloss_dx[0]);
  EXPECT_FLOAT_EQ(expected_res.dloss_dx[1], res.dloss_dx[1]);
  EXPECT_FLOAT_EQ(expected_res.dloss_dx[2], res.dloss_dx[2]);

  expected_res.f = -0.0005;
  expected_res.dloss_df = 0.0035;
  expected_res.dloss_dx[0] = 1.0;
  expected_res.dloss_dx[1] = 0.0;
  expected_res.dloss_dx[2] = 0.25;
  layer.lossWithGradients(1, { 1.0, 0.0, 0.25 },
			  &next_layer, &next_layer_weights,
			  0.0, &res);
  EXPECT_FLOAT_EQ(expected_res.f, res.f);
  EXPECT_FLOAT_EQ(expected_res.dloss_df, res.dloss_df);
  EXPECT_FLOAT_EQ(expected_res.dloss_dx[0], res.dloss_dx[0]);
  EXPECT_FLOAT_EQ(expected_res.dloss_dx[1], res.dloss_dx[1]);
  EXPECT_FLOAT_EQ(expected_res.dloss_dx[2], res.dloss_dx[2]);
}

TEST(PReluNNLayerTest, TestGradientAccuracy) {
  GradientTest gradient_test;
  PReluNNLayer layer(3, 1, 0.01);
  layer.inWeights.data = { 1.2, 0.0, 0.5, 0.0 };
  vector<float> inputs { 1.0, 0.2, 0.4 };
  aResult res(3);
  for (float i = -5.0; i < 5.0; i += 0.001) {
    layer.inWeights.at(0, 1) = i;
    layer.lossWithGradients(0, inputs,
			    nullptr, nullptr,
			    1.0, &res);
    gradient_test.add_point(i, res.loss, res.dloss_df*res.dloss_dx[1]);
  }
  EXPECT_TRUE(gradient_test.gradientsMatchWithin(1e-5));
}

class NNTest : public ::testing::Test {
 public:
  void SetUp() {
    NNParams params(10, 40, 1e-4, 1, 10, 0.01);
    nn.reset(new NN(params));
    nn->addLayer(LayerType::RELU, 4);
    nn->addOutputLayer(LayerType::SIGMOID);
  }

  bool floatVector2DsEqualTo(const vector<vector<float>>& v1,
			      const vector2d<float>& v2) {
    if (v1.size() != v2.row_size) {
      std::cerr << " sizes don't match: expected " << v1.size() <<
	", actual " << v2.row_size << std::endl;
      return false;
    }
    for (size_t i = 0; i < v1.size(); i++) {
      if (v1[i].size() != v2.col_size) {
	std::cerr << "size of vector " << i << " does not match col_size: " <<
	  v2.col_size << std::endl;
	return false;
      }
      for (size_t j = 0; j < v1[i].size(); j++) {
	if (abs(v1[i][j] - v2.at(i, j)) > 1e-6) {
	  std::cerr << "unequal at " << i << ", " << j << ": expected " <<	  
	    v1[i][j] << ", actual " << v2.at(i, j) << std::endl;
	  return false;
	}
      }
    }
    return true;
  }

  std::unique_ptr<NN> nn;
};

TEST_F(NNTest, InitializeWeights) {
  nn->initializeWeights([](size_t i, size_t j, size_t k) {
      return static_cast<float>(0.1*(j+1));
    });
  vector<vector<float>> expectedInWeightsLayer0 =
    {{ 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1 },
     { 0.2, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2 },
     { 0.3, 0.3, 0.3, 0.3, 0.3, 0.3, 0.3, 0.3, 0.3, 0.3 },
     { 0.4, 0.4, 0.4, 0.4, 0.4, 0.4, 0.4, 0.4, 0.4, 0.4 }};
  vector<vector<float>> expectedInWeightsLayer1 =
     {{ 0.1, 0.1, 0.1, 0.1 }};
  EXPECT_TRUE(floatVector2DsEqualTo(expectedInWeightsLayer0, nn->layers[0]->inWeights));
  EXPECT_TRUE(floatVector2DsEqualTo(expectedInWeightsLayer1, nn->layers[1]->inWeights));
}

TEST_F(NNTest, NNInferenceReluToSigmoid) {
  nn->initializeWeights([](size_t i, size_t j, size_t k) {
      return static_cast<float>(0.1*(j+1));
    });
  vector<float> inputs { 0.5, 1.0, -0.1, 2.5, 0.0, 0.0, -0.2, 0.6, 0.5, 0.3 };
  std::unique_ptr<vector<vector<float>>> outputs(nn->makeOutputVector());
  float result = nn->inference(inputs, outputs.get());
  ASSERT_EQ(3, outputs->size());
  ASSERT_EQ(4, (*outputs)[1].size());
  ASSERT_EQ(1, (*outputs)[2].size());
  EXPECT_FLOAT_EQ(0.51, (*outputs)[1][0]);
  EXPECT_FLOAT_EQ(1.02, (*outputs)[1][1]);
  EXPECT_FLOAT_EQ(1.53, (*outputs)[1][2]);
  EXPECT_FLOAT_EQ(2.04, (*outputs)[1][3]);
  EXPECT_FLOAT_EQ(0.62480646, (*outputs)[2][0]);
  EXPECT_FLOAT_EQ(0.62480646, result);
}

TEST_F(NNTest, NNBackPropagateAtDifferentLearningRates) {
  nn->initializeWeights([](size_t i, size_t j, size_t k) {
      return static_cast<float>(0.1*(j+1));
    });
  vector<float> inputs { 0.5, 1.0, -0.1, 2.5, 0.0, 0.0, -0.2, 0.6, 0.5, 0.3 };
  vector<pair<vector<float>, float>> examples;
  float output = 1.0;
  examples.push_back(make_pair(inputs, output));
  GDOptimizerParams params;
  params.learning_rate = 0.01;
  float loss = nn->backpropagate(examples, params);
  EXPECT_FLOAT_EQ(0.47031331, loss);
  vector<vector<float>> updatedWeightsLayer1 =
    {{0.101913, 0.103827, 0.10574, 0.107654}};
  vector<vector<float>> updatedWeightsLayer0 = {
    {0.100191, 0.100382, 0.0999618, 0.100956, 0.1,
     0.1, 0.0999235, 0.100229, 0.100191, 0.100115},
    {0.200195, 0.20039, 0.199961, 0.200974, 0.2,
     0.2, 0.199922, 0.200234, 0.200195, 0.200117},
    {0.300198, 0.300397, 0.29996, 0.300992, 0.3,
     0.3, 0.299921, 0.300238, 0.300198, 0.300119},
    {0.400202, 0.400404, 0.39996, 0.40101, 0.4,
     0.4, 0.399919, 0.400242, 0.400202, 0.400121}};
  EXPECT_TRUE(floatVector2DsEqualTo(updatedWeightsLayer0, nn->layers[0]->inWeights));
  EXPECT_TRUE(floatVector2DsEqualTo(updatedWeightsLayer1, nn->layers[1]->inWeights));

  // Try again with twice the learning rate.
  nn->params->learningRate = 0.02;
  nn->initializeWeights([](size_t i, size_t j, size_t k) {
      return static_cast<float>(0.1*(j+1));
    });
  loss = nn->backpropagate(examples, params);
  // Change in weights should be more significant now.
  updatedWeightsLayer1 = {{0.101913, 0.103827, 0.10574, 0.107654}};
  updatedWeightsLayer0 = {
    {0.100191, 0.100382, 0.0999618, 0.100956, 0.1,
     0.1, 0.0999235, 0.100229, 0.100191, 0.100115},
    {0.200195, 0.20039, 0.199961, 0.200974, 0.2,
     0.2, 0.199922, 0.200234, 0.200195, 0.200117},
    {0.300198, 0.300397, 0.29996, 0.300992, 0.3,
     0.3, 0.299921, 0.300238, 0.300198, 0.300119},
    {0.400202, 0.400404, 0.39996, 0.40101, 0.4,
     0.4, 0.399919, 0.400242, 0.400202, 0.400121}};
  EXPECT_TRUE(floatVector2DsEqualTo(updatedWeightsLayer0, nn->layers[0]->inWeights));
  EXPECT_TRUE(floatVector2DsEqualTo(updatedWeightsLayer1, nn->layers[1]->inWeights));
}

