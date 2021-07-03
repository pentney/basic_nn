#include "dataset.h"

#include "gtest/gtest.h"

class DatasetTest : public ::testing::Test {
 public:
  void SetUp() {
    examples_.push_back({"1.0", "green", "-2.5", "100", "2g", "2"});
    examples_.push_back({"0.0", "blue", "-1.5", "-100", "3f", "2"});
    examples_.push_back({"0.0", "green", "-2.0", "53", "4f", "0"});
    examples_.push_back({"0.0", "red", "-5.5", "22", "4f", "0"});
    examples_.push_back({"1.0", "yellow", "-2.5", "1000", "4f", "1"});

    field_names_ = { "foo", "bar", "baz", "quux", "quuux", "label" };
    expected_output_features_ =
      { "foo", "bar_blue", "bar_green", "bar_red", "bar_yellow",
	"baz", "quux", "quuux_2g", "quuux_3f", "quuux_4f" };
  }

  bool exampleMatches(pair<vector<float>, float> example,
		      const vector<float>& expected_features,
		      float expected_label) {
    EXPECT_EQ(expected_features.size(), example.first.size());
    for (size_t i = 0; i < expected_features.size(); i++) {
      EXPECT_FLOAT_EQ(expected_features[i], example.first[i]) << ": index " << i;
    }
    EXPECT_FLOAT_EQ(expected_label, example.second);
    return true;
  }

  vector<vector<string>> examples_;
  vector<string> field_names_;
  vector<string> expected_output_features_;
  std::unique_ptr<Dataset> dataset_;
};

TEST_F(DatasetTest, ParsesFeatures) {
  dataset_.reset(new Dataset(field_names_,
			     field_names_.size() - 1));
  for (vector<string> example: examples_) {
    dataset_->add_row(example);
  }
  dataset_->process_features();
  EXPECT_EQ(expected_output_features_, dataset_->output_features());

  pair<vector<float>, float> example;
  EXPECT_TRUE(dataset_->next(&example));
  EXPECT_TRUE(exampleMatches(example,
			     {1.0, 0.0, 1.0, 0.0, 0.0, 0.75, 0.18181819,
				 1.0, 0.0, 0.0}, 1.0));
  EXPECT_TRUE(dataset_->next(&example));
  EXPECT_TRUE(exampleMatches(example,
			     {0.0, 1.0, 0.0, 0.0, 0.0, 1, 0.0,
				 0.0, 1.0, 0.0}, 1.0));
  EXPECT_TRUE(dataset_->next(&example));
  EXPECT_TRUE(exampleMatches(example,
			     {0.0, 0.0, 1.0, 0.0, 0.0, 0.875, 0.13909091,
				 0.0, 0.0, 1.0}, 0.0));
  EXPECT_TRUE(dataset_->next(&example));
  EXPECT_TRUE(exampleMatches(example,
			     {0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.11090909,
				 0.0, 0.0, 1.0}, 0.0));
  EXPECT_TRUE(dataset_->next(&example));
  EXPECT_TRUE(exampleMatches(example,
			     {1.0, 0.0, 0.0, 0.0, 1.0, 0.75, 1.0,
				 0.0, 0.0, 1.0}, 0.5));
  EXPECT_FALSE(dataset_->next(&example));
}


TEST_F(DatasetTest, ParsesFeaturesUnscaled) {
  dataset_.reset(new Dataset(field_names_,
			     field_names_.size() - 1,
			     0));
  for (vector<string> example: examples_) {
    dataset_->add_row(example);
  }
  dataset_->process_features();
  EXPECT_EQ(expected_output_features_, dataset_->output_features());

  pair<vector<float>, float> example;
  EXPECT_TRUE(dataset_->next(&example));
  EXPECT_TRUE(exampleMatches(example,
			     {1.0, 0.0, 1.0, 0.0, 0.0, -2.5, 100.0,
				 1.0, 0.0, 0.0}, 2.0));
  EXPECT_TRUE(dataset_->next(&example));
  EXPECT_TRUE(exampleMatches(example,
			     {0.0, 1.0, 0.0, 0.0, 0.0, -1.5, -100.0,
				 0.0, 1.0, 0.0}, 2.0));
  EXPECT_TRUE(dataset_->next(&example));
  EXPECT_TRUE(exampleMatches(example,
			     {0.0, 0.0, 1.0, 0.0, 0.0, -2.0, 53.0,
				 0.0, 0.0, 1.0}, 0.0));
  EXPECT_TRUE(dataset_->next(&example));
  EXPECT_TRUE(exampleMatches(example,
			     {0.0, 0.0, 0.0, 1.0, 0.0, -5.5, 22.0,
				 0.0, 0.0, 1.0}, 0.0));
  EXPECT_TRUE(dataset_->next(&example));
  EXPECT_TRUE(exampleMatches(example,
			     {1.0, 0.0, 0.0, 0.0, 1.0, -2.5, 1000,
				 0.0, 0.0, 1.0}, 1.0));
  EXPECT_FALSE(dataset_->next(&example));
}

TEST_F(DatasetTest, UnscalesLabels) {
  dataset_.reset(new Dataset(field_names_,
			     field_names_.size() - 1,
			     0));
  for (vector<string> example: examples_) {
    dataset_->add_row(example);
  }
  dataset_->process_features();
  EXPECT_FLOAT_EQ(0.0, dataset_->unscale_label(0.0));
  EXPECT_FLOAT_EQ(1.0, dataset_->unscale_label(0.5));
  EXPECT_FLOAT_EQ(2.0, dataset_->unscale_label(1.0));
  EXPECT_FLOAT_EQ(4.0, dataset_->unscale_label(2.0));
  EXPECT_FLOAT_EQ(-2.0, dataset_->unscale_label(-1.0));
}
