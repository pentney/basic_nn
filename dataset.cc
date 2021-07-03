#include "dataset.h"

#include <cmath>
#include <iostream>
#include <set>
#include <string>
#include <utility>
#include <vector>

using std::make_pair;
using std::pair;
using std::set;
using std::sqrt;
using std::string;
using std::vector;

void Dataset::add_row(const vector<string>& fields) {
  char *endptr;
  for (size_t i = 0; i < field_names_.size(); i++) {
    // If we've found non-numeric values for this feature
    // or it's not convertible to a float, we will assume
    // it's categorical.
    float val = strtod(fields[i].data(), &endptr);
    if (*endptr == 0) {
      auto range_pair_iter = range_index_.find(i);
      if (range_pair_iter == range_index_.end()) {
	range_index_[i] = make_pair(val, val);
      } else {
	auto& range_pair = range_pair_iter->second;
	if (range_pair.first > val) {
	  range_pair.first = val;
	}
	if (range_pair.second < val) {
	  range_pair.second = val;
	}
      }
      
      auto mv = means_variances_.find(i);
      if (mv == means_variances_.end()) {
	means_variances_[i] = make_pair(val, 0.0);
      } else {
	auto& mv_pair = mv->second;
	size_t n = examples_.size();
	mv_pair.second = (n/(n+1)) *
	  (mv_pair.second +
	   (((mv_pair.first - val) * (mv_pair.first - val)) / (n+1)));
	mv_pair.first = (mv_pair.first * n + val) / (n + 1);
      }
    } else {
      field_index_[i].insert(fields[i]);
    }
  }
  examples_.push_back(fields);
}

float Dataset::scale(float val, float min_val,
		      float max_val) {
  if (scale_ != 0.0) {
    if (min_val == max_val) {
      return 0.0;
    } else {
      return ((val - min_val) * scale_ /
	      (max_val - min_val));
    }
  }
  return val;
}

/*
float Dataset::standardize(float val, float mean,
			    float std_dev) {
  return (val - mean) / std_dev;
}
*/
float Dataset::unscale_label(float val) {
  auto range = range_index_.find(label_index_);
  if (range == range_index_.end()) {
    // No scaling; just return as is.
    return val;
  }
  return (val * (range->second.second - range->second.first)) +
    range->second.first;
}
/*
float Dataset::unstandardize_label(float val) {
  auto mv = means_variances_.find(label_index_);
  if (mv == means_variances_.end()) {
    // No standardization; return as is.
    return val;
  }
  return (val * mv->second.second) + mv->second.first;
}
*/
void Dataset::process_features() {
  for (size_t i = 0; i < field_names_.size(); i++) {
    if (i == label_index_) {      
      continue;
    }
    auto fields = field_index_.find(i);
    if (fields != field_index_.end()) {
      feature_index_.push_back(output_features_.size());
      for (const string& feature_value : fields->second) {
	output_features_.push_back(field_names_[i] + "_" + feature_value);
      }
    } else {
      feature_index_.push_back(output_features_.size());
      output_features_.push_back(field_names_[i]);
    }
  }
  // Change variances to square roots, just to avoid recomputing
  // standard deviations later.
  for (auto mv : means_variances_) {
    mv.second.second = sqrt(mv.second.second);
  }
  pos_ = examples_.begin();
}

void Dataset::process_example(const vector<string>& fields,
			      pair<vector<float>, float>* example) {
  vector<float>& features = example->first;
  features.clear();
  features.reserve(output_features_.size());
  for (size_t i = 0; i < field_names_.size(); i++) {
    if (i == label_index_) {
      float val = strtod(fields[i].data(), nullptr);
      auto range = range_index_.find(i);
      if (range != range_index_.end()) {
	example->second = scale(val, range->second.first,
				range->second.second);
      }	else {
	example->second = val;
      }
      continue;
    }
    auto range = range_index_.find(i);
    if (range != range_index_.end()) {
      float val = strtod(fields[i].data(), nullptr);
      features.push_back(scale(val, range->second.first,
			       range->second.second));
    } else {
      for (const string& value : field_index_[i]) {
	features.push_back(fields[i] == value ? 1.0 : 0.0);
      }
    }
  }
}

bool Dataset::next(pair<vector<float>, float>* example) {
  if (!hasNext()) {
    return false;
  }
  vector<string> input_example = *pos_++;
  process_example(input_example, example);
  return true; 
}
