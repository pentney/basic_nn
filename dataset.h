#include <string>
#include <map>
#include <set>
#include <string>
#include <vector>

using std::map;
using std::pair;
using std::set;
using std::string;
using std::vector;

class Dataset {
 public:
 Dataset(const vector<string>& field_names,
	 size_t label_index,
	 float scale = 1.0) : field_names_(field_names),
    label_index_(label_index),
    scale_(scale),
    pos_(examples_.begin()) {}

  const vector<string>& output_features() {
    return output_features_;
  }
  void add_row(const vector<string>& fields);
  void process_features();
  void process_example(const vector<string>& fields,
		       pair<vector<float>, float>* example);
  bool hasNext() { return pos_ != examples_.end(); } 
  bool next(pair<vector<float>, float>* example);

  float scale(float val, float min_val, float max_val);
  // Given a scaled label value, unscale it according to
  // the processing for this dataset.
  float unscale_label(float val);

 private:
  // Ranges for numeric features.
  map<size_t, pair<float, float>> range_index_;
  // Mean/variances of numeric features,
  // for normal scaling (not implemented yet).
  map<size_t, pair<float, float>> means_variances_;
  // Sets of unique values for non-numeric features.
  map<size_t, set<string>> field_index_;
  // Map of input feature column index to output feature columns index.
  vector<size_t> feature_index_;

  // Raw examples, to process as requested.
  vector<vector<string>> examples_;
  vector<string> output_features_;

  vector<string> field_names_;  
  size_t label_index_;
  float scale_;
  vector<vector<string>>::iterator pos_;
};
