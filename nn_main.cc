#include "csv.h"
#include "dataset.h"
#include "nn.h"

#include <iomanip>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <ctime>

void processDate(const string& date, vector<string>* fields) {
  std::istringstream is(date);
  std::ostringstream os;
  std::tm tmb;
  is >> std::get_time(&tmb, "%d-%m-%Y %H:%M");
  os << tmb.tm_year;
  fields->push_back(os.str()); os.str(""); os.clear();
  os << tmb.tm_mon;
  fields->push_back(os.str()); os.str(""); os.clear();
  os << tmb.tm_mday;
  fields->push_back(os.str()); os.str(""); os.clear();
  os << tmb.tm_hour;
  fields->push_back(os.str()); os.str(""); os.clear();
}

int main(int argc, char **argv) {
  io::CSVReader<7> in("Plant_1_Generation_Data.csv");
  vector<string> field_names = {
    "YEAR", "MONTH", "DAY", "HOUR",  "PLANT_ID", "SOURCE_KEY",
		    "DC_POWER", "AC_POWER", "DAILY_YIELD"};
  in.read_header(io::ignore_extra_column,
		    "DATE_TIME", "PLANT_ID", "SOURCE_KEY",
		    "DC_POWER", "AC_POWER", "DAILY_YIELD",
		 "TOTAL_YIELD");
  Dataset dataset(field_names,
		  field_names.size() - 1);
  vector<string> values(7);
  while(in.read_row(values[0], values[1], values[2], values[3],
		    values[4], values[5], values[6])) {
    vector<string> full_fields;
    processDate(values[0], &full_fields);
    for (size_t i = 1; i < values.size(); i++) {
      full_fields.push_back(values[i]);
    }
    dataset.add_row(full_fields);
  }
  dataset.process_features();
  size_t num_fields = dataset.output_features().size();
  NNParams params(num_fields, 200, 1e-8, 4, 10, 0.001);
  NN nn(params);
  nn.addLayer(LayerType::RELU, 10);
  nn.addOutputLayer(LayerType::RELU);
  srand(42);
  nn.initializeWeights([](size_t i, size_t j, size_t k) {
      return static_cast<float>(((rand() % 100)*0.01)-0.5);
    },
    [](size_t i, size_t j) {
      return static_cast<float>(((rand() % 100)*0.01)-0.5);
    });

  pair<vector<float>, float> example;
  for (dataset.next(&example); dataset.hasNext();
       dataset.next(&example)) {
    nn.submitForAdd(example);
  }

  TrainingReport report;
  nn.train(&report);
  std::cout << report.toString() << std::endl;
  return 0;
}
