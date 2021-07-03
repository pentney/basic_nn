#include <assert.h>
#include <math.h>
#include <map>
#include <utility>
#include <vector>

#include <iostream>

using std::make_pair;
using std::pair;
using std::vector;

namespace {
  typedef std::map<float, pair<float, float>> mapping;
}  // namespace

class GradientTest {
private:
  mapping x_to_y_dy_dx;

public:
  GradientTest() {}
  void add_point(float x, float y, float dy_dx) {
    x_to_y_dy_dx[x] = make_pair(y, dy_dx);
  }
  
  void add_points(vector<float> x,
		  vector<float> y,
		  vector<float> dy_dx) {
    for (size_t i = 0; i < x.size(); i++) {
      add_point(x[i], y[i], dy_dx[i]);
    }
  }

  float pairError(float x1, float x2) {
    assert (x1 != x2);
    pair<float, float> y_dy_dx_1 = x_to_y_dy_dx.at(x1),
      y_dy_dx_2 = x_to_y_dy_dx.at(x2);
    float diff = y_dy_dx_2.first - y_dy_dx_1.first;  // f(x2) - f(x1)
    //std::cout << "x1: " << x1 <<  " y2 - y1: " << diff << " x2 - x1: " << (x2 - x1) << " deriv: " << (y_dy_dx_1.second * (x2-x1)) << std::endl;
    float res = abs(diff - (y_dy_dx_1.second * (x2 - x1)));
    //std::cout << " pair error: " << res << std::endl;
    return res;
  }

  float totalError() {
    assert(x_to_y_dy_dx.size() >= 2);
    float error = 0.0;
    mapping::iterator iter = x_to_y_dy_dx.begin(),
      next = x_to_y_dy_dx.begin();
    for (++next; next != x_to_y_dy_dx.end();
	 ++iter, ++next) {
      error += pairError(iter->first, next->first);
    }
    return error;
  }

  bool gradientsMatchOnAverage(float tolerance) {
    return (totalError() / x_to_y_dy_dx.size()) < tolerance;
  }

  bool gradientsMatchWithin(float tolerance) {
    assert(x_to_y_dy_dx.size() >= 2);
    mapping::iterator iter = x_to_y_dy_dx.begin(),
      next = x_to_y_dy_dx.begin();
    for (++next; next != x_to_y_dy_dx.end();
	 ++iter, ++next) {
      if (pairError(iter->first, next->first) > tolerance) {
	return false;
      }
    }
    return true;
  }
};
