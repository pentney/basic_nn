#include <math.h>
#include <map>
#include <utility>

class GradientTest {
  typedef map<double, pair<double, double>> mapping;
private:
  mapping x_to_y_dy_dx;

public:
  GradientTest() {}
  void add_point(double x, double y, double dy_dx) {
    x_to_y_dy_dx[x] = make_pair(y, dy_dx);
  }
  
  void add_points(vector<double> x,
		  vector<double> y,
		  vector<double> dy_dx) {
    for (int i = 0; i < x.size(); i++) {
      add_point(x[i], y[i], dy_dx[i]);
    }
  }

  double pairError(double x1, double x2) {
    assert (x1 != x2);
    pair<double, double> y_dy_dx_1 = x_to_y_dy_dx.at(x1),
      y_dy_dx_2 = x_to_y_dy_dx.at(x2);
    double diff = y_dy_dx_2.first - y_dy_dx_1.first;  // f(x2) - f(x1)
    return abs((diff / (x2 - x1)) - y_dy_dx_1.second);
  }

  double totalError() {
    double error = 0.0;
    mapping::iterator iter = x_to_y_dy_dx.begin();
    for (mapping::iterator next = iter + 1;
	 next != x_to_y_dy_dx.end();
	 ++iter, ++next) {
      error += pairError(iter.first, next.first);
    }
    return error;
  }

  bool gradientsMatchOnAverage(double tolerance) {
    return (totalError() / x_to_dy_dx.size()) < tolerance;
  }
};
