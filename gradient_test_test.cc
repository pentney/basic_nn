#include "gradient_test.h"

#include <math.h>

#include "gtest/gtest.h"

#include <iostream>

TEST(GradientTestTest, SquaredMatches) {
  GradientTest gradient_test;
  for (float x = 0.0; x < 10.0; x += 0.01) {
    gradient_test.add_point(x, x * x, 2 * x);
  }
  EXPECT_TRUE(gradient_test.gradientsMatchOnAverage(1e-3));

}

TEST(GradientTestTest, WrongDerivDoesntMatch) {
  GradientTest gradient_test;
  for (float x = 0.0; x < 10.0; x += 0.01) {
    gradient_test.add_point(x, x * x, 3 * x);
  }
  EXPECT_FALSE(gradient_test.gradientsMatchOnAverage(1e-3));
}

TEST(GradientTestTest, SquaredDoesntMatchTooCoarse) {
  GradientTest gradient_test;
  for (float x = 0.0; x < 10.0; x += 0.1) {
    gradient_test.add_point(x, x * x, 2 * x);
  }
  EXPECT_FALSE(gradient_test.gradientsMatchOnAverage(1e-3));
  EXPECT_TRUE(gradient_test.gradientsMatchOnAverage(1e-1));
}

TEST(GradientTestTest, LogMatches) {
  GradientTest gradient_test;
  // More fine grained for small values.
  for (float x = 0.01; x < 0.5; x += 0.001) {
    gradient_test.add_point(x, log(x), 1.0/x);
  }
  for (float x = 0.5; x < 5.0; x += 0.01) {
    gradient_test.add_point(x, log(x), 1.0/x);
  }
  EXPECT_TRUE(gradient_test.gradientsMatchOnAverage(1e-4));
}

TEST(GradientTestTest, SineMatches) {
  GradientTest gradient_test;
  for (float x = 0.01; x < 0.5; x += 0.01) {
    gradient_test.add_point(x, sin(x), cos(x));
  }
  EXPECT_TRUE(gradient_test.gradientsMatchOnAverage(1e-4));
}

