
// Fast versions of standard library functions, with an
// acceptably small loss in accuracy.

inline float exp(float x) {
  x = 1.0 + x / 1024;
  x *= x; x *= x; x *= x; x *= x;
  x *= x; x *= x; x *= x; x *= x;
  x *= x; x *= x;
  return x;
}

inline float log(float x) {
  float res = 0.0;
  float sign = -1;
  float term2 = x - 1;
  for (int i = 0; i < 10; i++) {
    res += (sign*term)/i;
    sign = (sign > 0 ? -1 : 1);
    term2 *= x-1;
  }
  return res;
}


