// Unit test for utils::objective_func: confirms parameters are parsed and
// H(x) is computed correctly.  Run from the project root so "H_test.txt" is
// found:  ./build.sh test_objective && ./build/test_objective
#include "utils.hpp"
#include <iostream>
#include <cmath>
#include <initializer_list>

using utils::Vec;

static int failures = 0;

static void approx(const char* name, double got, double want, double tol = 1e-9) {
    if (std::fabs(got - want) > tol) {
        std::cerr << "FAIL " << name << ": got " << got << ", want " << want << "\n";
        ++failures;
    } else {
        std::cout << "ok   " << name << " = " << got << "\n";
    }
}

static Vec vec(std::initializer_list<double> vals) {
    Vec x(static_cast<int>(vals.size()));
    int i = 0;
    for (double v : vals) x(i++) = v;
    return x;
}

int main() {
    utils::objective_func obj("H_test.txt");

    // H(x) = -2 x0 - x1 + 0.5 x0 x1 - 0.3 x0 x1 x2
    // Each point isolates one or more coefficients:
    approx("H(0,0,0)", obj.H(vec({0, 0, 0})),  0.0);   // constant (none) -> 0
    approx("H(1,0,0)", obj.H(vec({1, 0, 0})), -2.0);   // linear h0
    approx("H(0,1,0)", obj.H(vec({0, 1, 0})), -1.0);   // linear h1
    approx("H(0,0,1)", obj.H(vec({0, 0, 1})),  0.0);   // no h2
    approx("H(1,1,0)", obj.H(vec({1, 1, 0})), -2.5);   // + quad J01 = 0.5
    approx("H(1,1,1)", obj.H(vec({1, 1, 1})), -2.8);   // + cubic K012 = -0.3
    approx("H(2,1,1)", obj.H(vec({2, 1, 1})), -4.6);   // scaling check

    if (failures) {
        std::cerr << "\n" << failures << " TEST(S) FAILED\n";
        return 1;
    }
    std::cout << "\nALL TESTS PASSED\n";
    return 0;
}
