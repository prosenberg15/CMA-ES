#pragma once

#include <iostream>
#include <fstream>
#include <string>
#include <random>
#include <cmath>
#include <vector>
#include <array>
#include <algorithm>
#include <sstream>
#include <stdexcept>
#include <cassert>
#include <Eigen/Dense>

// some utility functions
namespace utils {

using Vec = Eigen::VectorXd;
using Mat = Eigen::MatrixXd;

/* 
   data structs to contain quadradtic and cubic
   parameters of H(x)
*/
struct T2 {int i, j; double Jij;};
struct T3 {int i, j, k; double Kijk;};

// ---------------------------------------------------------------------------
// Random number generation
// ---------------------------------------------------------------------------
// CMA-ES samples standard-normal vectors z ~ N(0, I) every generation, so we
// keep a single seeded generator around and hand out N(0,1) draws from it.
class RNG {
public:
    explicit RNG(unsigned seed = std::random_device{}())
        : gen_(seed), normal_(0.0, 1.0) {}

    // A single N(0,1) sample.
    double gauss() { return normal_(gen_); }

    // A length-n vector of i.i.d. N(0,1) samples.
    Vec gauss_vec(int n) {
        Vec v(n);
        for (int i = 0; i < n; ++i) v(i) = normal_(gen_);
        return v;
    }

private:
    std::mt19937_64 gen_;
    std::normal_distribution<double> normal_;
};

// ---------------------------------------------------------------------------
// Benchmark objective functions (all minimized, global optimum value 0)
// ---------------------------------------------------------------------------

// Sphere: f(x) = sum x_i^2.  Optimum at x = 0.
inline double sphere(const Vec& x) {
    return x.squaredNorm();
}

// Rosenbrock: classic narrow curved valley.  Optimum at x = (1, 1, ..., 1).
inline double rosenbrock(const Vec& x) {
    double sum = 0.0;
    for (int i = 0; i + 1 < x.size(); ++i) {
        const double a = x(i + 1) - x(i) * x(i);
        const double b = 1.0 - x(i);
        sum += 100.0 * a * a + b * b;
    }
    return sum;
}

class objective_func{

    public:
        objective_func(const std::string& fname){
            read_params(fname);
        }

        objective_func() = default;

        /* Read the parameters of the objective function */
        void read_params(const std::string& fname) {
            std::ifstream in(fname);
            if (!in) throw std::runtime_error("cannot open " + fname);

            std::string line;
            // read header, making sure to skip empty lines and comments
            while (std::getline(in, line)) {
                if (line.empty() || line[0] == '#') continue;
                std::istringstream hs(line);
                hs >> nvar_ >> R_;
                break;
            }

            hi = Vec::Zero(nvar_);
            Jij.clear();
            Kijk.clear();

            while (std::getline(in, line)) {
                if (line.empty() || line[0] == '#') continue; // skip empty and commented lines
                std::istringstream ss(line);
                int deg; ss >> deg; // 1st part of line gives order of term
                // 1st order terms
                if (deg == 1) {
                    int i; double c; ss >> i >> c;
                    check(i); hi(i) += c;
                } 
                // 2nd order terms
                else if (deg == 2) {
                    int i, j; double c; ss >> i >> j >> c;
                    check(i); check(j);
                    if (i == j) throw std::runtime_error("i<j for degree-2 terms");
                    if (i > j) std::swap(i, j);
                    Jij.push_back({i, j, c});
                } 
                // 3rd order terms
                else if (deg == 3) {
                    int i, j, k; double c; ss >> i >> j >> k >> c;
                    std::array<int,3> a{i,j,k};
                    std::sort(a.begin(), a.end());
                    check(i); check(j); check(k);
                    if (i == j || j == k || i == k) throw std::runtime_error("i<j<k for degree-3 terms");
                    Kijk.push_back({a[0],a[1],a[2],c});
                } else {
                    throw std::runtime_error("parameters must be 1st, 2nd or 3rd order");
                }
            }
        }

        // Write params to a file in format suitable for read_params()
        void write_params(const std::string& fname) const {
            std::ofstream out(fname);
            if (!out) throw std::runtime_error("cannot open " + fname + " for writing");
            out.precision(17);                       // enough digits to round-trip doubles

            out << nvar_ << " " << R_ << "\n";        // header: n R
            for (int i = 0; i < nvar_; ++i)           // linear terms:    1 i coeff
                out << "1 " << i << " " << hi(i) << "\n";
            for (const auto& t : Jij)                 // quadratic terms: 2 i j coeff
                out << "2 " << t.i << " " << t.j << " " << t.Jij << "\n";
            for (const auto& t : Kijk)                // cubic terms:     3 i j k coeff
                out << "3 " << t.i << " " << t.j << " " << t.k << " " << t.Kijk << "\n";
        }

        /* 
            Routine to compute objective function H(x)
                takes parameter vector, x
                returns scalar H(x)
        */
        double H(const Vec& x) const {

            assert(x.size() == nvar_);
            double Hloc = 0.0;

            // term 1
            for(int i = 0; i < nvar_; i++) Hloc += hi(i)*x(i);
            // term 2
            for(const auto& t : Jij) Hloc += t.Jij*x(t.i)*x(t.j);
            // term 3
            for(const auto& t : Kijk) Hloc += t.Kijk*x(t.i)*x(t.j)*x(t.k);

            return Hloc;

        }
        
        // function to generate test params
        void generate_test_params(int n, double R_val, unsigned seed){
            nvar_ = n;
            R_ = R_val;
            RNG rng(seed);

            const double scale = 4.0;
            // hi in [-2,2]
            hi = scale*rng.gauss_vec(nvar_);
            // Jij
            Jij.clear();
            for(int i = 0; i < nvar_; i++){
                for(int j = i+1; j < nvar_; j++){
                    Jij.push_back({i,j,scale*rng.gauss()});
                }
            }
            // Kijk
            Kijk.clear();
            for(int i = 0; i < nvar_; i++){
                for(int j = i+1; j < nvar_; j++){
                    for(int k = j+1; k < nvar_; k++){
                        Kijk.push_back({i,j,k,scale*rng.gauss()-0.5}); 
                    }
                }
            }
        }

        int nvar() const {return nvar_;};

        double R() const {return R_;};

        /* small function to insure parameters of H are valid*/
        void check(int i){
            if(i < 0 || i >= nvar_) throw std::runtime_error("invalid parameter for H(x)");
        }

    private:
        int nvar_ = 0;
        double R_ = 0.0;
        Vec hi;
        std::vector<T2> Jij;
        std::vector<T3> Kijk;

};

} // namespace utils
