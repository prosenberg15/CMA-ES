#pragma once
#include "params.hpp"
#include "utils.hpp"
#include<numeric>
#include<functional>
#include<Eigen/Dense>
#include<deque>

using Vec = Eigen::VectorXd;
using Mat = Eigen::MatrixXd;

class cmaes{

    public:
        cmaes(int n, double sigma0_, double R_, std::function<double(const Vec&)> objective, unsigned seed = 42) : 
            par(n), 
            d(par.n-1),
            obj(objective), 
            rng(seed) {
            R = R_;
            m = Vec::Zero(n);
            m.setConstant(R/(1.0*n));
            sigma = sigma0_;

            init_frame();
        
            p_c = Vec::Zero(n-1);
            p_sigma = Vec::Zero(n-1);

            B_C = Mat::Identity(d,d);
            D_C = Vec::Ones(d);

            EN = std::sqrt((double)d) * (1.0 - 0.25/d + 1.0/(21.0*d*d));

            sigma0 = sigma0_;

            hist_cap  = 10 + std::ceil(30.0*par.n/par.lambda);

            boundary_floor = 1e-8;

            feval_budget = 10000000;
        };

    private:

        const params par; // strategy params
        const int d; // int to store relevant dimension, n-1

        // distribution
        Vec m; // mean
        double sigma; // step size

        Mat E; // orthonormal frame
        Mat C; // covariance (n-1) x (n-1)

        // eigendecomposition of C
        Mat B_C; 
        Vec D_C;

        // evolution paths
        Vec p_c;
        Vec p_sigma;

        // population
        Mat Z, Y;
        Mat X;

        long counteval = 0;
        long eigeneval = 0;
        double R;

        double EN;
        int gen = 0;

        // instance of objective function
        std::function<double(const Vec&)> obj;

        // track best results
        Vec x_best;
        double H_best = INFINITY;

        std::string pending_stop;

        double sigma0;
        long feval_budget;

        std::deque<double> fhist;
        int hist_cap;

        double boundary_floor;

        // random number generator for sampling
        utils::RNG rng;

        // small function to build helmert basis (Euclidean ONB)
        static Mat helmert_basis(int n) {
            Mat B = Mat::Zero(n, n - 1);
            for (int k = 1; k <= n - 1; ++k) {
                double s = 1.0 / std::sqrt(double(k) * (k + 1));
                for (int i = 0; i < k; ++i) B(i, k - 1) = s;
                B(k, k - 1) = -double(k) * s;
            }
            return B;
        }

        /*
            g_m-orthonormalize frame
        */
        Mat build_frame() const {
            Mat B = helmert_basis(par.n);
            Mat M = B.transpose() * (m.cwiseInverse().asDiagonal() * B);
            Mat L = Eigen::LLT<Mat>(M).matrixL();
            return L.triangularView<Eigen::Lower>().solve(B.transpose()).transpose();
        }

        void init_frame() { 
            E = build_frame(); 
            C = Mat::Identity(d,d); 
        } 

        void sample(){
            Z = Mat::Zero(d,par.lambda);
            Y = Mat::Zero(d,par.lambda);
            X = Mat::Zero(par.n, par.lambda);
            // sample N(0,C)
            for(int k = 0; k < par.lambda; k++){
                Z.col(k) = rng.gauss_vec(d);
                Y.col(k) = B_C * (D_C.asDiagonal()* Z.col(k));
            
                // lift to tangent vector
                Vec v = E * Y.col(k);

                // retract
                Eigen::ArrayXd s = sigma * v.array() / m.array();   
                Vec ex = ((s - s.maxCoeff()).exp() * m.array()).matrix();  // m * exp(s - smax)
                X.col(k) = R * ex / ex.sum(); 

            }
        }

        void update(){
            // select mu best offspring
            // 1. evaluate objective function for all members of population
            Vec H_x(par.lambda);
            for(int k = 0; k < par.lambda; k++)
                H_x(k) = obj(X.col(k));

            //sort
            std::vector<int> inds(par.lambda);
            std::iota(inds.begin(), inds.end(), 0);
            std::sort(inds.begin(), inds.end(), [&](int a, int b) {
                return H_x(a) < H_x(b); 
            });

            counteval += par.lambda;
            // store best X, H(X)
            if(H_x(inds[0]) < H_best){
                x_best = X.col(inds[0]);
                H_best = H_x(inds[0]);
            }

            //fitness history
            if(fhist.size() >= hist_cap) fhist.pop_front();
            fhist.push_back(H_best);

            // store old m
            Vec m_old = m; 

            Vec zmean = Vec::Zero(d);
            Vec ymean = Vec::Zero(d);
            for (int k = 0; k < par.mu; ++k){
                zmean += par.weights(k) * Z.col(inds[k]);
                ymean += par.weights(k) * Y.col(inds[k]);
            }

            // update mean
            Vec vmean = E * ymean;
            Eigen::ArrayXd s = par.cm * sigma * vmean.array() / m.array();    // exponents
            Vec ex = ((s - s.maxCoeff()).exp() * m.array()).matrix();         // m * exp(s - smax)
            m = R * ex / ex.sum();

            if (m.minCoeff() < boundary_floor) {     // e.g. boundary_floor = 1e-10
                pending_stop = "Boundary reached";
                return;
            }

            assert(std::abs(m.sum() - R) < 1e-12);

            csa(zmean);

            // cov update
            double h_sigma = p_sigma.norm()/std::sqrt(1-std::pow(1-par.csigma,2*(gen+1))) < (1.4+2.0/(d+1))*EN ? 1 : 0;
            double delta_h_sigma = (1-h_sigma)*par.cc*(2.0-par.cc);
            p_c = (1.0-par.cc)*p_c + h_sigma*std::sqrt(par.cc*(2-par.cc)*par.mueff)*ymean;

            Mat rank_mu = Mat::Zero(d, d);
            for (int k = 0; k < par.mu; ++k)
                rank_mu += par.weights(k) * Y.col(inds[k]) * Y.col(inds[k]).transpose();

            C = (1.0 + par.c1*delta_h_sigma - par.c1 - par.cmu) * C
                + par.c1 * (p_c * p_c.transpose())
                + par.cmu * rank_mu;

            Mat E_old = E;

            E = build_frame();

            transport(m_old, E_old);

            Eigen::SelfAdjointEigenSolver<Mat> es(0.5*(C + C.transpose()));   // symmetrize first
            B_C = es.eigenvectors();
            D_C = es.eigenvalues().cwiseMax(0.0).cwiseSqrt();                 // sqrt-eigenvalues, guard tiny negatives

            gen++;
        }

        void csa(const Vec& zmean){
            // cumulative step-size adaptation
            p_sigma = (1-par.csigma)*p_sigma+std::sqrt(par.csigma*(2-par.csigma)*par.mueff)*(B_C*zmean);

            double fac = par.csigma/par.dsigma;
            sigma = sigma*std::exp(fac*(p_sigma.norm()/EN-1.0)); 
        }

        // call after m has been updated, needs the old mean and old frame
        void transport(const Vec& m_old, const Mat& E_old) {
            Vec pu = (m_old / R).cwiseSqrt();          // unit sphere pt (old)
            Vec qu = (m     / R).cwiseSqrt();          // unit sphere pt (new) — m is m_new now
            double denom = 1.0 + pu.dot(qu);

            Vec rs_old = m_old.cwiseSqrt();            // sqrt(m_old)
            Vec rs_new = m.cwiseSqrt();                // sqrt(m_new)

            Mat TE(par.n, d);                          // tau applied to each column of E_old
            for (int j = 0; j < d; ++j) {
                Vec w  = E_old.col(j).array() / rs_old.array();         // -> sphere tangent
                Vec wn = w - (qu.dot(w) / denom) * (pu + qu);            // sphere PT
                TE.col(j) = (wn.array() * rs_new.array()).matrix();      // -> simplex tangent @ m_new
            }

            Mat Q = E.transpose() * (m.cwiseInverse().asDiagonal() * TE); // E_new^T G_new TE

            assert((Q.transpose() * Q - Mat::Identity(d,d)).norm() < 1e-6);

            // move into the new frame
            p_c     = Q * p_c;
            p_sigma = Q * p_sigma;
            C       = Q * C * Q.transpose();
        }

    public:

        // struct to store result
        struct Result {
            Vec x_best;
            double H_best = INFINITY;
            int gens;
            long evals;
            std::string out_message;
        };

        Result optimize(int max_gen, long budget = 10000000){
            feval_budget = budget;
            std::string out_message;
            int g = 0;
            for(; g < max_gen; g++){
                step();
                if(!(out_message = check_termination()).empty()) break; //
            }   
            if(out_message.empty()) out_message = "Reaching max. # of iters";
            return {x_best, H_best, g, counteval, out_message};
        }

        bool   history_full() const { return (int)fhist.size() >= hist_cap; }

        double fitness_range() const {
            if (fhist.empty()) return INFINITY;
            auto mm = std::minmax_element(fhist.begin(), fhist.end());
            return *mm.second - *mm.first;
        }

        // run one generation: sample a population, then select + update mean, transport
        void step() { sample(); update(); }

        // read-only access to current state
        const Vec& mean()      const { return m; }
        double     step_size() const { return sigma; }
        // ratio of longest to shortest principal axis of C (1 = isotropic)
        double     cov_axis_ratio() const { return D_C.maxCoeff() / D_C.minCoeff(); }

        // returns the three orthonormality residuals from the note
        void check_frame() const {
            Mat G = m.cwiseInverse().asDiagonal();
            double orth = (E.transpose() * G * E - Mat::Identity(d,d)).norm();
            double tang = (Eigen::RowVectorXd::Ones(par.n) * E).norm();  // 1^T E
            double cent = (E.transpose() * E - (R/par.n)*Mat::Identity(d,d)).norm();
            std::cout << "||E^T G E - I||   = " << orth << "\n";
            std::cout << "||1^T E||         = " << tang << "\n";
            std::cout << "||E^T E - (R/n)I|| = " << cent << "\n";
        }

        // draw a population and check the simplex invariants
        void check_sample() {
            sample();
            double maxsum = (X.colwise().sum().array() - R).abs().maxCoeff();
            double minx   = X.minCoeff();
            Vec    cmean  = X.rowwise().mean();          // empirical mean over offspring
            std::cout << "max|colsum - R|  = " << maxsum << "   (should be ~0)\n";
            std::cout << "min(X)           = " << minx   << "   (should be > 0)\n";
            std::cout << "||colmean - m||  = " << (cmean - m).norm()
                      << "   (small for small sigma / large lambda)\n";
        }

        std::string check_termination() {
            if (!pending_stop.empty()) return pending_stop;          // boundary bail-out
            if (fitness_range() < 1e-12 && history_full()) return "TolFun";
            if (sigma * D_C.maxCoeff() < 1e-12 * sigma0)   return "TolX";
            if (sigma * D_C.maxCoeff() > 1e4  * sigma0)    return "TolUpSigma";
            if (std::pow(cov_axis_ratio(),2) > 1e14)       return "ConditionCov";
            if (counteval >= feval_budget)                 return "MaxFEvals";
            return "";
        }


};