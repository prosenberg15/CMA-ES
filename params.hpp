#pragma once
#include<stdlib.h>
#include<stdexcept>
#include<cmath>
#include<string>
#include<fstream>
#include<sstream>
#include <Eigen/Dense>

using Vec = Eigen::VectorXd;

// container for run config. params
struct RunConfig {
    int n = 0;
    double R = 1.0;
    bool generate = true; // true --> generate params for obj function internally, false --> read from file
    std::string file;
    unsigned seed = 42;
    int lambda = 0;   // population size; 0 = use the default from params
};

// routine to read run configuration from a key-value file
RunConfig parse_config(const std::string& fname) {
    std::ifstream in(fname);
    if (!in) throw std::runtime_error("cannot open config " + fname);
    RunConfig cfg;
    std::string line;
    while (std::getline(in, line)) {
        if (line.empty() || line[0] == '#') continue;
        std::istringstream ss(line);
        std::string key; ss >> key;
        if      (key == "n")    ss >> cfg.n;
        else if (key == "R")    ss >> cfg.R;
        else if (key == "mode") {
            std::string m; ss >> m;
            if      (m == "generate") cfg.generate = true;
            else if (m == "read")     cfg.generate = false;
            else throw std::runtime_error("config: mode must be 'generate' or 'read'");
        }
        else if (key == "file")   ss >> cfg.file;
        else if (key == "seed")   ss >> cfg.seed;
        else if (key == "lambda") ss >> cfg.lambda;
        else throw std::runtime_error("unknown config key: " + key);
    }
    // validate the combination
    if (!cfg.generate && cfg.file.empty())
        throw std::runtime_error("config: mode=read requires a 'file'");
    if (cfg.generate && cfg.n <= 0)
        throw std::runtime_error("config: mode=generate requires n > 0");
    if (cfg.lambda < 0 || cfg.lambda == 1)
         throw std::runtime_error("config: invalid lambda, requires lambda = 0 (default) or lambda >=2");
    return cfg;
}


struct params{
    int n, lambda, mu;
    Vec weights;
    double mueff;
    double cc, csigma, c1, cmu, dsigma;
    double cm = 1.0;
    double alpha_cov = 2.0;

    params(int n_, int lambda_ = 0){

        n = n_;

        // default lambda
        lambda = (lambda_ > 0) ? lambda_ : 4 + std::floor(3*std::log(n));

        // initial mu
        mu = lambda/2;

        // set initial weights
        weights = Vec::Zero(mu);

        for(int i = 0; i < mu; i++){
            weights(i) = std::log(0.5*(lambda+1))-std::log(i+1);
        }
        
        mueff = std::pow(weights.sum(),2)/weights.squaredNorm();
        if(mueff<1 || mueff>mu) throw std::runtime_error("improper mueff");

        weights/=weights.sum();

        if(std::abs(weights.sum()-1.0)>1e-10) throw std::runtime_error("improper normalization for weights");
        
        csigma = (mueff + 2.0)/(n+mueff+5.0);
        dsigma = 1.0+2.0*std::max(0.0,std::sqrt((mueff-1)/(n+1))-1)+csigma;
        cc = (4.0+mueff/n)/(n+4.0+2*mueff/n);
        c1 = alpha_cov/((n+1.3)*(n+1.3)+mueff);

        double num   = mueff+1.0/mueff-2.0;
        double denom = (n+2)*(n+2)+alpha_cov*mueff/2.0;
        cmu = std::min(1-c1,alpha_cov*(num/denom));
    }
};