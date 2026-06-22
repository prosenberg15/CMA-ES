#include<stdlib.h>
#include<iostream>
#include "params.hpp"
#include "utils.hpp"
#include "cmaes.hpp"

int main(int argc, char** argv){
    std::string cfg_path = (argc > 1) ? argv[1] : "run.cfg";
    RunConfig cfg = parse_config(cfg_path);

    params p1(10);

    std::cout<<"========params check========\n";
    std::cout<<"lambda       = "<<p1.lambda<<"\n";
    std::cout<<"mu           = "<<p1.mu<<"\n";
    std::cout<<"mueff        = "<<p1.mueff<<"\n";
    std::cout<<"sum(weights) = "<<p1.weights.sum()<<"\n";
    std::cout<<"c_sigma      = "<<p1.csigma<<"\n";
    std::cout<<"c_1          = "<<p1.c1<<"\n";
    std::cout<<"c_mu         = "<<p1.cmu<<"\n";
    std::cout<<"d_sigma      = "<<p1.dsigma<<"\n";

    cmaes opt(10, 0.3, 1.0, utils::sphere);

    std::cout<<"\n========frame check========\n";
    opt.check_frame();

    std::cout<<"\n========sample check========\n";
    opt.check_sample();

    // ---- one-generation mean-update check ----
    // Linear objective sum(a_i x_i): the mean should drift toward the vertex
    // with the SMALLEST coefficient (here index 1, a=1).
    std::cout<<"\n========mean update: interior optimum (40 gens)========\n";
    // shifted sphere sum (x_i - t_i)^2 with t an interior simplex point:
    // constrained optimum is exactly t, so the mean should converge to t.
    Vec t(4); t << 0.10, 0.40, 0.30, 0.20;
    auto shifted = [t](const Vec& x){ return (x - t).squaredNorm(); };
    cmaes opt_q(4, 0.05, 1.0, shifted);
    std::cout<<"target t = "<<t.transpose()<<"\n";
    std::cout<<"gen  0   ||m-t|| = "<<(opt_q.mean()-t).norm()
             <<"   sigma = "<<opt_q.step_size()<<"\n";
    for (int g = 1; g <= 80; ++g) {
        opt_q.step();
        if (g % 10 == 0)
            std::cout<<"gen "<<(g<10?" ":"")<<g<<"   ||m-t|| = "<<(opt_q.mean()-t).norm()
                     <<"   sigma = "<<opt_q.step_size()<<"\n";
    }
    std::cout<<"(with CSA, sigma should now adapt; ||m-t|| can shrink below the fixed-sigma floor)\n";

    // ---- anisotropic objective: exercises the covariance update ----
    // ill-conditioned shifted sphere sum kappa_i (x_i - t_i)^2, condition ~1000.
    // C=I would crawl (sigma limited by the stiff direction); the covariance
    // should adapt -> axis ratio grows well above 1.
    std::cout<<"\n========anisotropic: covariance adaptation========\n";
    Vec t2(4);    t2    << 0.10, 0.40, 0.30, 0.20;
    Vec kappa(4); kappa << 1.0, 1.0, 1.0, 1000.0;
    auto aniso = [t2,kappa](const Vec& x){
        return (kappa.array() * (x - t2).array().square()).sum();
    };
    cmaes opt_a(4, 0.05, 1.0, aniso);
    std::cout<<"kappa = "<<kappa.transpose()<<"   (condition number ~1000)\n";
    cmaes::Result res = opt_a.optimize(5000);
    std::cout<<"stopped: "<<res.out_message<<"\n";
    std::cout<<"  generations  = "<<res.gens<<"\n";
    std::cout<<"  evaluations  = "<<res.evals<<"\n";
    std::cout<<"  f_best       = "<<res.H_best<<"\n";
    std::cout<<"  x_best       = "<<res.x_best.transpose()<<"\n";
    std::cout<<"  target t     = "<<t2.transpose()<<"\n";
    std::cout<<"  ||x_best-t|| = "<<(res.x_best-t2).norm()<<"\n";
    std::cout<<"  C axis ratio = "<<opt_a.cov_axis_ratio()<<"\n";

    // ---- real degree-3 objective H, loaded from file ----
    std::cout<<"\n========real objective H (from file)========\n";
    utils::objective_func of("H_test.txt");
    // n and R now come straight from the loaded objective's header
    cmaes optH(of.nvar(), 0.2, of.R(), [&of](const Vec& x){ return of.H(x); });
    cmaes::Result resH = optH.optimize(5000);
    std::cout<<"stopped: "<<resH.out_message<<"\n";
    std::cout<<"  generations  = "<<resH.gens<<"\n";
    std::cout<<"  evaluations  = "<<resH.evals<<"\n";
    std::cout<<"  f_best       = "<<resH.H_best<<"\n";
    std::cout<<"  x_best       = "<<resH.x_best.transpose()<<"\n";
    std::cout<<"  sum(x_best)  = "<<resH.x_best.sum()<<"\n";

    // ---- full pipeline: generate random H -> write -> reload -> optimize ----
    std::cout<<"\n========generated random H (n=10) pipeline========\n";
    utils::objective_func gen;
    gen.generate_test_params(10, 1.0, 123);
    gen.write_params("H_random.txt");
    utils::objective_func og("H_random.txt");      // reload from disk
    cmaes optG(og.nvar(), 0.2, og.R(), [&og](const Vec& x){ return og.H(x); });
    cmaes::Result resG = optG.optimize(5000);
    std::cout<<"stopped: "<<resG.out_message<<"\n";
    std::cout<<"  generations  = "<<resG.gens<<"\n";
    std::cout<<"  evaluations  = "<<resG.evals<<"\n";
    std::cout<<"  f_best       = "<<resG.H_best<<"\n";
    std::cout<<"  sum(x_best)  = "<<resG.x_best.sum()<<"\n";
    std::cout<<"  min(x_best)  = "<<resG.x_best.minCoeff()<<"\n";

    // ---- config-driven run (the actual program entry point) ----
    std::cout<<"\n========config-driven run ("<<cfg_path<<")========\n";
    utils::objective_func cfg_obj;
    if (cfg.generate) cfg_obj.generate_test_params(cfg.n, cfg.R, cfg.seed);
    else              cfg_obj = utils::objective_func(cfg.file);

    cmaes optFull(cfg_obj.nvar(), 0.2, cfg_obj.R(),
                  [&cfg_obj](const Vec& x){ return cfg_obj.H(x); });
    cmaes::Result resFull = optFull.optimize(5000);
    std::cout<<"stopped: "<<resFull.out_message<<"\n";
    std::cout<<"  generations  = "<<resFull.gens<<"\n";
    std::cout<<"  evaluations  = "<<resFull.evals<<"\n";
    std::cout<<"  f_best       = "<<resFull.H_best<<"\n";
    std::cout<<"  sum(x_best)  = "<<resFull.x_best.sum()<<"\n";
    std::cout<<"  min(x_best)  = "<<resFull.x_best.minCoeff()<<"\n";

    return 0;
}