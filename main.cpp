#include<stdlib.h>
#include<iostream>
#include "params.hpp"
#include "utils.hpp"
#include "cmaes.hpp"

int main(int argc, char** argv){
    std::string cfg_path = (argc > 1) ? argv[1] : "run.cfg";
    RunConfig cfg = parse_config(cfg_path);

    std::cout<<"\n========Running from config. file ("<<cfg_path<<")========\n";
    utils::objective_func obj;
    if (cfg.generate) obj.gen_params_with_validation(cfg.n, cfg.R, cfg.seed);
    else              obj = utils::objective_func(cfg.file);

    std::cout<<obj.get_uniform_obj_val()<<"\n";

    cmaes opt(obj.nvar(), 0.2, obj.R(),
                  [&obj](const Vec& x){ return obj.H(x); },
                  cfg.seed, cfg.lambda);
    cmaes::Result res = opt.optimize(5000);
    std::cout<<"stopped: "<<res.out_message<<"\n";
    std::cout<<"  generations  = "<<res.gens<<"\n";
    std::cout<<"  evaluations  = "<<res.evals<<"\n";
    std::cout<<"  sum(x_best)  = "<<res.x_best.sum()<<"\n";
    std::cout<<"  min(x_best)  = "<<res.x_best.minCoeff()<<"\n";

    std::cout<<"Uniform objective val. = "<<obj.get_uniform_obj_val()<<"\n";
    std::cout<<"Final objective val.   = "<<res.H_best<<"\n";

    return 0;
}