/**
 * CMA-ES, Covariance Matrix Adaptation Evolution Strategy
 * Copyright (c) 2014 Inria
 * Author: Emmanuel Benazera <emmanuel.benazera@lri.fr>
 *
 * This file is part of libcmaes.
 *
 * libcmaes is free software: you can redistribute it and/or modify
 * it under the terms of the GNU Lesser General Public License as published by
 * the Free Software Foundation, either version 3 of the License, or
 * (at your option) any later version.
 *
 * libcmaes is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU Lesser General Public License for more details.
 *
 * You should have received a copy of the GNU Lesser General Public License
 * along with libcmaes.  If not, see <http://www.gnu.org/licenses/>.
 */

#include "esoptimizer.h"
#include "cmastrategy.h"
#include "llogging.h"
#include "1plus1cmastrategy.h"
#include "cmaes.h"

using namespace libcmaes;

FitFunc cigtab = [](const double *x, const int N)
{
  int i;
  double sum = 1e4*x[0]*x[0] + 1e-4*x[1]*x[1];
  for(i = 2; i < N; ++i)
    sum += x[i]*x[i];
//  std::cerr<<sum<<std::endl;
  return sum;
};


ConstrFitFunc onePlusOneProb_g06 = [](const double *x, const int N, std::vector<double>& violation)
{
    double fitness = pow((x[0]-10),3)+pow((x[1]-20),3);
    double constr1 = - pow((x[0]-5),2)- pow((x[1]-6),2) + 100;
    double constr2 = pow((x[0]-6),2) + pow((x[1]-5),2) -82.81;
    violation.reserve(2);
    violation[0]=constr1;
    violation[1]=constr2;
//    std::cout<<violation[0]<<"  "<<violation[1]<<std::endl;
    return fitness;
};


int main(int argc, char *argv[])
{
#ifdef HAVE_GLOG
  google::InitGoogleLogging(argv[0]);
  FLAGS_logtostderr=1;
  google::SetLogDestination(google::INFO,"");
  //FLAGS_log_prefix=false;
#endif
  std::vector<double> x0 = {1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0};
  double sigma = 0.2;
  int lambda = 10;
  CMAParameters<> cmaparams(x0,sigma,lambda);
  ESOptimizer<CMAStrategy<CovarianceUpdate>,CMAParameters<>> cmaes(cigtab,cmaparams);
  cmaes.optimize();
  double edm = cmaes.edm();
  std::cerr << "EDM " << edm << " / EDM/fm=" << edm / cmaes.get_solutions().best_candidate().get_fvalue() << std::endl;

// 1+1cmaes paper problem g06
  int dim  = 2;
  double lbounds[dim],ubounds[dim];
  lbounds[0]=13;lbounds[1]=0;
  ubounds[0]=100;ubounds[1]=100;
  std::vector<double> opePlusOne_x0 = {20.0,20.0};
  libcmaes:: GenoPheno<pwqBoundStrategy> gp(lbounds,ubounds,dim); // genotype / phenotype transform associated to bounds.
  CMAParameters<GenoPheno<pwqBoundStrategy>> onePlusOne_cmaparams(opePlusOne_x0,0.20,1,1,gp);
//  CMAParameters<> onePlusOne_cmaparams(opePlusOne_x0,sigma,-1);
  ESOptimizer<OnePlusOneCMAStrategy<ConstrainedCovarianceUpdate,GenoPheno<pwqBoundStrategy>>,CMAParameters<GenoPheno<pwqBoundStrategy>>> onePlusOneCmaes(onePlusOneProb_g06,onePlusOne_cmaparams);
  onePlusOneCmaes.optimize();
//  double edm1 = onePlusOneCmaes.constr_edm();
  std::cerr << "Solution  " <<onePlusOneCmaes.get_solutions().best_candidate().get_fvalue() << std::endl;

  //1+1cmaes paper problem g07
//    int dim  = 2;
//    double lbounds[dim],ubounds[dim];
//    lbounds[0]=13;lbounds[1]=0;
//    ubounds[0]=100;ubounds[1]=100;
//    std::vector<double> opePlusOne_x0 = {0.0,0.0};
//    libcmaes:: GenoPheno<pwqBoundStrategy> gp(lbounds,ubounds,dim); // genotype / phenotype transform associated to bounds.
//    CMAParameters<GenoPheno<pwqBoundStrategy>> onePlusOne_cmaparams(opePlusOne_x0,0.1,-1,0,gp);
//    ESOptimizer<OnePlusOneCMAStrategy<ConstrainedCovarianceUpdate,GenoPheno<pwqBoundStrategy>>,CMAParameters<GenoPheno<pwqBoundStrategy>>> onePlusOneCmaes(onePlusOneProb_g06,onePlusOne_cmaparams);
//    onePlusOneCmaes.optimize();
//    std::cerr << "Solution  " <<onePlusOneCmaes.get_solutions().best_candidate().get_fvalue() << std::endl;
}
