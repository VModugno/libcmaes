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
#include <ctime>

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


ConstrFitFunc onePlusOneProb_g06 = [](const double *x, const int N, double* violation)
{
	//DEBUG
	std::cout <<"candidate =" <<x[0] << " " << x[1] << std::endl;
    float fitness = pow((x[0]-10),3)+pow((x[1]-20),3);
    float constr1 = - pow((x[0]-5),2)- pow((x[1]-5),2) + 100;
    float constr2 = pow((x[0]-6),2) + pow((x[1]-5),2) -82.81;
    violation[0]=constr1;
    violation[1]=constr2;
    return fitness;
};

ConstrFitFunc onePlusOneProb_tr2 = [](const double *x, const int N, double* violation)
{
    double fitness = pow((x[0]),2)+pow((x[1]),2);
    double constr1 = 2 - x[0] - x[1];
    violation[0]=constr1;
    return fitness;
};

ConstrFitFunc onePlusOneProb_g07 =[](const double *x, const int N, double* violation)
{
    double fitness = pow(x[0],2) + pow(x[1],2) + x[0]*x[1] - 14.0*x[0] -16.0*x[1] + pow((x[2]-10),2)
                 + 4*pow((x[3]-5),2) + pow((x[4]-3),2) + 2*pow((x[5]-1),2) + 5*pow(x[6],2)
                 + 7*pow((x[7]-11),2) + 2*pow((x[8]-10),2) + 1*pow((x[9]-7),2) + 45;

    violation[0] = 4*x[0] + 5*x[1] - 3*x[6] +9*x[7] - 105;
    violation[1] = 10*x[0] - 8*x[1] - 17*x[6] +2*x[7];
    violation[2] = -8*x[0] + 2*x[1] + 5*x[8] -2*x[9] - 12;
    violation[3] = -3*x[0] + 6*x[1] +12*pow((x[8]-8),2) -7*x[9];
    violation[4] = 3*pow((x[0]-2),2) + 4*pow((x[1]-3),2) + 2*pow(x[2],2) -7*x[3] - 120;
    violation[5] = pow(x[0],2) + 2*pow((x[1]-2),2) - 2*x[0]*x[1] +14*x[4] - 6*x[5];
    violation[6] = 5*pow(x[0],2) + 8*pow(x[1],2) + pow((x[2]-6),2) - 2*x[3] - 40;
    violation[7] = pow((x[0]-8),2) + 4*pow((x[1]-4),2) + 6*pow(x[4],2) -2*x[5] - 60;

    return fitness;
};

ConstrFitFunc onePlusOneProb_g09 =[](const double *x, const int N, double* violation)
{
    double fitness = pow(x[0]-10,2) + 5*pow(x[1]-12,2) + pow(x[2],4) + 3*pow(x[3]-11,2)
                  +  10*pow(x[4],6) + 7*pow(x[5],2) + pow(x[6],4) -4*x[5]*x[6] -10*x[5] -8*x[6];

    violation[0] = -127 + 2*pow(x[0],2) + 3*pow(x[1],4) + x[2] + 4*pow(x[3],2) + 5*x[4];
    violation[1] = -196 + 23*x[0] + pow(x[1],2) + 6*pow(x[5],2) -8*x[6];
    violation[2] = -282 + 7*x[0] + 3*x[1] +10*pow(x[3],2) + x[4] -x[4];
    violation[3] = 4*pow(x[0],2) + pow(x[1],2) -3*x[0]*x[1] + 2*pow(x[2],2) +5*x[5] -11*x[6];

    return fitness;
};

ConstrFitFunc onePlusOneProb_g10 =[](const double *x, const int N, double* violation)
{
    double fitness = x[0] + x[1] + x[2];

    violation[0] = 0.0025*(x[3] + x[5]) - 1;
    violation[1] = 0.0025*(x[4] + x[6] -x[3]) -1;
    violation[2] = 0.01*(x[7]-x[4]) - 1;
    violation[3] = -x[0]*x[5] + 833.33252*x[3] + 100*x[0] -83333.333;
    violation[4] = -x[1]*x[6] +1250*x[4] + x[1]*x[3] -1250*x[3];
    violation[5] = -x[2]*x[7] + 1250000 + x[2]*x[4] -2500*x[4];

    return fitness;
};

ConstrFitFunc onePlusOneProb_240 =[](const double *x, const int N, double* violation)
{
    double fitness = 0;
    violation[0] = 0;
    for (int i=0;i<5;++i){
        fitness = fitness + x[i];
        violation[0]=violation[0]+ (9+1+i)*x[i];
    }
    violation[0] = violation[0] - 50000;
    return -fitness;
};

ConstrFitFunc onePlusOneProb_241 =[](const double *x, const int N, double* violation)
{
    double fitness = 0;
    violation[0] = 0;
    for (int i=0;i<5;++i){
        fitness = fitness + x[i]*(i+1);
        violation[0]=violation[0]+ (9+1+i)*x[i];
    }
    violation[0] = violation[0] - 50000;
    return -fitness;
};

ConstrFitFunc onePlusOneProb_hb = [](const double *x, const int N, double* violation)
{
    double fitness = 5.3578547*pow(x[2],2) + 0.8356891*x[0]*x[4] + 37.293239*x[0] - 40792.141;
    double h1 = 85.334407 + 0.0056858*x[1]*x[4]+ 0.00026*x[0]*x[3] - 0.0022053*x[2]*x[4] ;
    double h2 = 80.51249 + 0.0071317*x[1]*x[4]+ 0.0029955*x[0]*x[1]+ 0.0021813*pow(x[2],2);
    double h3 = 9.300961 + 0.0047026*x[2]*x[4]+ 0.0012547*x[0]*x[2]+ 0.0019085*x[2]*x[3] ;

    violation[0] = -h1;
    violation[1] = h1 -92;
    violation[2] = 90 - h2;
    violation[3] = h2 -110 ;
    violation[4] = 20 - h3;
    violation[5] = h3 -25;

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
  double sigma = 0.1;
  int lambda = 1;
  int seed = (int)time(0);

//// 1+1cmaes paper problem g06
  const int dim_g06  = 2;
  double lbounds_g06[dim_g06] = {13.0,0.0};
  double ubounds_g06[dim_g06] = {100.0, 100.0};
  std::vector<double> opePlusOne_g06_x0 = {14.6111,2.1491};
  libcmaes:: GenoPheno<pwqBoundStrategy> gp_g06(lbounds_g06,ubounds_g06,dim_g06); // genotype / phenotype transform associated to bounds.
  CMAParameters<GenoPheno<pwqBoundStrategy>> onePlusOne_cmaparams_g06(opePlusOne_g06_x0,sigma,lambda,seed,gp_g06,true,2);
  ESOptimizer<OnePlusOneCMAStrategy<ConstrainedCovarianceUpdate,GenoPheno<pwqBoundStrategy>>,CMAParameters<GenoPheno<pwqBoundStrategy>>> onePlusOneCmaes_g06(onePlusOneProb_g06,onePlusOne_cmaparams_g06);
  onePlusOneCmaes_g06.optimize();
  std::cerr << "Solution  " <<onePlusOneCmaes_g06.get_solutions().get_best_seen_candidate().get_fvalue() << std::endl;

//// 1+1cmaes paper problem tr2
//  const int dim_tr2 = 2;
//  double lbounds_tr2[dim_tr2] = {0.0,0.0};
//  double ubounds_tr2[dim_tr2] = {100.0, 100.0};
//  std::vector<double> opePlusOne_tr2_x0 = {50,50};
//  libcmaes:: GenoPheno<pwqBoundStrategy> gp_tr2(lbounds_tr2,ubounds_tr2,dim_tr2);
//  CMAParameters<GenoPheno<pwqBoundStrategy>> onePlusOne_cmaparams_tr2(opePlusOne_tr2_x0,sigma,lambda,seed,gp_tr2,true,1);
//  ESOptimizer<OnePlusOneCMAStrategy<ConstrainedCovarianceUpdate,GenoPheno<pwqBoundStrategy>>,CMAParameters<GenoPheno<pwqBoundStrategy>>> onePlusOneCmaes_tr2(onePlusOneProb_tr2,onePlusOne_cmaparams_tr2);
//  onePlusOneCmaes_tr2.optimize();
//  std::cerr << "Solution  " <<onePlusOneCmaes_tr2.get_solutions().best_candidate().get_fvalue() << std::endl;

//// 1+1cmaes paper problem g07
//  const int dim_g07 = 10;
//  double lbounds_g07[dim_g07] = {-10,-10,-10,-10,-10,-10,-10,-10,-10,-10};
//  double ubounds_g07[dim_g07] = {10,10,10,10,10,10,10,10,10,10};
//  std::vector<double> opePlusOne_g07_x0 = {2.0,2.0,8.0,5.0,1,2.0,2.0,9.0,8.0,8.0};
//  libcmaes:: GenoPheno<pwqBoundStrategy> gp_g07(lbounds_g07,ubounds_g07,dim_g07);
//  CMAParameters<GenoPheno<pwqBoundStrategy>> onePlusOne_cmaparams_g07(opePlusOne_g07_x0,sigma,lambda,seed,gp_g07,true,8);
//  ESOptimizer<OnePlusOneCMAStrategy<ConstrainedCovarianceUpdate,GenoPheno<pwqBoundStrategy>>,CMAParameters<GenoPheno<pwqBoundStrategy>>> onePlusOneCmaes_g07(onePlusOneProb_g07,onePlusOne_cmaparams_g07);
//  onePlusOneCmaes_g07.optimize();
//  std::cerr << "Solution  " <<onePlusOneCmaes_g07.get_solutions().best_candidate().get_fvalue() << std::endl;

//// 1+1cmaes paper problem g09
//  const int dim_g09 = 7;
//  double lbounds_g09[dim_g09] = {-10,-10,-10,-10,-10,-10,-10};
//  double ubounds_g09[dim_g09] = {10,10,10,10,10,10,10};
//  std::vector<double> opePlusOne_g09_x0 = {0,0,0,0,0,0,0};
//  libcmaes:: GenoPheno<pwqBoundStrategy> gp_g09(lbounds_g09,ubounds_g09,dim_g09);
//  CMAParameters<GenoPheno<pwqBoundStrategy>> onePlusOne_cmaparams_g09(opePlusOne_g09_x0,sigma,lambda,seed,gp_g09,true,4);
//  ESOptimizer<OnePlusOneCMAStrategy<ConstrainedCovarianceUpdate,GenoPheno<pwqBoundStrategy>>,CMAParameters<GenoPheno<pwqBoundStrategy>>> onePlusOneCmaes_g09(onePlusOneProb_g09,onePlusOne_cmaparams_g09);
//  onePlusOneCmaes_g09.optimize();
//  std::cerr << "Solution  " <<onePlusOneCmaes_g09.get_solutions().best_candidate().get_fvalue() << std::endl;

// 1+1cmaes paper problem g10
//  const int dim_g10 = 8;
//  double lbounds_g10[dim_g10] = {100,1000,1000,10,10,10,10,10};
//  double ubounds_g10[dim_g10] = {10000,10000,10000,1000,1000,1000,1000,1000};
//  std::vector<double> opePlusOne_g10_x0 = {600,2000,5000,200,200,200,200,400};
//  libcmaes:: GenoPheno<pwqBoundStrategy> gp_g10(lbounds_g10,ubounds_g10,dim_g10);
//  CMAParameters<GenoPheno<pwqBoundStrategy>> onePlusOne_cmaparams_g10(opePlusOne_g10_x0,sigma,lambda,seed,gp_g10,true,6);
//  ESOptimizer<OnePlusOneCMAStrategy<ConstrainedCovarianceUpdate,GenoPheno<pwqBoundStrategy>>,CMAParameters<GenoPheno<pwqBoundStrategy>>> onePlusOneCmaes_g10(onePlusOneProb_g10,onePlusOne_cmaparams_g10);
//  onePlusOneCmaes_g10.optimize();
//  std::cerr << "Solution  " <<onePlusOneCmaes_g10.get_solutions().best_candidate().get_fvalue() << std::endl;

//// 1+1cmaes paper problem 2.40 and 2.41 (Comment/Uncomment as required)
//    const int dim_240 = 5;
//    double lbounds_240[dim_240] = {0,0,0,0,0};
//    double ubounds_240[dim_240] = {5000,5000,5000,5000,5000};
//    std::vector<double> opePlusOne_240_x0 = {250,250,250,250,250};
//    libcmaes:: GenoPheno<pwqBoundStrategy> gp_240(lbounds_240,ubounds_240,dim_240);
//    CMAParameters<GenoPheno<pwqBoundStrategy>> onePlusOne_cmaparams_240(opePlusOne_240_x0,sigma,lambda,seed,gp_240,true,1);
//    // Prob 2.40
//    ESOptimizer<OnePlusOneCMAStrategy<ConstrainedCovarianceUpdate,GenoPheno<pwqBoundStrategy>>,CMAParameters<GenoPheno<pwqBoundStrategy>>> onePlusOneCmaes_240(onePlusOneProb_240,onePlusOne_cmaparams_240);
//    // Prob 2.41
////    ESOptimizer<OnePlusOneCMAStrategy<ConstrainedCovarianceUpdate,GenoPheno<pwqBoundStrategy>>,CMAParameters<GenoPheno<pwqBoundStrategy>>> onePlusOneCmaes_240(onePlusOneProb_241,onePlusOne_cmaparams_240);
//    onePlusOneCmaes_240.optimize();
//    std::cerr << "Solution  " <<onePlusOneCmaes_240.get_solutions().best_candidate().get_fvalue() << std::endl;
////    std::cerr << "At point "<<onePlusOneCmaes_240.get_solutions().best_candidate().get_x_dvec()<<std::endl;


// 1+1cmaes paper problem hb
//  const int dim_hb = 5;
//  double lbounds_hb[dim_hb] = {78,33,27,27,27};
//  double ubounds_hb[dim_hb] = {102,45,45,45,45};
//  std::vector<double> opePlusOne_hb_x0 = {78,33,27,27,27};
//  libcmaes:: GenoPheno<pwqBoundStrategy> gp_hb(lbounds_hb,ubounds_hb,dim_hb);
//  CMAParameters<GenoPheno<pwqBoundStrategy>> onePlusOne_cmaparams_hb(opePlusOne_hb_x0,sigma,lambda,seed,gp_hb,true,6);
//  ESOptimizer<OnePlusOneCMAStrategy<ConstrainedCovarianceUpdate,GenoPheno<pwqBoundStrategy>>,CMAParameters<GenoPheno<pwqBoundStrategy>>> onePlusOneCmaes_hb(onePlusOneProb_hb,onePlusOne_cmaparams_hb);
//  onePlusOneCmaes_hb.optimize();
//  std::cerr << "Solution  " <<onePlusOneCmaes_hb.get_solutions().best_candidate().get_fvalue() << std::endl;


}
