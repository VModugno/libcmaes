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

#include "covarianceupdate.h"
#include <iostream>

namespace libcmaes
{

  template <class TGenoPheno>
  void CovarianceUpdate::update(const CMAParameters<TGenoPheno> &parameters,
				Eigen::EigenMultivariateNormal<double> &esolver,
				CMASolutions &solutions)
  {


    // matlab code----------------------------------------------------------------

    /*n = size(action,2);
    //mean = zeros(nIterations, n);   got it
    //mean(1, :) = action;            got it
    //range = maxAction - minAction;  got it
    stdDev = range;
    //sigma = zeros(nIterations,1);   got it
    //sigma(1) = explorationRate;     got it
    C = diag(stdDev.^2);
    A{1} = chol(C);

    performances = zeros(1,nIterations);  // ?

    n_constraints = obj.penalty_handling.n_constraint;       // to add

    // initialization
    for j = 1 : n_constraints
	   v(j,:) = zeros(1,n);                                  // to add
    end
    V{1} = v;   %cell vector of matrix where each v is row   // to add
    s = zeros(nIterations,n);                                // ?
    d = 1 + n/2;                                             // d     to add
    c =  2/ (n + 2);                                         // c     to add
    c_p = 1/12;                                              // c_p   to add
    P_succ = 0;                                              // to add
    P_target = 2/12;                                         // to add
    c_cov_plus = 2 /(n^(2) + 6);                             // to add
    c_cov_minus = 0.4/(n^(1.6)+1);                           // to add
    c_c = 1/(n+2);                                           // to add
    beta = 0.1/(n+2);                                        // to add




	if(~isempty(violated_constrained)){ // some constraints are violated
		v = V{k};
		for (j = violated_constrained){
		   v(j,:) = (1-c_c)*V{k}(j,:) + c_c*(A{k}*z')';  //only if the constraints is violated, udpate exponentially fading record vj
		}
		V{k+1} = v;
		index = 1;
		//v(j,:); % to fix no commit without fixing
		for (j = violated_constrained){
		   w(index,:) = (A{k}^(-1)*v(j,:)')';
		   index = index + 1;
		}
		mean(k+1,:) = mean(k,:);                                                                                                                             % no update mean
		performances(k+1) = performances(k);
		costs(k+1) = -performances(k);
		s(k+1,:) = s(k,:);                                                                                                                                      % no update s
		value = zeros(size(A{1}));
		index = 1;
		for(j = violated_constrained) {
		   value = value +  (v(j,:)'*w(index,:))/(w(index,:)*w(index,:)');
		   index = index + 1;
		}
		A{k+1} = A{k} - (beta)/length(violated_constrained) * value;                                                             // update A if constraint violation is true
		sigma(k+1) = sigma(k); // no update sigma
	}
	else{ //% all the constraints are satisfied
		if(performances_new > performances(k))
			P_succ = (1-c_p)*P_succ + c_p;
		else
			P_succ = (1-c_p)*P_succ;
		sigma(k+1) = sigma(k)*exp( (1/d) * (P_succ - P_target) / (1-P_target) );                                                                                // update sigma
		if(performances_new > performances(k)){ // perfomance is better
			mean(k+1,:) = offsprings;                                                                                                                           // update mean
			performances(k+1) = performances_new;
			costs(k+1) = -performances_new;
			s(k+1,:) = (1-c)*s(k,:) + sqrt(c*(2-c))*(A{k}*z')';                                       //only if the constraints are not violated upDate exponentially fading record s
			w = (A{k}^(-1)*s(k+1,:)')';
			A{k+1} = sqrt(1 - c_cov_plus)*A{k} + ( sqrt(1-c_cov_plus)/norm(w)^2 )*(sqrt(1 + (c_cov_plus*norm(w)^2)/(1-c_cov_plus) ) - 1 )*s(k+1,:)'*w; // update A if perfor_new > perf(k)
			V{k+1} = V{k};   //no update v
		 }
		 else{ // perfomance is worse
			mean(k+1,:) = mean(k,:);                                                                                                                            // no update mean
			performances(k+1) = performances(k);
			costs(k+1) = -performances(k);
			s(k+1,:) = s(k,:);                                                                                                                                    // no update s
			V{k+1} = V{k};                                                                                                                                        // no update v
			if(k>5){
			   if(performances_new > performances(k-5)) % perfomance is worse but better than the last fifth predecessor
				  A{k+1} = sqrt(1 + c_cov_minus)*A{k} + ( sqrt(1 + c_cov_minus)/norm(z)^2 )*( sqrt(1 - (c_cov_minus*norm(z)^2)/(1 + c_cov_minus)) - 1 )*A{k}*(z'*z); // update A if perf_new>perf(k-5)
			   else
				  A{k+1} = A{k};  // A no update
			}
			else
			   A{k+1} = A{k};

		 }
	} **/



    // compute mean, Eq. (2)
    dVec xmean = dVec::Zero(parameters._dim);
    for (int i=0;i<parameters._mu;i++)
      xmean += parameters._weights[i] * solutions._candidates.at(i).get_x_dvec();
    
    // reusable variables.
    dVec diffxmean = 1.0/solutions._sigma * (xmean-solutions._xmean); // (m^{t+1}-m^t)/sigma^t
    if (solutions._updated_eigen && !parameters._sep) //TODO: shall not recompute when using gradient, as it is computed in ask.
      solutions._csqinv = esolver._eigenSolver.operatorInverseSqrt();
    else if (parameters._sep)
      solutions._sepcsqinv = solutions._sepcov.cwiseInverse().cwiseSqrt();
    
    // update psigma, Eq. (3)
    solutions._psigma = (1.0-parameters._csigma)*solutions._psigma;
    if (!parameters._sep)
      solutions._psigma += parameters._fact_ps * solutions._csqinv * diffxmean;
    else
      solutions._psigma += parameters._fact_ps * solutions._sepcsqinv.cwiseProduct(diffxmean);
    double norm_ps = solutions._psigma.norm();

    // update pc, Eq. (4)
    solutions._hsig = 0;
    double val_for_hsig = sqrt(1.0-pow(1.0-parameters._csigma,2.0*(solutions._niter+1)))*(1.4+2.0/(parameters._dim+1-parameters._fixed_p.size()))*parameters._chi;
    if (norm_ps < val_for_hsig)
      solutions._hsig = 1; //TODO: simplify equation instead.
    solutions._pc = (1.0-parameters._cc) * solutions._pc + solutions._hsig * parameters._fact_pc * diffxmean;
    dMat spc;
    if (!parameters._sep)
      spc = solutions._pc * solutions._pc.transpose();
    else spc = solutions._pc.cwiseProduct(solutions._pc);
    
    // covariance update, Eq (5).
    dMat wdiff;
    if (!parameters._sep)
      wdiff = dMat::Zero(parameters._dim,parameters._dim);
    else wdiff = dMat::Zero(parameters._dim,1);
    for (int i=0;i<parameters._mu;i++)
    {
		dVec difftmp = solutions._candidates.at(i).get_x_dvec() - solutions._xmean;
		if (!parameters._sep)
		  wdiff += parameters._weights[i] * (difftmp*difftmp.transpose());
		else wdiff += parameters._weights[i] * (difftmp.cwiseProduct(difftmp));
    }
    wdiff *= 1.0/(solutions._sigma*solutions._sigma);
    if (!parameters._sep)
    	solutions._cov = (1-parameters._c1-parameters._cmu+(1-solutions._hsig)*parameters._c1*parameters._cc*(2.0-parameters._cc))*solutions._cov + parameters._c1*spc + parameters._cmu*wdiff;
    else
    {
    	solutions._sepcov = (1-parameters._c1-parameters._cmu+(1-solutions._hsig)*parameters._c1*parameters._cc*(2.0-parameters._cc))*solutions._sepcov + parameters._c1*spc + parameters._cmu*wdiff;
    }
    
    // sigma update, Eq. (6)
    if (parameters._tpa < 2)
      solutions._sigma *= std::exp((parameters._csigma / parameters._dsigma) * (norm_ps / parameters._chi - 1.0));
    else if (solutions._niter > 0)
      solutions._sigma *= std::exp(solutions._tpa_s / parameters._dsigma);
    
    // set mean.
    if (parameters._tpa)
      solutions._xmean_prev = solutions._xmean;
    solutions._xmean = xmean;
  }

  template CMAES_EXPORT void CovarianceUpdate::update(const CMAParameters<GenoPheno<NoBoundStrategy> >&,Eigen::EigenMultivariateNormal<double>&,CMASolutions&);
  template CMAES_EXPORT void CovarianceUpdate::update(const CMAParameters<GenoPheno<pwqBoundStrategy> >&,Eigen::EigenMultivariateNormal<double>&,CMASolutions&);
  template CMAES_EXPORT void CovarianceUpdate::update(const CMAParameters<GenoPheno<NoBoundStrategy,linScalingStrategy> >&,Eigen::EigenMultivariateNormal<double>&,CMASolutions&);
  template CMAES_EXPORT void CovarianceUpdate::update(const CMAParameters<GenoPheno<pwqBoundStrategy,linScalingStrategy> >&,Eigen::EigenMultivariateNormal<double>&,CMASolutions&);
}
