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

#include "constrainedcovarianceupdate.h"
#include <iostream>

namespace libcmaes
{

  template <class TGenoPheno>
  void ConstrainedCovarianceUpdate::update(const CMAParameters<TGenoPheno> &parameters,
				                           Eigen::EigenMultivariateNormal<double> &esolver,
				                           CMASolutions &solutions)
  {
	// some constraints are violated and constraints evaluation is active

	if(solutions._violated_constrained && parameters._constraints_on){
		 //no update mean
		 //no update s ( or as is defined here pc)
		 //no update sigma
		 int index  = 0;
		 int jj     = 0;
		 dMat w     = dMat::Zero(solutions._vci.size(),parameters._dim);
		 dMat value = dMat::Zero(solutions._A.rows(),solutions._A.cols());
         dMat val1;
         double val2;
		 for(unsigned int j = 0;j<solutions._vci.size();j++){
			 jj = solutions._vci[j];
			 //update exponentially fading record vj
//             if (j==0){solutions._vc.row(jj).array()=0;}
			 solutions._vc.row(jj).array() = (1-parameters._c_constr)*solutions._vc.row(jj) + parameters._c_constr*((solutions._A*solutions._z).transpose());
			 // w(index,:) = (A{k}^(-1)*v(j,:)')';
             w.row(index).array() = solutions._A.inverse()*solutions._vc.row(jj).transpose();
			 // value = value +  (v(j,:)'*w(index,:))/(w(index,:)*w(index,:)');
             val1 = solutions._vc.row(jj).transpose()*w.row(index);
             val2 = w.row(index)*w.row(index).transpose();
             value = value + val1/(val2);
			 //
			 index = index + 1;
		 }
		 //update A if constraint violation is true
		 solutions._A = solutions._A - parameters._beta/solutions._vci.size() * value;
		 // update performance with the last best value
		 solutions._performances.push_back(solutions._performances.back());
	}
	else{  // all the constraints are satisfied
        if(solutions._candidates[0].get_fvalue() > solutions._performances.back())
			solutions._Psucc = (1- parameters._c_p)*solutions._Psucc + parameters._c_p;
		else
			solutions._Psucc = (1-parameters._c_p)*solutions._Psucc;
		// update sigma, sigma(k+1) = sigma(k)*exp( (1/d) * (P_succ - P_target) / (1-P_target) );
		solutions._sigma = solutions._sigma*std::exp( (1/parameters._d) * (solutions._Psucc - parameters._P_target) / (1-parameters._P_target) );

        if(solutions._candidates[0].get_fvalue() > solutions._performances.back()){ // performance is better
			//no update v
			dMat w     = dVec::Zero(parameters._dim);
			// update mean
			solutions._xmean = solutions._candidates[0].get_x_dvec();
			// update performance with the new best
			solutions._performances.push_back(solutions._candidates[0].get_fvalue());
			// update exponentially fading record s = _pc  s(k+1,:) = (1-c)*s(k,:) + sqrt(c*(2-c))*(A{k}*z')';
		    solutions._pc = (1 - parameters._cs)*solutions._pc + sqrt(parameters._cs*(2-parameters._cs))*(solutions._A*solutions._z);
			// w = (A{k}^(-1)*s(k+1,:)')';
			w = solutions._A.inverse()*solutions._pc;

			// update A, A{k+1} = sqrt(1 - c_cov_plus)*A{k} + ( sqrt(1-c_cov_plus)/norm(w)^2 )*(sqrt(1 + (c_cov_plus*norm(w)^2)/(1-c_cov_plus) ) - 1 )*s(k+1,:)'*w
            dMat A1 = sqrt(1 - parameters._c_cov_plus)*solutions._A;
            dMat A2 = ( sqrt(1-parameters._c_cov_plus)/w.squaredNorm() )* (sqrt(1 + (parameters._c_cov_plus*w.squaredNorm())/(1-parameters._c_cov_plus) ) - 1 )* (solutions._pc*w.transpose());
//            dMat A3 =
//            solutions._A = sqrt(1 - parameters._c_cov_plus)*solutions._A + ( sqrt(1-parameters._c_cov_plus)/w.squaredNorm() ) * (sqrt(1 + (parameters._c1*w.squaredNorm())/(1-parameters._c_cov_plus) ) - 1 )* (solutions._pc.transpose()*w);
            solutions._A = A1 + A2;
		 }
		 else{ // performance is worse
			 // no update mean
			 // no update s ( or as is defined here pc)
			 // no update vc
			 // update performance with the last best value
			 solutions._performances.push_back(solutions._performances.back());
			 if(solutions._niter>5){
                if(solutions._candidates[0].get_fvalue() > solutions._performances[solutions._niter-5]){ // performance is worse but better than the last fifth predecessor
			       // update A, A{k+1} = sqrt(1 + c_cov_minus)*A{k} + ( sqrt(1 + c_cov_minus)/norm(z)^2 )*( sqrt(1 - (c_cov_minus*norm(z)^2)/(1 + c_cov_minus)) - 1 )*A{k}*(z'*z)
                    dMat A_1 = sqrt(1 + parameters._c_cov_minus)*solutions._A;
                    dMat A_2 = ( sqrt(1 + parameters._c_cov_minus)/solutions._z.squaredNorm() )*( sqrt(1 - (parameters._c_cov_minus*solutions._z.squaredNorm()/(1 + parameters._c_cov_minus))) - 1 )*solutions._A*(solutions._z.transpose()*solutions._z);
                    solutions._A = A_1  + A_2 ;
                }
			    else{
			     // A no update covariance
			    }
			 }
			 else{
				 // A no update covariance
			 }

		 }

	}

    solutions._cov  = solutions._A*solutions._A.transpose();


  }

  // TODO fix this with the right template
//  template CMAES_EXPORT void ConstrainedCovarianceUpdate::update(const CMAParameters<GenoPheno<NoBoundStrategy> >&,Eigen::EigenMultivariateNormal<double>&,CMASolutions&);
  template CMAES_EXPORT void ConstrainedCovarianceUpdate::update(const CMAParameters<GenoPheno<pwqBoundStrategy> >&,Eigen::EigenMultivariateNormal<double>&,CMASolutions&);
//  template CMAES_EXPORT void ConstrainedCovarianceUpdate::update(const CMAParameters<GenoPheno<NoBoundStrategy,linScalingStrategy> >&,Eigen::EigenMultivariateNormal<double>&,CMASolutions&);
  template CMAES_EXPORT void ConstrainedCovarianceUpdate::update(const CMAParameters<GenoPheno<pwqBoundStrategy,linScalingStrategy> >&,Eigen::EigenMultivariateNormal<double>&,CMASolutions&);
}
