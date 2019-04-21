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

#ifndef CONSTRCOVARIANCEUPDATE_H
#define CONSTRCOVARIANCEUPDATE_H

#include "cmaparameters.h"
#include "cmasolutions.h"
#include "eigenmvn.h"

namespace libcmaes
{

/**
   * \brief This is an implementation of 1+1CMA-ES with constrained covariance optimization. It uses the reference algorithm
   *        and termination criteria of the following paper:
   *        Dirk V. Arnold and Nikolaus Hansen. 2012. A (1+1)-CMA-ES for constrained optimisation.
   *        In Proceedings of the 14th annual conference on Genetic and evolutionary computation (GECCO '12),
   *        Terence Soule (Ed.). ACM, New York, NY, USA, 297-304. DOI: https://doi.org/10.1145/2330163.2330207
   */
  class ConstrainedCovarianceUpdate{
	  public:
		/**
		 * \brief update the covariance matrix.
		 * @param parameters current set of parameters
		 * @param esolver Eigen eigenvalue solver
		 * @param solutions currrent set of solutions.
		 */
		template <class TGenoPheno>
		static void update(const CMAParameters<TGenoPheno> &parameters,
				   Eigen::EigenMultivariateNormal<double> &esolver,
				   CMASolutions &solutions);
  };

}

#endif
