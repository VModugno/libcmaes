/**
 * (1+1)CMA-ES with constraints, Covariance Matrix Adaptation Evolution Strategy
 * Copyright (c) 2019
 * Author: Valerio Modugno <valerio.modugno@gmail.com>
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

#ifndef ONEPLUSONECMASTRATEGY_H
#define ONEPLUSONECMASTRATEGY_H

#include "esostrategy.h"
#include "cmaparameters.h"
#include "cmasolutions.h"
#include "cmastopcriteria.h"
#include "covarianceupdate.h"
#include "acovarianceupdate.h"
#include "vdcmaupdate.h"
#include "eigenmvn.h"
#include <fstream>

namespace libcmaes
{

  /**
   * \brief This is an implementation of 1+1CMA-ES with constrained covariance optimization. It uses the reference algorithm
   *        and termination criteria of the following paper:
   *        Dirk V. Arnold and Nikolaus Hansen. 2012. A (1+1)-CMA-ES for constrained optimisation.
   *        In Proceedings of the 14th annual conference on Genetic and evolutionary computation (GECCO '12),
   *        Terence Soule (Ed.). ACM, New York, NY, USA, 297-304. DOI: https://doi.org/10.1145/2330163.2330207
   */
  template <class TCovarianceUpdate,class TGenoPheno=GenoPheno<NoBoundStrategy>>
    class CMAES_EXPORT OnePlusOneCMAStrategy : public ESOStrategy<CMAParameters<TGenoPheno>,CMASolutions,CMAStopCriteria<TGenoPheno> >
    {
    public:
      /**
       * \brief dummy constructor
       */
	  OnePlusOneCMAStrategy();

      /**
       * \brief constructor.
       * @param func objective function to minimize
       * @param parameters stochastic search parameters
       */
      OnePlusOneCMAStrategy(ConstrFitFunc &func,CMAParameters<TGenoPheno> &parameters);

      /**
       * \brief constructor for starting from an existing solution.
       * @param func objective function to minimize
       * @param parameters stochastic search parameters
       * @param cmasols solution object to start from
       */
      OnePlusOneCMAStrategy(ConstrFitFunc &func,
		                    CMAParameters<TGenoPheno> &parameters,
		                    const CMASolutions &cmasols);

      ~OnePlusOneCMAStrategy();

      /**
       * \brief generates nsols new candidate solutions, sampled from a
       *        multivariate normal distribution.
       * return A matrix whose rows contain the candidate points.
       */
      dMat ask();


      /**
	 * \brief overwrite of the eval function to deal with constraints
	 * return nothing.
	 */
      void eval(const dMat &candidates,
    	        const dMat &phenocandidates=dMat(0,0));

      /**
       * \brief Updates the covariance matrix and prepares for the next iteration.
       */
      void tell();

      /**
       * \brief Stops search on a set of termination criterias, see reference paper.
       * @return true if search must stop, false otherwise.
       */
      bool stop();

      /**
       * \brief Finds the minimum of the objective function. It makes
       *        alternate calls to ask(), tell() and stop() until
       *        one of the termination criteria triggers.
       * @param evalf custom eval function
       * @param askf custom ask function
       * @param tellf custom tell function
       * @return success or error code, as defined in opti_err.h
       * Note: the termination criteria code is held by _solutions._run_status
       */
    int optimize(const EvalFunc &evalf, const AskFunc &askf, const TellFunc &tellf);

      /**
       * \brief Finds the minimum of the objective function. It makes
       *        alternate calls to ask(), tell() and stop() until
       *        one of the termination criteria triggers.
       * @return success or error code, as defined in opti_err.h
       * Note: the termination criteria code is held by _solutions._run_status
       */
    int optimize()
    {
      return optimize(std::bind(&OnePlusOneCMAStrategy<TCovarianceUpdate,TGenoPheno>::eval,this,std::placeholders::_1,std::placeholders::_2),
		      std::bind(&OnePlusOneCMAStrategy<TCovarianceUpdate,TGenoPheno>::ask,this),
		      std::bind(&OnePlusOneCMAStrategy<TCovarianceUpdate,TGenoPheno>::tell,this));
    }

      /**
       * \brief Stream the internal state of the search into an output file,
       *        as defined in the _parameters object.
       */
      void plot();

    protected:
      Eigen::EigenMultivariateNormal<double> _esolver;  /**< multivariate normal distribution sampler, and eigendecomposition solver. */
      CMAStopCriteria<TGenoPheno> _stopcriteria; /**< holds the set of termination criteria, see reference paper. */
      std::ofstream *_fplotstream = nullptr; /**< plotting file stream, not in parameters because of copy-constructor hell. */

    public:
    static ProgressFunc<CMAParameters<TGenoPheno>,CMASolutions> _defaultPFunc; /**< the default progress function. */
    static PlotFunc<CMAParameters<TGenoPheno>,CMASolutions> _defaultFPFunc; /**< the default plot to file function. */
    };

}

#endif
