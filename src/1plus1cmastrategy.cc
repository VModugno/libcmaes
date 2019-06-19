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


#include "libcmaes_config.h"
#include "1plus1cmastrategy.h"
#include "opti_err.h"
#include "llogging.h"
#include <iostream>
#include <chrono>
#include <ctime>
#include <iomanip>

namespace libcmaes
{

  template <class TGenoPheno> using eostrat = ESOStrategy<CMAParameters<TGenoPheno>,CMASolutions,CMAStopCriteria<TGenoPheno> >;

  template <class TCovarianceUpdate, class TGenoPheno>
  int pfuncdef_impl(const CMAParameters<TGenoPheno> &cmaparams, const CMASolutions &cmasols)
  {
    LOG_IF(INFO,!cmaparams.quiet()) << std::setprecision(std::numeric_limits<double>::digits10) << "iter=" << cmasols.niter() << " / evals=" << cmasols.fevals() << " / f-value=" << cmasols.best_candidate().get_fvalue() <<  " / sigma=" << cmasols.sigma() << " / last_iter=" << cmasols.elapsed_last_iter() << std::endl;
    return 0;
  }
  template <class TCovarianceUpdate, class TGenoPheno>
  ProgressFunc<CMAParameters<TGenoPheno>,CMASolutions> OnePlusOneCMAStrategy<TCovarianceUpdate,TGenoPheno>::_defaultPFunc = &pfuncdef_impl<TCovarianceUpdate,TGenoPheno>;

  template<class TCovarianceUpdate, class TGenoPheno>
  int fpfuncdef_full_impl(const CMAParameters<TGenoPheno> &cmaparams, const CMASolutions &cmasols, std::ofstream &fplotstream)
  {
    std::string sep = " ";
    if (cmasols.niter() == 0)
      {
	std::chrono::time_point<std::chrono::system_clock> tnow = std::chrono::system_clock::now();
	std::time_t tdate = std::chrono::system_clock::to_time_t(tnow);
	fplotstream << cmaparams.dim() << sep << cmaparams.get_seed() << " / " << std::ctime(&tdate) << std::endl; // date and seed
      }
    fplotstream << fabs(cmasols.best_candidate().get_fvalue()) << sep << cmasols.fevals() << sep << cmasols.sigma() << sep << (cmasols.min_eigenv() == 0 ? 1.0 : sqrt(cmasols.max_eigenv()/cmasols.min_eigenv())) << sep;
    fplotstream << cmasols.get_best_seen_candidate().get_fvalue() << sep << cmasols.get_candidate(cmasols.size() / 2).get_fvalue() << sep << cmasols.get_worst_seen_candidate().get_fvalue() << sep << cmasols.min_eigenv() << sep << cmasols.max_eigenv() << sep; // best ever fvalue, median fvalue, worst fvalue, max eigen, min eigen
    if (cmasols.get_best_seen_candidate().get_x_size())
      fplotstream << cmasols.get_best_seen_candidate().get_x_dvec().transpose() << sep;  // xbest
    else fplotstream << dVec::Zero(cmaparams.dim()).transpose() << sep;
    if (!cmasols.eigenvalues().size())
      fplotstream << dVec::Zero(cmaparams.dim()).transpose() << sep;
    else fplotstream << cmasols.eigenvalues().transpose() << sep;
    fplotstream << cmasols.stds(cmaparams).transpose() << sep;
    fplotstream << cmaparams.get_gp().pheno(cmasols.xmean()).transpose();
    fplotstream << sep << cmasols.elapsed_last_iter();
#ifdef HAVE_DEBUG
    fplotstream << sep << cmasols._elapsed_eval << sep << cmasols._elapsed_ask << sep << cmasols._elapsed_tell << sep << cmasols._elapsed_stop;
#endif
    fplotstream << std::endl;
    return 0;
    }
  template<class TCovarianceUpdate, class TGenoPheno>
  int fpfuncdef_impl(const CMAParameters<TGenoPheno> &cmaparams, const CMASolutions &cmasols, std::ofstream &fplotstream)
  {
    std::string sep = " ";
    fplotstream << fabs(cmasols.best_candidate().get_fvalue()) << sep << cmasols.fevals() << sep << cmasols.sigma() << sep << (cmasols.min_eigenv() == 0 ? 1.0 : sqrt(cmasols.max_eigenv()/cmasols.min_eigenv())) << sep;
    if (!cmasols.eigenvalues().size())
      fplotstream << dVec::Zero(cmaparams.dim()).transpose() << sep;
    else fplotstream << cmasols.eigenvalues().transpose() << sep;
    fplotstream << cmasols.stds(cmaparams).transpose() << sep;
    fplotstream << cmaparams.get_gp().pheno(cmasols.xmean()).transpose();
    fplotstream << sep << cmasols.elapsed_last_iter();
#ifdef HAVE_DEBUG
    fplotstream << sep << cmasols._elapsed_eval << sep << cmasols._elapsed_ask << sep << cmasols._elapsed_tell << sep << cmasols._elapsed_stop;
#endif
    fplotstream << std::endl;
    return 0;
    }
  template<class TCovarianceUpdate, class TGenoPheno>
  PlotFunc<CMAParameters<TGenoPheno>,CMASolutions> OnePlusOneCMAStrategy<TCovarianceUpdate,TGenoPheno>::_defaultFPFunc = &fpfuncdef_impl<TCovarianceUpdate,TGenoPheno>;

  template <class TCovarianceUpdate, class TGenoPheno>
  OnePlusOneCMAStrategy<TCovarianceUpdate,TGenoPheno>::OnePlusOneCMAStrategy()
    :ESOStrategy<CMAParameters<TGenoPheno>,CMASolutions,CMAStopCriteria<TGenoPheno> >()
  {
  }

  template <class TCovarianceUpdate, class TGenoPheno>
  OnePlusOneCMAStrategy<TCovarianceUpdate,TGenoPheno>::OnePlusOneCMAStrategy(ConstrFitFunc &func,CMAParameters<TGenoPheno> &parameters)
    :ESOStrategy<CMAParameters<TGenoPheno>,CMASolutions,CMAStopCriteria<TGenoPheno> >(func,parameters)
  {

    eostrat<TGenoPheno>::_pfunc = _defaultPFunc;
    if (!parameters._full_fplot)
      eostrat<TGenoPheno>::_pffunc = _defaultFPFunc;
    else eostrat<TGenoPheno>::_pffunc = &fpfuncdef_full_impl<TCovarianceUpdate,TGenoPheno>;
    _esolver = Eigen::EigenMultivariateNormal<double>(false,eostrat<TGenoPheno>::_parameters._seed); // seeding the multivariate normal generator.
    LOG_IF(INFO,!eostrat<TGenoPheno>::_parameters._quiet) << "CMA-ES / dim=" << eostrat<TGenoPheno>::_parameters._dim << " / lambda=" << eostrat<TGenoPheno>::_parameters._lambda << " / sigma0=" << eostrat<TGenoPheno>::_solutions._sigma << " / mu=" << eostrat<TGenoPheno>::_parameters._mu << " / mueff=" << eostrat<TGenoPheno>::_parameters._muw << " / c1=" << eostrat<TGenoPheno>::_parameters._c1 << " / cmu=" << eostrat<TGenoPheno>::_parameters._cmu << " / tpa=" << (eostrat<TGenoPheno>::_parameters._tpa==2) << " / threads=" << Eigen::nbThreads() << std::endl;
    if (!eostrat<TGenoPheno>::_parameters._fplot.empty())
    {
		_fplotstream = new std::ofstream(eostrat<TGenoPheno>::_parameters._fplot);
		_fplotstream->precision(std::numeric_limits<double>::digits10);
    }
    auto mit=eostrat<TGenoPheno>::_parameters._stoppingcrit.begin();
    while(mit!=eostrat<TGenoPheno>::_parameters._stoppingcrit.end())
    {
		_stopcriteria.set_criteria_active((*mit).first,(*mit).second);
		++mit;
    }

     // initialize 0 mean and identity covariance for sampling
     dVec init_mean(eostrat<TGenoPheno>::_parameters._dim);
//     _esolver.setMean(init_mean);
     _esolver.setCovar(Eigen::MatrixXd::Identity(eostrat<TGenoPheno>::_parameters._dim,eostrat<TGenoPheno>::_parameters._dim));

     // initialize _A matrix using range
     Eigen::MatrixXd stdev=Eigen::MatrixXd::Identity(eostrat<TGenoPheno>::_parameters._dim,eostrat<TGenoPheno>::_parameters._dim);
     dVec range = eostrat<TGenoPheno>::_parameters.get_gp().get_boundstrategy().getRange();
     for (int i=0;i<eostrat<TGenoPheno>::_parameters._dim;++i){
         init_mean(i)= 0.0;
         stdev(i,i) = range(i)*range(i);
     }
     _esolver.setMean(init_mean);
     Eigen::LLT<Eigen::MatrixXd> lltOfA(stdev);
     eostrat<TGenoPheno>::_solutions._A = lltOfA.matrixL();
     //_num_constraints = num_constraints;
     //eostrat<TGenoPheno>::_solutions._num_constraints = num_constraints;
     // initialize constraints_violation array with number of constraints
     if(eostrat<TGenoPheno>::_parameters._constraints_on){
    	 eostrat<TGenoPheno>::_solutions._constraints_violations = new double [eostrat<TGenoPheno>::_parameters._Nconstr];
         // initialize exponentially fading record
         eostrat<TGenoPheno>::_solutions._vc = Eigen::MatrixXd::Zero(eostrat<TGenoPheno>::_parameters._Nconstr,eostrat<TGenoPheno>::_parameters._dim);
     }


  }
  //TODO to check this constructor, maybe missing elements
  template <class TCovarianceUpdate, class TGenoPheno>
  OnePlusOneCMAStrategy<TCovarianceUpdate,TGenoPheno>::OnePlusOneCMAStrategy(ConstrFitFunc &func,
							 CMAParameters<TGenoPheno> &parameters,
                             const CMASolutions &cmasols)
    :ESOStrategy<CMAParameters<TGenoPheno>,CMASolutions,CMAStopCriteria<TGenoPheno> >(func,parameters,cmasols)
  {
    eostrat<TGenoPheno>::_pfunc = _defaultPFunc;
    if (!parameters._full_fplot)
      eostrat<TGenoPheno>::_pffunc = _defaultFPFunc;
    else eostrat<TGenoPheno>::_pffunc = &fpfuncdef_full_impl<TCovarianceUpdate,TGenoPheno>;
    _esolver = Eigen::EigenMultivariateNormal<double>(false,eostrat<TGenoPheno>::_parameters._seed); // seeding the multivariate normal generator.
    LOG_IF(INFO,!eostrat<TGenoPheno>::_parameters._quiet) << "CMA-ES / dim=" << eostrat<TGenoPheno>::_parameters._dim << " / lambda=" << eostrat<TGenoPheno>::_parameters._lambda << " / sigma0=" << eostrat<TGenoPheno>::_solutions._sigma << " / mu=" << eostrat<TGenoPheno>::_parameters._mu << " / mueff=" << eostrat<TGenoPheno>::_parameters._muw << " / c1=" << eostrat<TGenoPheno>::_parameters._c1 << " / cmu=" << eostrat<TGenoPheno>::_parameters._cmu << " / lazy_update=" << eostrat<TGenoPheno>::_parameters._lazy_update << std::endl;
    if (!eostrat<TGenoPheno>::_parameters._fplot.empty())
      _fplotstream = new std::ofstream(eostrat<TGenoPheno>::_parameters._fplot);
    // initialize constraints_violation array with number of constraints
    if(eostrat<TGenoPheno>::_parameters._constraints_on){
    	eostrat<TGenoPheno>::_solutions._constraints_violations = new double [eostrat<TGenoPheno>::_parameters._Nconstr];
        // initialize exponentially fading record
        eostrat<TGenoPheno>::_solutions._vc = Eigen::MatrixXd::Zero(eostrat<TGenoPheno>::_parameters._Nconstr,eostrat<TGenoPheno>::_parameters._dim);
    }

  }

  template <class TCovarianceUpdate, class TGenoPheno>
  OnePlusOneCMAStrategy<TCovarianceUpdate,TGenoPheno>::~OnePlusOneCMAStrategy()
  {
    if (!eostrat<TGenoPheno>::_parameters._fplot.empty())
      delete _fplotstream;
  }

  template <class TCovarianceUpdate, class TGenoPheno>
  dMat OnePlusOneCMAStrategy<TCovarianceUpdate,TGenoPheno>::ask()
  {
  	  #ifdef HAVE_DEBUG
		std::chrono::time_point<std::chrono::system_clock> tstart = std::chrono::system_clock::now();
	  #endif


	//DEBUG
	std::cout<< "current mean = "<< eostrat<TGenoPheno>::_solutions._xmean << std::endl;
    // sampling from current distribution
    dVec pop_unbounded;

    eostrat<TGenoPheno>::_solutions._z = _esolver.samples(1,1.0);
    pop_unbounded                      = eostrat<TGenoPheno>::_solutions._xmean + eostrat<TGenoPheno>::_solutions._sigma*eostrat<TGenoPheno>::_solutions._A*eostrat<TGenoPheno>::_solutions._z; // Eq. (1)
    dVec pop = pop_unbounded;
    //debug
    std::cout << "pop = " << pop << std::endl;
    // checking bounds (in principle we should use the geno pheno mechanism for now this one is ok)
    for (int i=0;i<pop_unbounded.size();++i ){
        double lb =eostrat<TGenoPheno>::_parameters.get_gp().get_boundstrategy().getLBound(i);
        double ub =eostrat<TGenoPheno>::_parameters.get_gp().get_boundstrategy().getUBound(i);
        if (pop_unbounded(i) < lb  ){
            pop(i) = lb;
        }
        else if (pop_unbounded(i) > ub){
            pop(i) = ub;
        }
    }

    // if some parameters are fixed, reset them. (for now not used)
    if (!eostrat<TGenoPheno>::_parameters._fixed_p.empty())
    {
		for (auto it=eostrat<TGenoPheno>::_parameters._fixed_p.begin();
			 it!=eostrat<TGenoPheno>::_parameters._fixed_p.end();++it)
		{
			pop.block((*it).first,0,1,pop.cols()) = dVec::Constant(pop.cols(),(*it).second).transpose();
		}
    }

#ifdef HAVE_DEBUG
    std::chrono::time_point<std::chrono::system_clock> tstop = std::chrono::system_clock::now();
    eostrat<TGenoPheno>::_solutions._elapsed_ask = std::chrono::duration_cast<std::chrono::milliseconds>(tstop-tstart).count();
#endif
    return pop;
  }

  template<class TCovarianceUpdate, class TGenoPheno>
    void OnePlusOneCMAStrategy<TCovarianceUpdate,TGenoPheno>::eval(const dMat &candidates,const dMat &phenocandidates)
  {
  	#ifdef HAVE_DEBUG
  		std::chrono::time_point<std::chrono::system_clock> tstart = std::chrono::system_clock::now();
  	#endif
      // one candidate per row.
      for (int r=0;r<candidates.cols();r++)
      {
    	  eostrat<TGenoPheno>::_solutions._candidates.at(r).set_x(candidates.col(r));
    	  eostrat<TGenoPheno>::_solutions._candidates.at(r).set_id(r);
    	  //DEBUG
    	  // todo check this candidate phenocandidates things
    	  //std::cout << "phenocandidates = "<<phenocandidates << std::endl;

    	//for now i comment this line because i do nto understand the bound strategy implemented in the software
  		//if (phenocandidates.size())
        //    eostrat<TGenoPheno>::_solutions._candidates.at(r).set_fvalue(eostrat<TGenoPheno>::_cfunc(phenocandidates.col(r).data(),candidates.rows(),eostrat<TGenoPheno>::_solutions._constraints_violations));
        //else{
            eostrat<TGenoPheno>::_solutions._candidates.at(r).set_fvalue(eostrat<TGenoPheno>::_cfunc(candidates.col(r).data(),candidates.rows(),eostrat<TGenoPheno>::_solutions._constraints_violations));
        //}
            //DEBUG
            std::cout <<"current perfomance = " <<eostrat<TGenoPheno>::_solutions._candidates.at(r).get_fvalue() << std::endl;

      }

      // updating the support structure for covariance matrix update
      eostrat<TGenoPheno>::_solutions._violated_constrained = false;
      eostrat<TGenoPheno>::_solutions._vci.clear();
      for(int i=0;i <this->_parameters._Nconstr;++i){
		  double testVal = eostrat<TGenoPheno>::_solutions._constraints_violations[i];
		  if(testVal > -0.00){
			  eostrat<TGenoPheno>::_solutions._violated_constrained = true;
			  eostrat<TGenoPheno>::_solutions._vci.push_back(i);
		   }

	  }

      int nfcalls = candidates.cols();
      eostrat<TGenoPheno>::update_fevals(nfcalls);
	  #ifdef HAVE_DEBUG
		  std::chrono::time_point<std::chrono::system_clock> tstop = std::chrono::system_clock::now();
		  _solutions._elapsed_eval = std::chrono::duration_cast<std::chrono::milliseconds>(tstop-tstart).count();
	  #endif
  }




  template <class TCovarianceUpdate, class TGenoPheno>
  void OnePlusOneCMAStrategy<TCovarianceUpdate,TGenoPheno>::tell()
  {
    //debug
    //DLOG(INFO) << "tell()\n";
    //debug

#ifdef HAVE_DEBUG
    std::chrono::time_point<std::chrono::system_clock> tstart = std::chrono::system_clock::now();
#endif
    // sort candidates.

    // call on tpa computation of s(t)
    //if (eostrat<TGenoPheno>::_parameters._tpa == 2 && eostrat<TGenoPheno>::_niter > 0)
    //  eostrat<TGenoPheno>::tpa_update();

    // update function value history, as needed.

    // CMA-ES update, depends on the selected 'flavor'.
//    eostrat<TGenoPheno>::_parameters._constraints_on = true;

    // TODO to fix by putting the value _constraints_on inside the cmaparameters or esostrategy
    //if(eostrat<TGenoPheno>::_solutions._constraints_violations != NULL){
    //    eostrat<TGenoPheno>::_parameters._constraints_on = true;
    //}
    //else{
    //    eostrat<TGenoPheno>::_parameters._constraints_on = false;
    //}
    //-----

    // TODO fix this
    //debug
    std::cout << "flag constraints violation = "<< eostrat<TGenoPheno>::_solutions._violated_constrained <<std::endl;
    if(!eostrat<TGenoPheno>::_solutions._violated_constrained)
    	eostrat<TGenoPheno>::_solutions.update_best_candidates();
    //-----
    //eostrat<TGenoPheno>::_solutions.update_1plus1_sol_params(_num_constraints);
    TCovarianceUpdate::update(eostrat<TGenoPheno>::_parameters,_esolver,eostrat<TGenoPheno>::_solutions);


    eostrat<TGenoPheno>::_solutions.update_eigenv(_esolver._eigenSolver.eigenvalues(),_esolver._eigenSolver.eigenvectors());

#ifdef HAVE_DEBUG
    std::chrono::time_point<std::chrono::system_clock> tstop = std::chrono::system_clock::now();
    eostrat<TGenoPheno>::_solutions._elapsed_tell = std::chrono::duration_cast<std::chrono::milliseconds>(tstop-tstart).count();
#endif
  }

  template <class TCovarianceUpdate, class TGenoPheno>
  bool OnePlusOneCMAStrategy<TCovarianceUpdate,TGenoPheno>::stop()
  {
    if (eostrat<TGenoPheno>::_solutions._run_status < 0){ // an error occured, most likely out of memory at cov matrix creation.
        return true;
    }

    if (eostrat<TGenoPheno>::_pfunc(eostrat<TGenoPheno>::_parameters,eostrat<TGenoPheno>::_solutions)){ // progress function.
        return true; // end on progress function internal termination, possibly custom.
    }

    if (!eostrat<TGenoPheno>::_parameters._fplot.empty())
      plot();

    if (eostrat<TGenoPheno>::_niter == 0){
      return false;
    }
    eostrat<TGenoPheno>::_solutions._run_status = _stopcriteria.stop(eostrat<TGenoPheno>::_parameters,eostrat<TGenoPheno>::_solutions);
    int run_status = eostrat<TGenoPheno>::_solutions._run_status;
    if (run_status != CONT){
        std::cout<<"Check cmastopcriteria.h for details on exit status"<<std::endl;
        std::cout<<"1+1CMAES Current Exit status: "<<run_status<<std::endl;
        if (run_status != 7 && run_status != 1 && run_status != 10){
            std::cout<<"Chage seed and retry"<<std::endl;
        }

        return true;
    }
    else return false;
  }

  template <class TCovarianceUpdate, class TGenoPheno>
  int OnePlusOneCMAStrategy<TCovarianceUpdate,TGenoPheno>::optimize(const EvalFunc &evalf, const AskFunc &askf, const TellFunc &tellf)
  {
    //debug
    //DLOG(INFO) << "optimize()\n";
    //debug
	// initialize values for the perfomances
	// we suppose the first candidate is always feasible feasible
	{
	  	eostrat<TGenoPheno>::_solutions._initial_candidate = Candidate(eostrat<TGenoPheno>::_cfunc(eostrat<TGenoPheno>::_solutions._xmean.data(),eostrat<TGenoPheno>::_parameters._dim,eostrat<TGenoPheno>::_solutions._constraints_violations),
	  								       eostrat<TGenoPheno>::_solutions._xmean);
	  	eostrat<TGenoPheno>::_solutions._best_seen_candidate = eostrat<TGenoPheno>::_solutions._initial_candidate;
	  	//DEBUG
	  	std::cout <<"first perfomance = " <<eostrat<TGenoPheno>::_solutions._initial_candidate.get_fvalue() << std::endl;
	  	this->update_fevals(1);
	  	// check for constraints violation
	  	eostrat<TGenoPheno>::_solutions._violated_constrained = false;
	    for(int i=0;i < this->_parameters._Nconstr;++i){
			double testVal = eostrat<TGenoPheno>::_solutions._constraints_violations[i];
			if(testVal > -0.00){
				eostrat<TGenoPheno>::_solutions._violated_constrained = true;
		    }

	    }
	    // feasibility results
	    if(eostrat<TGenoPheno>::_solutions._violated_constrained)
	    	std::cout << "something wrong! the starting solutions is not feasible restart with a fesiable one!"<< std::endl;
	    else{
	    	// i update everything only if the first solutions is feasible
	    	eostrat<TGenoPheno>::_solutions._performances.push_back(eostrat<TGenoPheno>::_solutions._initial_candidate.get_fvalue());
	    	eostrat<TGenoPheno>::_solutions.update_best_candidates();
	    }

	}

    std::chrono::time_point<std::chrono::system_clock> tstart = std::chrono::system_clock::now();
    int count = 0;
    dMat candidates;

    while(!stop())
    {
        candidates = askf();

        //DEBUG
        std::cout <<"candidate"<<candidates << std::endl;

		evalf(candidates,eostrat<TGenoPheno>::_parameters._gp.pheno(candidates));
		tellf();
		eostrat<TGenoPheno>::inc_iter();
		std::chrono::time_point<std::chrono::system_clock> tstop = std::chrono::system_clock::now();
		eostrat<TGenoPheno>::_solutions._elapsed_last_iter = std::chrono::duration_cast<std::chrono::milliseconds>(tstop-tstart).count();
		tstart = std::chrono::system_clock::now();
        ++count;
    }

    if (eostrat<TGenoPheno>::_solutions._run_status >= 0)
      return OPTI_SUCCESS;
    else return OPTI_ERR_TERMINATION; // exact termination code is in eostrat<TGenoPheno>::_solutions._run_status.
  }

  template <class TCovarianceUpdate, class TGenoPheno>
  void OnePlusOneCMAStrategy<TCovarianceUpdate,TGenoPheno>::plot()
  {
    eostrat<TGenoPheno>::_pffunc(eostrat<TGenoPheno>::_parameters,eostrat<TGenoPheno>::_solutions,*_fplotstream);
  }

//  template class OnePlusOneCMAStrategy<ConstrainedCovarianceUpdate,GenoPheno<NoBoundStrategy>>;
  template class OnePlusOneCMAStrategy<ConstrainedCovarianceUpdate,GenoPheno<pwqBoundStrategy>>;
//  template class OnePlusOneCMAStrategy<ConstrainedCovarianceUpdate,GenoPheno<NoBoundStrategy,linScalingStrategy> >;
  template class OnePlusOneCMAStrategy<ConstrainedCovarianceUpdate,GenoPheno<pwqBoundStrategy,linScalingStrategy> >;

}
