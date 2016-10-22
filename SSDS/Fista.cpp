/////////////////////////////////////////////////////////////////////////
// FISTA to computes 
//   argmin_V 0.5/LipC*|X-XV|||_F^2  +  sum_{ij} theta_{ij} ||v^i - v^j||_1
//             + sum_{ ij } rho_{ ij } || v_i - v_j ||_1
//             + lambda ||V||_{1,2}
//   Citation
//   --------
//   Hongxing Wang, Yoshinobu Kawahara, Chaoqun Weng, and Junsong Yuan,
//   "Representative Selection with Structured Sparsity",
//   Pattern Recognition, vol. 63, pp. 268 - 278, 2017,
//   http://dx.doi.org/10.1016/j.patcog.2016.10.014.
/////////////////////////////////////////////////////////////////////////

//#include <iostream>
#include <Eigen/Eigen>
#include <cmath>
//#include <matrix.h>
//#include "stdafx.h"
#include "Fista.h"
#include "proxflow.h"

//const double ERR_FT = 1e-10;
//const double ERR_DY = 1e-10;
//const int ITER_FT = 30;
//const int ITER_DY = 10;


void Prox::ProxDykastra::getFrom(Eigen::MatrixXd Z_rev)
{
	Z_ = Z_rev;
}


void Prox::Fista::Solve(Eigen::MapMatrixDouble V)
{
	//N_(N), Z_(Z), w_row_(w_row), w_col_(w_col)
	V = Eigen::MatrixXd::Zero(N_, N_);
	Eigen::MatrixXd W = Eigen::MatrixXd::Zero(N_, N_);
	Eigen::MatrixXd Z(N_, N_);
	Eigen::MatrixXd U(N_, N_);
	double t = 1.0;
	double s;
	
	typedef void (ProxDykastra::*pt2fun)(Eigen::MapMatrixDouble);	
	pt2fun prox_solve = NULL; 

	ProxDykastra prox_dykastra(lambda_l1l2_, N_, Z, w_row_, w_col_,tols_DY_);
	prox_solve = &ProxDykastra::Solve_LD_parallel;
	
	double err_FISTA = 1;
	//cout<<"\t\t\tFista test"<<endl;
	//#pragma omp parallel for
	for (int iter = 1; (iter <= tols_FT_.ITER) && (err_FISTA > tols_FT_.ERR); ++iter)
	{
		//LOG ("Fista in # %d iteratin\n", iter);
		//std::cout<<"Fista in #"<<iter<<" iteration"<<std::endl;
		Z = W + invLipC_*XTX_*(Eigen::MatrixXd::Identity(N_, N_) - W);
		U = V;
		//prox_dykastra(lambda_l1l2_, N_, Z, w_row_, w_col_);
		prox_dykastra.getFrom(Z);
		(prox_dykastra.*prox_solve)(V); 
		s = t - 1.0;
		t = 0.5*(1.0 + std::sqrt(1.0 + 4.0 * t*t));
		W = V + (s/t)*(V - U);
		err_FISTA = (V-U).cwiseAbs().sum()/(N_*N_);
	}

}

void Prox::ProxDykastra::Solve_LD_parallel(Eigen::MapMatrixDouble V)
{
	
	//N_(N), Z_(Z), w_row_(w_row), w_col_(w_col)
	V = Z_;
	Eigen::MatrixXd V_old = V;

	Eigen::MatrixXd P = Eigen::MatrixXd::Zero(N_, N_);
	Eigen::MatrixXd Q = Eigen::MatrixXd::Zero(N_, N_);
	Eigen::MatrixXd Yp = Eigen::MatrixXd::Zero(N_, N_);
	Eigen::MatrixXd Yq = Eigen::MatrixXd::Zero(N_, N_);

	Eigen::VectorXd h = Eigen::VectorXd::Zero(N_);
	Eigen::VectorXd g = Eigen::VectorXd::Zero(N_);

	double coef[2] = {0.5,0.5}; // sum must equal to 1
	double err_Dykastra = 1;
	P = V;
	Q = V;
	//#pragma omp parallel for
	for (int iter = 1; iter <= tols_DY_.ITER && err_Dykastra > tols_DY_.ERR; ++iter)
	{
		//LOG ("  Dykastra _parallel in # %d iteratin\n", iter);
		V_old = V;
		#pragma omp parallel for private(h,g)
		for (int i = 0; i < N_; ++i)
		{
			#pragma omp parallel sections
			{
				#pragma omp section
				{
					ProximalParaflow proximal_paraflow_row(N_, P.row(i),
						w_col_.num_weights, w_col_.weights,
						w_col_.max_degree, w_col_.weights_ir, w_col_.weights_jc, 1.0, 1.0/coef[0]);
					proximal_paraflow_row.ConstructGraph();

					proximal_paraflow_row.Solve();

					h = Eigen::VectorXd::Zero(N_);
					proximal_paraflow_row.CalcV(h);
					Yp.row(i) = h * ( ((1 - ((1.0/coef[0])*lambda_l1l2_ / (h.norm() + EPS))) > 0.0) ? (1 - ((1.0/coef[0])*lambda_l1l2_ / (h.norm() + EPS))) : 0.0 );
				}
				#pragma omp section
				{
					ProximalParaflow proximal_paraflow_col(N_, Q.col(i),
						w_row_.num_weights, w_row_.weights,
						w_row_.max_degree, w_row_.weights_ir, w_row_.weights_jc, 1.0, 1.0/coef[1]);
					proximal_paraflow_col.ConstructGraph();
		
					proximal_paraflow_col.Solve();
	
					g = Eigen::VectorXd::Zero(N_);
					proximal_paraflow_col.CalcV(g);
					Yq.col(i) = g;
				}
			}

		}


		V = coef[0]*Yp + coef[1]*Yq;

		P = V + P - Yp;
		Q = V + Q - Yq;
		err_Dykastra = (V - V_old).cwiseAbs().sum()/(N_*N_);
		
	}
	//V = Z_;
}

