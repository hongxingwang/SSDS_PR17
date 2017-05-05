/////////////////////////////////////////////////////////////////////////
// FISTA to compute 
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

#ifndef FISTA_INCLUDE_FISTA_H_
#define FISTA_INCLUDE_FISTA_H_

//#include "Graph.h"
//#include "Solver.h"
//#include <iostream>
#include <Eigen/Eigen>
#include "matrix.h"
#include <omp.h>
#include <mex.h>

//#define EPS 1.0E-10  /** allowed computational error bound **/


//#ifndef NDEBUG
//#define LOG(...) 
//#else
//#define LOG(...) printf(__VA_ARGS__)
//#endif

#define LOG(...) printf(__VA_ARGS__)


namespace Eigen {
	template<typename T> Map<Matrix<T, Dynamic, Dynamic, ColMajor>> matlab2Eigen(const mxArray * pMat, bool needTranspose = true) {
		Map< Matrix<T, Dynamic, Dynamic, ColMajor>> matrixMap((T*)mxGetPr(pMat), mxGetM(pMat), mxGetN(pMat));
		return matrixMap;
	}
	typedef Map< Matrix<double, Dynamic, Dynamic, ColMajor>> MapMatrixDouble;
}



//#pragma pack(1)
namespace Prox
{
	struct weight;
	class Fista;
	class ProxDykastra;

	struct weight{
		size_t num_weights;
		double max_degree;
		double * weights;
		size_t* weights_ir;
		size_t* weights_jc;
	};

	struct tolerance{
		int ITER;
		double ERR;	
	};


	class Fista{
		//friend class ProxDykastra;
	public:
		Fista (const double lambda_l1l2, const double invLipC, const size_t N, const Eigen::MapMatrixDouble XTX,
			const weight w_row, const weight w_col, const tolerance tols_FT, const tolerance tols_DY)
			:lambda_l1l2_(lambda_l1l2), invLipC_(invLipC), N_(N), XTX_(XTX), 
			w_row_(w_row), w_col_(w_col), tols_FT_(tols_FT), tols_DY_(tols_DY){
			;
		};
		//void Solve1(Eigen::MapMatrixDouble V); // a test function
		void Solve(Eigen::MapMatrixDouble V);
		//void ProxDykastra(Eigen::MapMatrixDouble Z,Eigen::MapMatrixDouble &V);
		//void dykastra(Eigen::MatrixXd &V);

	private:
		const double lambda_l1l2_;
		const double invLipC_;
		const size_t N_;
		const Eigen::MatrixXd XTX_;
		const weight  w_row_;
		const weight  w_col_;
		const tolerance tols_FT_;
		const tolerance tols_DY_;
	};
	

	class ProxDykastra{
		//friend class Fista;
	public:
		//ProxDykastra();
		ProxDykastra(
			const double lambda_l1l2, const size_t N, Eigen::MatrixXd Z,
			const weight w_row, const weight w_col, const tolerance tols_DY)
			: lambda_l1l2_(lambda_l1l2), N_(N), Z_(Z), w_row_(w_row), w_col_(w_col), tols_DY_(tols_DY){
			;
		};
		void getFrom(Eigen::MatrixXd Z_rev);
		void Solve_LD_parallel(Eigen::MapMatrixDouble V);	

	private:
		const double lambda_l1l2_;
		const size_t N_;
		Eigen::MatrixXd Z_;
		const weight  w_row_;
		const weight  w_col_;
		const tolerance tols_DY_;
	};
	
}


//
//#pragma pack()

#endif  // PROXFLOW_INCLUDE_PROXFLOW_H_
