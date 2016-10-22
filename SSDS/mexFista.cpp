/////////////////////////////////////////////////////////////////////////
// A mex function to compute
//   argmin_V 0.5/LipC*|X-XV|||_F^2  +  sum_{ij} theta_{ij} ||v^i - v^j||_1
//             + sum_{ ij } rho_{ ij } || v_i - v_j ||_1
//             + lambda ||V||_{1,2}
//
//   Usage: see SSDS.m 
//
//   Citation
//   --------
//   Hongxing Wang, Yoshinobu Kawahara, Chaoqun Weng, and Junsong Yuan,
//   "Representative Selection with Structured Sparsity",
//   Pattern Recognition, vol. 63, pp. 268 - 278, 2017,
//   http://dx.doi.org/10.1016/j.patcog.2016.10.014.
/////////////////////////////////////////////////////////////////////////

#include "mex.h"
#include "Fista.h"
#include <Eigen/Eigen> 
//#include<iostream>

#pragma comment(lib,"libmx.lib")  
#pragma comment(lib,"libmex.lib")  
#pragma comment(lib,"libmat.lib")  



void mexFunction(int nlhs, mxArray* plhs[], int nrhs, const mxArray* prhs[]) {
   //Checks arguments.
  if (nrhs != 7)
    mexErrMsgTxt("Eight input arguments required.");
  if (nlhs > 1)
    mexErrMsgTxt("Too many output arguments.");

  if (!mxIsDouble(prhs[0]))
	  mexErrMsgTxt("The 1st input argument must be double.");
  if (!mxIsDouble(prhs[1]))
	  mexErrMsgTxt("The 2nd input argument must be double.");
  if (!mxIsDouble(prhs[2]) || mxIsComplex(prhs[2]) || mxIsSparse(prhs[2]))
	  mexErrMsgTxt("The 3rd input argument must be double, real, and dense.");
  if (!mxIsStruct(prhs[3]))
    mexErrMsgTxt("The 4th input argument must be a struct.");
  if (!mxIsStruct(prhs[4]))
	  mexErrMsgTxt("The 5th input argument must be a struct.");
  if (!mxIsStruct(prhs[5]))
	  mexErrMsgTxt("The 6th input argument must be a struct.");
  if (!mxIsStruct(prhs[6]))
	  mexErrMsgTxt("The 7th input argument must be a struct.");

  Prox::weight W[2] = { { 0, 0, NULL, NULL, NULL }, { 0, 0, NULL, NULL, NULL } }; // w_row, w_col

  const double lambda_l1l2 = mxGetScalar(prhs[0]);
  const double invLipC = mxGetScalar(prhs[1]);
 
  const Eigen::MapMatrixDouble XTX = Eigen::matlab2Eigen<double>(prhs[2]);
  const size_t N = XTX.rows();

  mxArray* prhs_field; 

  for (size_t i = 0; i < 2; ++i)
  {
	  // prhs_weights
	  prhs_field = mxGetField(prhs[3+i], 0, "weights");
	  W[i].num_weights = *(mxGetJc(prhs_field) + mxGetN(prhs_field));
	  W[i].weights = mxGetPr(prhs_field);
	  W[i].weights_ir = mxGetIr(prhs_field);
	  W[i].weights_jc = mxGetJc(prhs_field);

	  // prhs_max_degree
	  prhs_field = mxGetField(prhs[3 + i], 0, "max_degree");	
	  W[i].max_degree = mxGetScalar(prhs_field);
  }

   

  Prox::tolerance tols[2] = {{0,0.0},{0,0.0}};
	for (size_t i = 0; i < 2; ++i)
	{
		//prhs_ITER
		prhs_field = mxGetField(prhs[5 + i], 0, "ITER");		
		tols[i].ITER = ((int)mxGetScalar(prhs_field));

		//prhs_ERR
		prhs_field = mxGetField(prhs[5 + i], 0, "ERR");		
		tols[i].ERR = mxGetScalar(prhs_field);	   
	}

  prhs_field = NULL;

  plhs[0] = mxCreateNumericMatrix(N, N, mxDOUBLE_CLASS, mxREAL);
  Eigen::MapMatrixDouble V = Eigen::matlab2Eigen<double>(plhs[0]);  

  Prox::Fista fista(lambda_l1l2, invLipC, N, XTX, W[0], W[1], tols[0], tols[1]);
  fista.Solve(V);
  printf ("Done!\n");
}

