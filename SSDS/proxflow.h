// Modified by Hongxing Wang, 2014
// Author: KANEMURA Atsunori <kanemura.atsunori@gmail.com>
#ifndef PROXFLOW_INCLUDE_PROXFLOW_H_
#define PROXFLOW_INCLUDE_PROXFLOW_H_

#include "Graph.h"
#include "Solver.h"
#include <stdio.h>
//#include <mex.h>
#include <Eigen/Eigen> 


#define EPS 1.0E-10  /** allowed computational error bound **/

#ifndef NDEBUG
#define LOG(...) 
#else
#define LOG(...) printf(__VA_ARGS__)
#endif

//#define LOG(...) printf(__VA_ARGS__)
using namespace paraF;

//#pragma pack(1)

class ProximalParaflow {
  friend class ProxflowTest;
 public:
  ProximalParaflow(
	const size_t N, Eigen::VectorXd z,
    const size_t num_weights, const double* weights, const double max_degree,
    const size_t* weights_ir, const size_t* weights_jc,
    const double lambda, const double lambda2)
      : N_(N), z_(z), num_weights_(num_weights), weights_(weights), max_degree_(max_degree),
        weights_ir_(weights_ir), weights_jc_(weights_jc),
		lambda_(lambda), lambda2_(lambda2){
    FindBeta();
  };
  void ConstructGraph(void);
  void Solve(void);
  void CalcV(Eigen::VectorXd &v);

 private:
  const size_t N_;
  Eigen::VectorXd z_;
  const size_t num_weights_;
  const double* weights_;
  const double max_degree_;
  const size_t* weights_ir_;
  const size_t* weights_jc_;
  const double lambda_;
  const double lambda2_;
  Graph graph_;
  double beta_;
  double large_;
  void FindBeta();
  double F(const std::vector<size_t>& S) const;
  void AddArc(const int tail, const int head, const double cap);
};

//#pragma pack()

#endif  // PROXFLOW_INCLUDE_PROXFLOW_H_
