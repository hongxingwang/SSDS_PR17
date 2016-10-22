// Modified by Hongxing Wang, 2014
// Author: KANEMURA Atsunori <kanemura.atsunori@gmail.com>
//#include "stdafx.h"
#include "proxflow.h"
const float large_top = 100000.0;

// Computes (-1/lambda)' * z * e + (lambda2) * e' L e + (lambda3) * sum_g min( card(g),1 ) + beta * e.
double ProximalParaflow::F(const std::vector<size_t>& S) const {
  // Linear part.
  double f1 = 0;
  for (int i = 0; i < S.size(); i++) {
    f1 += z_(S[i]);
  }
  // Create a vector select_stats with N_ elements of default value 0  
  std::vector<int> select_stats(N_);
  // Quadratic part.
  double f2 = 0;
  if (lambda2_ != 0){
	  for (int i = 0; i < S.size(); i++)
		select_stats[S[i]] = 1;
	  for (size_t col = 0; col < N_; col++) {
		for (size_t i = weights_jc_[col]; i < weights_jc_[col + 1]; i++) {
		  const size_t row = weights_ir_[i];
		  if (select_stats[col] != select_stats[row])
			f2 += weights_[i];
		}
	  }
  }
    

  double f = (-1 / lambda_) * f1  + S.size() * beta_
             + lambda2_ * f2;
  printf ("\n Characters\n");
  LOG("F = %g\n", f);
  printf("F = %g\n", f);
  return f;
}

void ProximalParaflow::CalcV(Eigen::VectorXd &v) {

  const size_t num_subsets = graph_.lambdas.size() + 1;
  std::vector<size_t> S;
  std::vector<double> Fs(num_subsets+1);//
  std::vector<size_t> Ns(num_subsets+1);//
  for (size_t s = 0; s <= num_subsets; s++) {//
    for(size_t n = 0; n < N_; n++)
      if (graph_.nodes[1 + n].label == num_subsets + 1 - s)
        S.push_back(n);
    Fs[s] = F(S);
    Ns[s] = S.size();
  }
  for (size_t n = 0; n < N_; n++) {
    size_t ind = num_subsets - graph_.nodes[1 + n].label;
    v(n) = (Fs[ind + 1] - Fs[ind]) / (Ns[ind + 1] - Ns[ind] + EPS);
  }

  // Correct the offset and converted by -lambda.
  for (size_t n = 0; n < N_; n++)
	 v(n) = -lambda_ * (v(n) - beta_);

}

void ProximalParaflow::FindBeta() {
  //const double max_z = *std::max_element(z_, z_ + N_);
  const double max_z = z_.cwiseAbs().maxCoeff();
  //const double max_degree = 0; // Should be max_j sum_i W_ij
  beta_ = max_degree_ + (1 / lambda_) * max_z; // 
  large_ = large_top / ( EPS + std::min(max_degree_, (1 / lambda_) * max_z) );
 
  if (beta_ < 0)
    beta_ = 0;
  LOG("beta = %g\n", beta_);
}

void ProximalParaflow::Solve() {
  Solver solver;
  Problem problem;
  problem.type = Problem::PARAMETRIC_FLOW_PROBLEM;
  problem.hasLambdaRange = true;
  problem.minLambda = -10e+16*large_;
  problem.maxLambda = beta_*large_; // to be large enough to find breakpoints
  solver.verbosity = Solver::QUIET_VERBOSITY;
  solver.Solve(graph_, problem);
}

void ProximalParaflow::AddArc(const int tail, const int head,
                              const double cap) {
  paraF::Arc* arc;
  Node* node_tail = &graph_.nodes[tail];
  Node* node_head = &graph_.nodes[head];
  LOG("Adding an arc (%d, %d, %g)\n", tail, head, cap);
  if (tail == 0)  // Source.
    arc = graph_.AddArcFromSource(node_tail, node_head, cap, 0);
  else if (head == graph_.nodes.size() - 1)  // Sink.
    arc = graph_.AddArcToSink(node_tail, node_head, cap, -1);
  else
    arc = graph_.AddArc(node_tail, node_head, cap);
  paraF::Arc* revArc = graph_.AddArc(node_head, node_tail, 0);
  arc->revArc = revArc;
  revArc->revArc = arc;
}


void ProximalParaflow::ConstructGraph() {
  // Initializes a graph.
  const int num_nodes = (int)(N_) + 2;
  //const int num_edges = 2 * N_ + num_weights_ + 2 * num_groups_ * (N_ + 1); // N_ + num_groups_
  const int num_edges = (int) (2 * N_ + num_weights_); // N_ + num_groups_
  graph_.Reserve(2 * num_edges + 4 * num_nodes, num_nodes);
  for (int i = 0; i < num_nodes; i++)
    graph_.AddNode();
  const int source = 0;
  const int sink = num_nodes - 1;

  // Linear weights.
  for (int n = 0; n < N_; n++) {
    double cap1 = (-1.0 / lambda_) * z_(n) + beta_; // must add beta_
    double cap2 = beta_; // must add beta_, 2014-04-01
	AddArc(source, 1 + n, cap1*large_); // *large
    AddArc(1 + n, sink, cap2*large_); // *large
  }

  // Quadratic weights.
  if(lambda2_ != 0)
  {
	  for (int col = 0; col < N_; col++) {
		for (int i = (int) weights_jc_[col]; i < weights_jc_[col + 1]; i++) {
		  const int row = (int) weights_ir_[i];
		  const double weight = lambda2_ * weights_[i];
		  if (abs(weight) > EPS){
			  AddArc(1 + row, 1 + col, weight*large_);} // *large
		}
	  }
  } 

  //--------------------------------------------------------------------

  // Post-processing.
  graph_.AddAuxArcs(&graph_.nodes[source], &graph_.nodes[sink]);
  graph_.PrepareNodes(&graph_.nodes[source], &graph_.nodes[sink]);
  graph_.PrepareArcs();
}
