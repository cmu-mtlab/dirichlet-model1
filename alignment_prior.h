#include <vector>
#include <iostream>
#include <cmath>
#include "cpyp/m.h"

using std::abs;
using std::vector;
using std::cerr;
using std::endl;
using namespace cpyp;

struct alignment_prior {
  // i is index into target sentence
  // j is index into source sentence
  // m is length of target sentence
  // n is length of source sentence (not counting NULL)
  bool use_null;
  virtual double prob(unsigned short i, unsigned short j, unsigned short m, unsigned short n) const = 0;
  virtual double null_prob(unsigned short i, unsigned short m, unsigned short n) const = 0;
  virtual double log_likelihood(const vector<vector<unsigned short>>& alignments, const vector<vector<unsigned>>& src_corpus) const = 0;
};

// A uniform pure IBM1-style prior on alignments
struct uniform_alignment_prior : alignment_prior {
  double prob(unsigned short i, unsigned short j, unsigned short m, unsigned short n) const override {
    return 1.0;
  }

  double null_prob(unsigned short i, unsigned short m, unsigned short n) const override {
    return use_null ? 1.0 : 0.0;
  }

  double log_likelihood(const vector<vector<unsigned short>>& alignments, const vector<vector<unsigned>>& src_corpus) const override {
    return 0.0;
  }
};

// C. Dyer, V. Chahuneau, and N. Smith (2013)'s alignment prior that favors the diagonal
// See http://www.cs.cmu.edu/~nasmith/papers/dyer+chahuneau+smith.naacl13.pdf
struct diagonal_alignment_prior : alignment_prior {
  diagonal_alignment_prior(double initial_tension, double initial_p0, bool use_null) {
    this->tension = initial_tension;
    this->p0 = initial_p0;
    this->use_null = use_null;
  }

  double prob(unsigned short i, unsigned short j, unsigned short m, unsigned short n) const override {
    return (1.0 - p0) * exp(-tension * abs(1.0 * i / m - 1.0 * j / n));
  }

  double null_prob(unsigned short i, unsigned short m, unsigned short n) const override {
    return use_null ? p0 : 0.0;
  }

  double log_likelihood(const vector<vector<unsigned short>>& alignments, const vector<vector<unsigned>>& src_corpus) const override {
    return log_likelihood(alignments, src_corpus, p0, tension);
  }

  double log_likelihood(const vector<vector<unsigned short>>& alignments, const vector<vector<unsigned>>& src_corpus, double p0, double tension) const {
    assert(src_corpus.size() == alignments.size());

    const double p0_alpha = 1.0;
    const double p0_beta = 100.0;
    const double tension_shape = 7.0;
    const double tension_rate = 1.0;

    double llh = Md::log_beta_density(p0, p0_alpha, p0_beta) +
                 Md::log_gamma_density(tension, tension_shape, tension_rate);

    for(unsigned s = 0; s < src_corpus.size(); ++s) {
      unsigned short n = src_corpus[s].size() - 1;
      unsigned short m = alignments[s].size();
      double Z = 1.0; // TODO: Calculate Z for real
      for(unsigned short i = 0; i < alignments[s].size(); ++i) {
        unsigned short j = alignments[s][i];
        llh += log(prob(i, j, m, n) / Z);
      }
    }
    return llh;
  }

  template<typename Engine>
  void resample_hyperparameters(const vector<vector<unsigned short>>& alignments, const vector<vector<unsigned>>& src_corpus, Engine& eng, const unsigned nloop = 5, const unsigned niterations = 10) {
    for (unsigned iter = 0; iter < nloop; ++iter) {
      tension = slice_sampler1d([this, &alignments, &src_corpus](double prop_tension) { return this->log_likelihood(alignments, src_corpus, p0, prop_tension); },
                              tension, eng, std::numeric_limits<double>::min(),
                              std::numeric_limits<double>::infinity(), 0.0, niterations, 100*niterations);

      p0 = slice_sampler1d([this, &alignments, &src_corpus](double prop_p0) { return this->log_likelihood(alignments, src_corpus, prop_p0, tension); },
                         p0, eng, 0.0, 1.0, 0.0, niterations, 100*niterations);
    }

    cerr << "Resampled diagonal alignment parameters (p0=" << p0 << ",tension=" << tension  << ") = " << log_likelihood(alignments, src_corpus) << endl;
  }

  double tension;
  double p0;
};
