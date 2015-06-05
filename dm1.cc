#include <iostream>
#include <unordered_map>
#include <cstdlib>
#include "boost/archive/text_oarchive.hpp"
#include "boost/program_options.hpp"

#include "bilingual_corpus.h"
#include "cpyp/m.h"
#include "cpyp/random.h"
#include "cpyp/crp.h"
#include "cpyp/tied_parameter_resampler.h"
#include "alignment_prior.h"

using namespace std;
using namespace cpyp;
namespace po=boost::program_options;

Dict src_dict;
Dict tgt_dict;

double log_likelihood(const tied_parameter_resampler<crp<unsigned>>& p, 
                      const diagonal_alignment_prior& ap,
                      const vector<vector<unsigned>>& src_corpus,
                      const vector<crp<unsigned>>& ttable,
                      const vector<vector<unsigned short>>& a) {
  double llh = p.log_likelihood();
  for (auto& crp : ttable)
    llh += crp.log_likelihood();
  //llh += ap.log_likelihood(a, src_corpus);
  return llh;
}

void show_ttable(vector<crp<unsigned>>& ttable, Dict& src_dict, Dict& tgt_dict) {
  for (unsigned src_id = 1; src_id < ttable.size(); src_id++) {
    crp<unsigned>& p = ttable[src_id];
    vector<unsigned> ind(tgt_dict.max());
    for (unsigned tgt_id = 0; tgt_id < tgt_dict.max(); tgt_id++)
      ind[tgt_id] = tgt_id;

    cerr << src_dict.Convert(src_id) << "\n";
    partial_sort(ind.begin(), ind.begin() + 10, ind.end(), [&p, &tgt_dict](unsigned a, unsigned b) { return p.prob(a, 1.0 / tgt_dict.max()) > p.prob(b, 1.0 / tgt_dict.max()); });
    for (unsigned i = 0; i < 10; i++) {
      unsigned tgt_id = ind[i];
      cerr << "\t" << tgt_dict.Convert(tgt_id) << "\t" << p.prob(tgt_id, 1.0 / tgt_dict.max()) << endl;
    }
  }
}

void output_alignments(vector<vector<unsigned>>& tgt_corpus, vector<vector<unsigned short>>& a) {
  for (unsigned i = 0; i < tgt_corpus.size(); i++) {
    for (unsigned j = 0; j < tgt_corpus[i].size(); j++) {
      if (a[i][j] != 0) {
        cout << a[i][j] - 1 << "-" << j << " ";
      }
    }
    cout << "\n";
  }

}

int main(int argc, char** argv) {
  po::options_description options("Options");
  options.add_options()
    ("training_corpus,i", po::value<string>()->required(), "Training corpus, in format of source ||| target or docid ||| source ||| target")
    ("samples,n", po::value<int>()->required(), "Number of samples") 
    ("help", "Print help messages");
  po::variables_map args;
  try {
    po::store(po::parse_command_line(argc, argv, options), args);
    if (args.count("help")) {
       cerr << options << endl;
       return 0;
    }
    po::notify(args);
  }
  catch (po::error& e) {
    cerr << "ERROR: " << e.what() << endl << endl;
    cerr << options << endl;
    return 1;
  }

  MT19937 eng;
  string training_corpus_file = args["training_corpus"].as<string>();
  const bool use_alignment_prior = true;
  const bool use_null = true;
  diagonal_alignment_prior diag_alignment_prior(4.0, 0.01, use_null);
  const unsigned samples = args["samples"].as<int>();
  
  vector<vector<unsigned>> src_corpus;
  vector<vector<unsigned>> tgt_corpus;
  set<unsigned> src_vocab;
  set<unsigned> tgt_vocab;
  ReadFromFile(training_corpus_file, &src_dict, &src_corpus, &src_vocab, &tgt_dict, &tgt_corpus, &tgt_vocab);
  double uniform_target_word = 1.0 / tgt_vocab.size();
  assert(src_corpus.size() == tgt_corpus.size());
  // dicts contain 1 extra word, <bad>, so the values in src_corpus and tgt_corpus
  // actually run from [1, *_vocab.size()], instead of being 0-indexed.
  cerr << "Corpus size: " << src_corpus.size() << " documents\t (" << src_vocab.size() << "/" << tgt_vocab.size() << " word types)\n";

  vector<vector<unsigned short>> a;
  a.resize(tgt_corpus.size());
  vector<crp<unsigned>> ttable(src_vocab.size() + 1, crp<unsigned>(0.0, 0.001));
  tied_parameter_resampler<crp<unsigned>> ttable_params(1,1,1,1,0.1,1);
  for (unsigned i = 0; i < src_corpus.size(); ++i) {
    src_corpus[i].insert(src_corpus[i].begin(), 0);
  }
  for (unsigned i = 0; i < tgt_corpus.size(); ++i) {
    a[i].resize(tgt_corpus[i].size());
  }
  for (unsigned i = 0; i < src_vocab.size() + 1; i++) {
    ttable_params.insert(&ttable[i]);
  }

  unsigned longest_src_sent_length = 0;
  for (unsigned i = 0; i < src_corpus.size(); i++) {
    longest_src_sent_length = (src_corpus[i].size() > longest_src_sent_length) ? src_corpus[i].size() : longest_src_sent_length;
  }

  vector<double> probs(longest_src_sent_length);
  for (unsigned sample=0; sample < samples; ++sample) {
    cerr << "beginning loop with sample = " << sample << endl;
    for (unsigned i = 0; i < tgt_corpus.size(); ++i) {
      const auto& src = src_corpus[i];
      const auto& tgt = tgt_corpus[i];
      for (unsigned j = 0; j < tgt.size(); ++j) {
        unsigned short& a_ij = a[i][j];
        const unsigned t = tgt[j];
        if (sample > 0) { 
          ttable[src[a_ij]].decrement(t, eng);
        }
        probs.resize(src.size());
        for (unsigned k = 0; k < src.size(); ++k) {
          probs[k] = ttable[src[k]].prob(t, uniform_target_word);
          if (use_alignment_prior) {
            double alignment_prob;
            if (k == 0) {
              if(use_null) {
                alignment_prob = diag_alignment_prior.null_prob(j, tgt.size(), src.size() - 1);
              }
              else {
                alignment_prob = 0.0;
              }
            }
            else {
              alignment_prob = diag_alignment_prior.prob(j + 1, k, tgt.size(), src.size() - 1);
            } 
            probs[k] *= alignment_prob;
          }
        }
        multinomial_distribution<double> mult(probs);
        // random sample during the first iteration
        a_ij = sample ? mult(eng) : static_cast<unsigned>(sample_uniform01<float>(eng) * src.size());  
        a[i][j] = a_ij;
        if (a_ij < 0 || a_ij >= src.size())
          cerr << i << " " << j << " " << a_ij << " " << src.size() << "\n";
        assert(a_ij >= 0);
        assert(a_ij < src.size()); 
        ttable[src[a_ij]].increment(t, uniform_target_word, eng);
      }
    }

    output_alignments(tgt_corpus, a);
    if (sample % 10 == 9) {
      cerr << " [LLH=" << log_likelihood(ttable_params, diag_alignment_prior, src_corpus, ttable, a) << "]" << endl;
      if (sample % 30u == 29) {
        ttable_params.resample_hyperparameters(eng);
        diag_alignment_prior.resample_hyperparameters(a, src_corpus, eng);
      }
    } else { cerr << '.' << flush; }
  }

  if(src_dict.max() < 100)
    show_ttable(ttable, src_dict, tgt_dict);
  return 0;
}
