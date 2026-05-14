#pragma once

#include <fstream>
#include <memory>
#include <vector>

#include "dataloader/DataMatrix.hpp"
#include "objective/Loss.hpp"
#include "XGTree/XGTree.hpp"

using namespace std;

class XGBoost
{
private:
    vector<Tree> forest;
    shared_ptr<Loss> objective;

    int num_trees;
    float learning_rate;

    int max_depth;
    float lambda; // L2 regularisation term
    float gamma;  // Min gain to make a spilt
    float min_cover;

public:
    XGBoost(int num_trees, float learning_rate, int max_depth, float lambda, float gamma, float min_cover, shared_ptr<Loss> objective);
    void train(const DataMatrix &data, const vector<float> &labels);
    float predict(const vector<float> &sample) const;
    void load_model(const string &filepath);
    void save_model(const string &filepath) const;
};