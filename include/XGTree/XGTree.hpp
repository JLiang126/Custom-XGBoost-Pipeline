#pragma once

#include <iostream>
#include <memory>
#include <vector>

#include "XGNode.hpp"
#include "dataloader/DataMatrix.hpp"
#include "objective/Loss.hpp"

using namespace std;

class Tree
{
private:
    unique_ptr<Node> root;

    int max_depth;
    float lambda; // L2 regularisation term
    float gamma;  // Min gain to make a spilt
    float min_cover;

    float calc_leaf_weight(float sum_g, float sum_h) const; // calulates leafs weight using G and H

    // Total Gain = sim score left + sim score right + sim score root
    float calc_gain(float sum_g_left, float sum_h_left,
                    float sum_g_right, float sum_h_right,
                    float sum_g_root, float sum_h_root) const;

    void grow_tree(Node *curr,
                   const DataMatrix &data,
                   const vector<derivative> &derivatives,
                   const vector<int> &row_idxs,
                   int curr_depth);

    unique_ptr<Node> load_node(ifstream &in);
    void save_node(ofstream &out, Node *curr) const;

public:
    Tree(int max_depth, float reg_lambda, float gamma, float min_cover);

    void build(const DataMatrix &data, const vector<derivative> &derivatives);
    float predict(const vector<float> &features) const;
    void load(ifstream &in);
    void save(ofstream &out) const;
};