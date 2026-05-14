#pragma once

#include <memory>
#include <vector>

// #include "XGTree.hpp"

using namespace std;

class Tree;

class Node
{
private:
    bool is_leaf = false;

    float leaf_weight = 0.0f;

    int split_feature = -1;
    float split_val = 0.0f;
    float split_gain = 0.0f;

    unique_ptr<Node> left;
    unique_ptr<Node> right;

    friend class Tree;

public:
    Node() = default;
    float predict(const vector<float> &features) const;
};