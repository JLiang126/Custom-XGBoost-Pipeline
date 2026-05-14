#include "XGTree/XGNode.hpp"

float Node::predict(const vector<float> &features) const
{
    if (is_leaf)
        return leaf_weight;

    if (features[split_feature] < split_val)
        return left->predict(features);
    else
        return right->predict(features);

    return 0.0;
}