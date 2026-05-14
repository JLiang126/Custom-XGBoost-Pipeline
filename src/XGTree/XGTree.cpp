#include "XGTree/XGTree.hpp"

#include <algorithm>
#include <cmath>
#include <fstream>
#include <numeric>
#include <string>

Tree::Tree(int max_depth, float lambda, float gamma, float min_cover) : max_depth(max_depth), lambda(lambda), gamma(gamma), min_cover(min_cover) {}

float Tree::predict(const vector<float> &features) const
{
    if (root)
        return root->predict(features);
    return 0.0f;
}

float Tree::calc_leaf_weight(float sum_g, float sum_h) const { return -sum_g / (sum_h + lambda); }

float Tree::calc_gain(float sum_g_left, float sum_h_left,
                      float sum_g_right, float sum_h_right,
                      float sum_g_root, float sum_h_root) const
{
    float sim_left = (sum_g_left * sum_g_left) / (sum_h_left + lambda);
    float sim_right = (sum_g_right * sum_g_right) / (sum_h_right + lambda);
    float sim_root = (sum_g_root * sum_g_root) / (sum_h_root + lambda);

    return sim_left + sim_right - sim_root - gamma;
}

void Tree::grow_tree(Node *curr,
                     const DataMatrix &data,
                     const vector<derivative> &derivatives,
                     const vector<int> &row_idxs,
                     int curr_depth)
{
    float sum_g_root = 0.0f, sum_h_root = 0.0f;
    for (int i : row_idxs)
    {
        sum_g_root += derivatives[i].g;
        sum_h_root += derivatives[i].h;
    }

    if (curr_depth >= max_depth || row_idxs.size() < 2)
    {
        curr->is_leaf = true;
        curr->leaf_weight = calc_leaf_weight(sum_g_root, sum_h_root);
        return;
    }

    float best_gain = 0.0f;
    int best_feature_idx = -1;
    float best_spilt_val = 0.0f;

    vector<int> sorted_idxs = row_idxs;
    int num_rows = sorted_idxs.size();

    for (size_t i{0}; i < data.get_num_columns(); i++)
    {
        const vector<float> &feature_column = data.get_column(i);

        sort(sorted_idxs.begin(), sorted_idxs.end(), [&feature_column](int a, int b)
             { return feature_column[a] < feature_column[b]; });

        float sum_g_left = 0.0f;
        float sum_h_left = 0.0f;

        for (size_t j{0}; j < num_rows - 1; j++)
        {
            int curr_idx = sorted_idxs[j];
            int next_idx = sorted_idxs[j + 1];

            sum_g_left += derivatives[curr_idx].g;
            sum_h_left += derivatives[curr_idx].h;

            if (feature_column[curr_idx] == feature_column[next_idx])
                continue; // if the feats are the same they will continue to enter the left bucket before grading.

            float sum_g_right = sum_g_root - sum_g_left;
            float sum_h_right = sum_h_root - sum_h_left;

            if (sum_h_left < min_cover || sum_h_right < min_cover)
                continue;

            float split = (feature_column[curr_idx] + feature_column[next_idx]) / 2.0f;
            float curr_gain = calc_gain(sum_g_left, sum_h_left, sum_g_right, sum_h_right, sum_g_root, sum_h_root);

            if (curr_gain > best_gain)
            {
                best_gain = curr_gain;
                best_feature_idx = static_cast<int>(i);
                best_spilt_val = split;
            }
        }
    }

    if (best_gain <= 0.0f)
    {
        curr->is_leaf = true;
        curr->leaf_weight = calc_leaf_weight(sum_g_root, sum_h_root);
        return;
    }

    curr->split_feature = best_feature_idx;
    curr->split_val = best_spilt_val;
    curr->split_gain = best_gain;

    vector<int> left_samples, right_samples;
    left_samples.reserve(row_idxs.size());
    right_samples.reserve(row_idxs.size());

    const vector<float> &best_column = data.get_column(best_feature_idx);
    for (int i : row_idxs)
    {
        if (best_column[i] < best_spilt_val)
            left_samples.push_back(i);
        else
            right_samples.push_back(i);
    }

    if (left_samples.empty() || right_samples.empty())
    {
        curr->is_leaf = true;
        curr->leaf_weight = calc_leaf_weight(sum_g_root, sum_h_root);
        return;
    }

    curr->left = make_unique<Node>();
    curr->right = make_unique<Node>();

    grow_tree(curr->left.get(), data, derivatives, left_samples, curr_depth + 1);
    grow_tree(curr->right.get(), data, derivatives, right_samples, curr_depth + 1);
    return;
}

void Tree::build(const DataMatrix &data, const vector<derivative> &derivatives)
{
    root = make_unique<Node>();

    vector<int> row_idxs(data.get_num_rows());
    iota(row_idxs.begin(), row_idxs.end(), 0);

    grow_tree(root.get(), data, derivatives, row_idxs, 0);
    return;
}

unique_ptr<Node> Tree::load_node(ifstream &in)
{
    string type;
    if (!(in >> type))
        return nullptr;

    auto node = make_unique<Node>();

    if (type == "L")
    {
        node->is_leaf = true;
        in >> node->leaf_weight;
    }
    else if (type == "N")
    {
        node->is_leaf = false;
        in >> node->split_feature >> node->split_val;

        node->left = load_node(in);
        node->right = load_node(in);
    }

    return node;
}

void Tree::load(ifstream &in) { root = load_node(in); }

void Tree::save_node(ofstream &out, Node *curr) const
{
    if (!curr)
        return;

    if (curr->is_leaf)
        out << "L " << curr->leaf_weight << "\n";
    else
    {
        out << "N " << curr->split_feature << " " << curr->split_val << "\n";
        save_node(out, curr->left.get());
        save_node(out, curr->right.get());
    }
    return;
}

void Tree::save(ofstream &out) const { save_node(out, root.get()); }
