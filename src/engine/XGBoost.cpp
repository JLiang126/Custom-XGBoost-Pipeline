#include "engine/XGBoost.hpp"

#include <iostream>

XGBoost::XGBoost(int num_trees, float learning_rate, int max_depth, float lambda, float gamma, float min_cover, shared_ptr<Loss> objective) : num_trees(num_trees), learning_rate(learning_rate), max_depth(max_depth), lambda(lambda), gamma(gamma), min_cover(min_cover), objective(std::move(objective)) {}

void XGBoost::train(const DataMatrix &data, const vector<float> &labels)
{
    int num_rows = data.get_num_rows();
    vector<float> curr_preds(num_rows, 0.0f);
    vector<derivative> derivatives(num_rows);

    for (size_t i{0}; i < num_trees; i++)
    {
        objective->compute_gradients(curr_preds, labels, derivatives);

        Tree XGTree(max_depth, lambda, gamma, min_cover);
        XGTree.build(data, derivatives);

        for (size_t j{0}; j < num_rows; j++)
            curr_preds[j] += learning_rate * XGTree.predict(data.get_row(j));

        forest.push_back(std::move(XGTree));

        cout << "Tree" << i + 1 << "/" << num_trees << "built" << endl;
    }
    return;
}

float XGBoost::predict(const vector<float> &sample) const
{
    float prediction = 0.0f;
    for (const auto &tree : forest)
        prediction += learning_rate * tree.predict(sample);
    return prediction;
}

void XGBoost::load_model(const string &filepath)
{
    ifstream in(filepath);

    if (!in.is_open())
        throw runtime_error("Could not open model: " + filepath);

    in >> this->num_trees >> this->learning_rate >> this->max_depth;

    forest.clear();

    for (size_t i{0}; i < num_trees; i++)
    {
        Tree tree(max_depth, 0.0f, 0.0f, 0.0f);
        tree.load(in);
        forest.push_back(std::move(tree));
    }

    in.close();
    return;
}

void XGBoost::save_model(const string &filepath) const
{
    ofstream out(filepath);
    if (!out.is_open())
        throw runtime_error("Could not open file to save model: " + filepath);

    out << num_trees << " " << learning_rate << " " << max_depth << "\n";

    for (const auto &tree : forest)
        tree.save(out);

    out.close();
}