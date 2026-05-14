#include "objective/Loss.hpp"

using namespace std;

void MSELoss::compute_gradients(const vector<float> &preds,
                                const vector<float> &labels,
                                vector<derivative> &derivatives) const
{

    // Loss function: 0.5 * (predicted value - true value)^2;
    for (size_t i{0}; i < preds.size(); i++)
    {
        derivatives[i].g = preds[i] - labels[i];
        derivatives[i].h = 1.0f;
    }
}

float LogLoss::sigmoid(float x) const { return 1.0f / (1.0f + exp(-x)); }

void LogLoss::compute_gradients(const vector<float> &preds,
                                const vector<float> &labels,
                                vector<derivative> &derivatives) const
{
    // p = 1 / (1 - e^(-z)), z = log-odds score raw output of desision tree.
    // Loss function: -[y * log(p) + (1 - y) * log(1 - p)]
    for (size_t i{0}; i < preds.size(); i++)
    {
        float p = sigmoid(preds[i]);

        derivatives[i].g = p - labels[i];
        derivatives[i].h = p * (1.0f - p);
    }
}