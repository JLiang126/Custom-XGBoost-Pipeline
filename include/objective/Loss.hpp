#pragma once

#include <cmath>
#include <vector>

using namespace std;

struct derivative
{
    float g;
    float h;
};

class Loss
{
public:
    virtual void compute_gradients(const vector<float> &preds,
                                   const vector<float> &labels,
                                   vector<derivative> &derivatives) const = 0;

    virtual ~Loss() {}
};

// Regression
class MSELoss : public Loss
{
public:
    virtual void compute_gradients(const vector<float> &preds,
                                   const vector<float> &labels,
                                   vector<derivative> &derivatives) const override;
};

// Classification
class LogLoss : public Loss
{
private:
    float sigmoid(float n) const;

public:
    virtual void compute_gradients(const vector<float> &preds,
                                   const vector<float> &labels,
                                   vector<derivative> &derivatives) const override;
};