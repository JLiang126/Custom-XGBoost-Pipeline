#pragma once

#include <cstdint>
#include <vector>

using namespace std;

class Binner
{
private:
    int num_bins;

    // thresholds for each feature column - thresholds[feature_idx][bin_idx] = float_boundary
    vector<vector<float>> thresholds;
public:
    Binner(int bins);

    void fit(const vector<vector<float>> raw_data); 
};