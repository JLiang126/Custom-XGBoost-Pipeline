#pragma once

#include <string>
#include <vector>

using namespace std;

class DataMatrix
{
private:
    size_t num_rows;
    size_t num_columns;

    vector<string> feature_names;
    vector<vector<float>> columns;
    vector<vector<float>> rows;
    vector<float> labels;

public:
    DataMatrix(const string& filepath);

    size_t get_num_rows() const; 
    size_t get_num_columns() const;

    // The const at the end of a function signature tells the compiler that this method will not change the object it belongs to.
    const vector<string>& get_feature_names() const; 
    const vector<float>& get_column(size_t column) const;
    const vector<float>& get_row(size_t row) const;
    const vector<float>& get_labels() const;
};