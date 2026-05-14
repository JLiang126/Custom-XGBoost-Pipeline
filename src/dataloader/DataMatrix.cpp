#include "dataloader/DataMatrix.hpp"

#include <fstream>
#include <sstream>
#include <stdexcept>

DataMatrix::DataMatrix(const string &filepath) // This data matrix assumes all data is being fed numbers in the form of strings
{
    ifstream file(filepath);
    if (!file.is_open())
        throw runtime_error("Could not open filepath:" + filepath);

    string line;
    if (getline(file, line))
    {
        stringstream ss(line);
        string column_name;
        while (getline(ss, column_name, ','))
            feature_names.push_back(column_name);
    }

    vector<vector<float>> temp_matrix;
    while (getline(file, line)) // grab line
    {
        stringstream ss(line); // Turns the row into a steam so you can "read" it
        string cell;
        vector<float> temp_row;

        while (getline(ss, cell, ','))
        {
            try
            {
                temp_row.push_back(std::stof(cell));
            }
            catch (const std::exception &e)
            {
                continue;
            }
        }

        temp_matrix.push_back(temp_row);
    }

    num_rows = temp_matrix.size();
    if (num_rows > 0)
    {
        num_columns = temp_matrix[0].size() - 1;
        columns.resize(num_columns);
        rows.resize(num_rows);
        labels.reserve(num_rows);
    }

    for (size_t r = 0; r < num_rows; r++)
    {
        for (size_t c = 0; c < num_columns; c++)
        {
            columns[c].push_back(temp_matrix[r][c]);
            rows[r].push_back(temp_matrix[r][c]);
        }
        labels.push_back(temp_matrix[r].back()); // assume the label is at the back e.g. whether they have heart disease or not
    }
}

size_t DataMatrix::get_num_rows() const { return num_rows; }
size_t DataMatrix::get_num_columns() const { return num_columns; }

// The const at the end of a function signature tells the compiler that this method will not change the object it belongs to.
const vector<string> &DataMatrix::get_feature_names() const { return feature_names; }
const vector<float> &DataMatrix::get_column(size_t column) const { return columns[column]; }
const vector<float> &DataMatrix::get_row(size_t row) const { return rows[row]; }
const vector<float> &DataMatrix::get_labels() const { return labels; }