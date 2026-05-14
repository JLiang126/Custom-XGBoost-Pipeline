#include <pybind11/pybind11.h>
#include <pybind11/stl.h> // Automatically converts Python lists to std::vector

#include "engine/XGBoost.hpp"
#include "dataloader/DataMatrix.hpp"
#include "objective/Loss.hpp"

namespace py = pybind11;

PYBIND11_MODULE(my_xgboost, m)
{
    m.doc() = "Custom C++ XGBoost";

    py::class_<DataMatrix>(m, "DataMatrix")
        .def(py::init<const std::string &>(), py::arg("filepath"))
        .def("get_num_rows", &DataMatrix::get_num_rows)
        .def("get_num_columns", &DataMatrix::get_num_columns)
        .def("get_row", &DataMatrix::get_row, py::arg("row"))
        .def("get_column", &DataMatrix::get_column, py::arg("column"))
        .def("get_labels", &DataMatrix::get_labels)
        .def("get_feature_names", &DataMatrix::get_feature_names);

    py::class_<Loss, std::shared_ptr<Loss>>(m, "Loss");
    py::class_<MSELoss, Loss, std::shared_ptr<MSELoss>>(m, "MSELoss")
        .def(py::init<>());
    py::class_<LogLoss, Loss, std::shared_ptr<LogLoss>>(m, "LogLoss")
        .def(py::init<>());

    py::class_<XGBoost>(m, "XGBoost")
        .def(py::init<int, float, int, float, float, float, shared_ptr<Loss>>(),
             py::arg("num_trees"),
             py::arg("learning_rate"),
             py::arg("max_depth"),
             py::arg("lambda_reg"), // 'lambda' is a reserved keyword in Python!
             py::arg("gamma"),
             py::arg("min_cover"),
             py::arg("objective"))

        // Expose the training and prediction methods
        .def("train", &XGBoost::train, py::arg("data"), py::arg("labels"))
        .def("predict", &XGBoost::predict, py::arg("features"))
        .def("save_model", &XGBoost::save_model)
        .def("load_model", &XGBoost::load_model);
}