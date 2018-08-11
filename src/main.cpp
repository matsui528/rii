#include <pybind11/pybind11.h>
#include <pybind11/stl.h>  // To convert list of list <-> vec<vec<int>> for e.g. posting_lists
#include "rii.h"

namespace py = pybind11;

namespace rii {
PYBIND11_MODULE(main, m) {
    py::class_<RiiCpp>(m, "RiiCpp")
        .def(py::init<int, int, bool>())
        .def("set_codewords", &RiiCpp::SetCodewords)
        .def("reconfigure", &RiiCpp::Reconfigure)
        .def("add_codes", &RiiCpp::AddCodes)
        .def("query_linear", &RiiCpp::QueryLinear,
             py::arg("query").noconvert(),  // Prohibit implicit data conversion
             py::arg("topk"),
             py::arg("target_ids").noconvert()  // Prohibit implicit data conversion
             )
        .def("query_ivf", &RiiCpp::QueryIvf,
             py::arg("query").noconvert(),  // Prohibit implicit data conversion
             py::arg("topk"),
             py::arg("target_ids").noconvert(),  // Prohibit implicit data conversion
             py::arg("L")
             )
        .def("clear", &RiiCpp::Clear)
        .def_readwrite("verbose", &RiiCpp::verbose_)
        .def_readonly("coarse_centers", &RiiCpp::coarse_centers_)
        .def_readonly("flattened_codes", &RiiCpp::flattened_codes_)
        .def_readonly("posting_lists", &RiiCpp::posting_lists_)
        .def_property_readonly("N", &RiiCpp::GetN)
        .def_property_readonly("nlist", &RiiCpp::GetNumList)
        .def(py::pickle(
            [](const RiiCpp &p){
                return py::make_tuple(p.M_, p.Ks_, p.verbose_, p.codewords_,
                p.coarse_centers_, p.flattened_codes_, p.posting_lists_);
            },
            [](py::tuple t){
                if (t.size() != 7) {
                    throw std::runtime_error("Invalid state when reading pickled item");
                }
                RiiCpp p(t[0].cast<size_t>(), t[1].cast<size_t>(), t[2].cast<bool>());
                p.codewords_ = t[3].cast<std::vector<std::vector<std::vector<float>>>>();
                p.coarse_centers_ = t[4].cast<std::vector<std::vector<unsigned char>>>();
                p.flattened_codes_ = t[5].cast<std::vector<unsigned char>>();
                p.posting_lists_ = t[6].cast<std::vector<std::vector<int>>>();
                return p;
            }
        ));

#ifdef VERSION_INFO
    m.attr("__version__") = VERSION_INFO;
#else
    m.attr("__version__") = "dev";
#endif
}
} // namespace rii
