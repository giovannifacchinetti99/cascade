// Copyright 2022 Francesco Biscani (bluescarni@gmail.com), Dario Izzo (dario.izzo@gmail.com)
//
// This file is part of the cascade.py library.
//
// This Source Code Form is subject to the terms of the Mozilla
// Public License v. 2.0. If a copy of the MPL was not distributed
// with this file, You can obtain one at http://mozilla.org/MPL/2.0/.

#include <cstddef>
#include <cstdint>
#include <iostream>
#include <memory>
#include <optional>
#include <sstream>
#include <stdexcept>
#include <utility>
#include <variant>
#include <vector>

#include <boost/numeric/conversion/cast.hpp>

#include <fmt/core.h>

#include <oneapi/tbb/global_control.h>

#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include <heyoka/expression.hpp>

#include <cascade/sim.hpp>

#include "logging.hpp"

namespace cascade_py::detail
{

namespace
{

std::optional<oneapi::tbb::global_control> tbb_gc;

} // namespace

} // namespace cascade_py::detail

PYBIND11_MODULE(core, m)
{
    namespace py = pybind11;
    using namespace pybind11::literals;

    namespace hy = heyoka;

    using namespace cascade;
    namespace cpy = cascade_py;

    // Expose the logging setter functions.
    cpy::expose_logging_setters(m);

    // outcome enum.
    py::enum_<outcome>(m, "outcome")
        .value("success", outcome::success)
        .value("time_limit", outcome::time_limit)
        .value("collision", outcome::collision)
        .value("reentry", outcome::reentry)
        .value("exit", outcome::exit)
        .value("err_nf_state", outcome::err_nf_state);

    // Dynamics submodule.
    auto dynamics_module = m.def_submodule("dynamics");
    dynamics_module.def("kepler", &dynamics::kepler, "mu"_a = 1.);

    // sim class.
    py::class_<sim>(m, "sim", py::dynamic_attr{})
        .def(py::init<>())
        .def(py::init([](const py::array_t<double> &state, double ct,
                         std::optional<std::vector<std::pair<hy::expression, hy::expression>>> dyn_,
                         std::optional<std::variant<double, std::vector<double>>> c_radius_,
                         std::optional<double> d_radius_, const std::optional<py::array_t<double>> &pars_,
                         std::optional<double> tol_, bool ha, std::uint32_t n_par_ct, double conj_thresh) {
                 // Check the input state.
                 if (state.ndim() != 2) {
                     throw std::invalid_argument(fmt::format(
                         "The input state must have 2 dimensions, but instead an array with {} dimension(s) was provided",
                         state.ndim()));
                 }

                 if (state.shape(1) != 7) {
                     throw std::invalid_argument(fmt::format(
                         "An input state with 7 columns is expected, but the number of columns is instead {}",
                         state.shape(1)));
                 }

                 // Fetch the number of particles.
                 const auto nparts = state.shape(0);

                 // Flatten out the state vector.
                 auto state_vec = py::cast<std::vector<double>>(state.attr("flatten")());

                 // Init the dynamics.
                 auto dyn = dyn_ ? std::move(*dyn_) : std::vector<std::pair<hy::expression, hy::expression>>{};

                 // Init c_radius.
                 auto c_radius = c_radius_ ? std::move(*c_radius_) : 0.;

                 // Init d_radius.
                 auto d_radius = d_radius_ ? *d_radius_ : 0.;

                 // Prepare the pars vector.
                 std::vector<double> pars_vec;

                 if (pars_) {
                     const auto &pars = *pars_;

                     // Check the parameters array.
                     if (pars.ndim() != 2) {
                         throw std::invalid_argument(
                             fmt::format("The input array of parameter values must have 2 dimensions, but "
                                         "instead an array with {} dimension(s) was provided",
                                         pars.ndim()));
                     }

                     if (pars.shape(0) != nparts) {
                         throw std::invalid_argument(fmt::format("An input array of parameter values with {} row(s) is "
                                                                 "expected, but the number of rows is instead {}",
                                                                 nparts, pars.shape(0)));
                     }

                     // Flatten it out into pars_vec.
                     pars_vec = py::cast<std::vector<double>>(pars.attr("flatten")());
                 }

                 // Prepare the tolerance.
                 auto tol = tol_ ? *tol_ : 0.;

                 // NOTE: might have to re-check this if we ever offer the
                 // option to define event callbacks in the dynamics.
                 py::gil_scoped_release release;

                 return std::visit(
                     [&](auto &&cr_val) {
                         return sim(std::move(state_vec), ct, kw::dyn = std::move(dyn),
                                    kw::c_radius = std::forward<decltype(cr_val)>(cr_val), kw::d_radius = d_radius,
                                    kw::pars = std::move(pars_vec), kw::tol = tol, kw::high_accuracy = ha, kw::n_par_ct = n_par_ct, kw::conj_thresh = conj_thresh);
                     },
                     std::move(c_radius));
             }),
             "state"_a, "ct"_a, "dyn"_a = py::none{}, "c_radius"_a = py::none{}, "d_radius"_a = py::none{},
             "pars"_a = py::none{}, "tol"_a = py::none{}, "high_accuracy"_a = false, "n_par_ct"_a = 1, "conj_thresh"_a = 0.)
        .def_property_readonly("interrupt_info", &sim::get_interrupt_info)
        .def_property("time", &sim::get_time, &sim::set_time)
        .def_property("ct", &sim::get_ct, &sim::set_ct)
        .def_property("n_par_ct", &sim::get_n_par_ct, &sim::set_n_par_ct)
        .def_property("conj_thresh", &sim::get_conj_thresh, &sim::set_conj_thresh)
        .def_property_readonly("nparts", &sim::get_nparts)
        .def_property_readonly("npars", &sim::get_npars)
        .def_property_readonly("tol", &sim::get_tol)
        .def_property_readonly("high_accuracy", &sim::get_high_accuracy)
        .def_property_readonly("c_radius", &sim::get_c_radius)
        .def_property_readonly("d_radius", &sim::get_d_radius)
        .def(
            "step",
            [](sim &s) {
                // NOTE: might have to re-check this if we ever offer the
                // option to define event callbacks in the dynamics.
                py::gil_scoped_release release;

                return s.step();
            })
        .def(
            "propagate_until",
            [](sim &s, double t) {
                // NOTE: might have to re-check this if we ever offer the
                // option to define event callbacks in the dynamics.
                py::gil_scoped_release release;

                return s.propagate_until(t);
            },
            "t"_a)
        // Expose the state getters.
        .def_property_readonly("state",
                               [](const sim &s) {
                                   // Fetch a shared ptr to the internal state vector.
                                   auto sptr = s._get_state_ptr();

                                   // Box it into a unique_ptr.
                                   // NOTE: this looks funky, but it's only to cope with the fact
                                   // that the capsule mechanism only works in terms of raw pointers.
                                   auto uptr = std::make_unique<decltype(sptr)>(std::move(sptr));

                                   // Create the capsule from the raw pointer in uptr. The destructor
                                   // will put the raw pointer back into a unique_ptr, which, when destructed,
                                   // will invoke the destructor of the shared pointer, ultimately decreasing
                                   // the reference count of the state vector.
                                   py::capsule state_caps(uptr.get(), [](void *ptr) {
                                       std::unique_ptr<decltype(sptr)> vptr(static_cast<decltype(sptr) *>(ptr));
                                   });

                                   // NOTE: at this point, the capsule has been created successfully (including
                                   // the registration of the destructor). We can thus release ownership from uptr,
                                   // as now the capsule is responsible for destroying its contents. If the capsule
                                   // constructor throws, the destructor function is not registered/invoked, and the
                                   // destructor of uptr will take care of cleaning up.
                                   auto *ptr = uptr.release();

                                   auto ret = py::array_t<double>(
                                       py::array::ShapeContainer{boost::numeric_cast<py::ssize_t>(s.get_nparts()),
                                                                 static_cast<py::ssize_t>(7)},
                                       (**ptr).data(), std::move(state_caps));

                                   return ret;
                               })
        .def_property_readonly("pars",
                               [](const sim &s) {
                                   // NOTE: same idea as in the pars getter.
                                   auto pptr = s._get_pars_ptr();

                                   auto uptr = std::make_unique<decltype(pptr)>(std::move(pptr));

                                   py::capsule pars_caps(uptr.get(), [](void *ptr) {
                                       std::unique_ptr<decltype(pptr)> vptr(static_cast<decltype(pptr) *>(ptr));
                                   });

                                   auto *ptr = uptr.release();

                                   auto ret = py::array_t<double>(
                                       py::array::ShapeContainer{boost::numeric_cast<py::ssize_t>(s.get_nparts()),
                                                                 boost::numeric_cast<py::ssize_t>(s.get_npars())},
                                       (**ptr).data(), std::move(pars_caps));

                                   return ret;
                               })
        .def(
            "set_new_state_pars",
            [](sim &s, const py::array_t<double> &new_state, const std::optional<py::array_t<double>> &new_pars_) {
                // Check the new state.
                if (new_state.ndim() != 2) {
                    throw std::invalid_argument(fmt::format("The input state must have 2 dimensions, but instead an "
                                                            "array with {} dimension(s) was provided",
                                                            new_state.ndim()));
                }

                if (new_state.shape(1) != 7) {
                    throw std::invalid_argument(fmt::format(
                        "An input state with 7 columns is expected, but the number of columns is instead {}",
                        new_state.shape(1)));
                }

                // Flatten out.
                auto state_vec = py::cast<std::vector<double>>(new_state.attr("flatten")());

                // Prepare the pars vector.
                std::vector<double> pars_vec;

                if (new_pars_) {
                    const auto &new_pars = *new_pars_;

                    // Check the new pars vector.
                    if (new_pars.ndim() != 2) {
                        throw std::invalid_argument(fmt::format(
                            "The input array of parameter values must have 2 dimensions, but instead an array "
                            "with {} dimension(s) was provided",
                            new_pars.ndim()));
                    }

                    if (boost::numeric_cast<std::uint32_t>(new_pars.shape(1)) != s.get_npars()) {
                        throw std::invalid_argument(fmt::format("An array of parameter values with {} column(s) is "
                                                                "expected, but the number of columns is instead {}",
                                                                s.get_npars(), new_pars.shape(1)));
                    }

                    // Flatten out into pars_vec.
                    pars_vec = py::cast<std::vector<double>>(new_pars.attr("flatten")());
                }

                py::gil_scoped_release release;

                s.set_new_state_pars(std::move(state_vec), std::move(pars_vec));
            },
            "new_state"_a, "new_pars"_a = py::none{})
        // Remove particles.
        .def("remove_particles", &sim::remove_particles)
        // Repr.
        .def("__repr__", [](const sim &s) {
            std::ostringstream oss;
            oss << s;
            return oss.str();
        });

    m.def("set_nthreads", [](std::size_t n) {
        if (n == 0u) {
            cpy::detail::tbb_gc.reset();
        } else {
            cpy::detail::tbb_gc.emplace(oneapi::tbb::global_control::max_allowed_parallelism, n);
        }
    });

    m.def("get_nthreads", []() {
        return oneapi::tbb::global_control::active_value(oneapi::tbb::global_control::max_allowed_parallelism);
    });

    // Make sure the TBB control structure is cleaned
    // up before shutdown.
    auto atexit = py::module_::import("atexit");
    atexit.attr("register")(py::cpp_function([]() {
#if !defined(NDEBUG)
        std::cout << "Cleaning up the TBB control structure" << std::endl;
#endif
        cpy::detail::tbb_gc.reset();
    }));
}
