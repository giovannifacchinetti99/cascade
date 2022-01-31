// Copyright 2022 Francesco Biscani (bluescarni@gmail.com), Dario Izzo (dario.izzo@gmail.com)
//
// This file is part of the cascade library.
//
// This Source Code Form is subject to the terms of the Mozilla
// Public License v. 2.0. If a copy of the MPL was not distributed
// with this file, You can obtain one at http://mozilla.org/MPL/2.0/.

#ifndef CASCADE_DETAIL_SIM_DATA_HPP
#define CASCADE_DETAIL_SIM_DATA_HPP

#include <array>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <tuple>
#include <type_traits>
#include <utility>
#include <vector>

#include <oneapi/tbb/concurrent_queue.h>
#include <oneapi/tbb/concurrent_vector.h>

#include <heyoka/detail/dfloat.hpp>
#include <heyoka/llvm_state.hpp>
#include <heyoka/taylor.hpp>

#include <cascade/sim.hpp>

namespace cascade
{

namespace detail
{

// Minimal allocator that avoids value-init
// in standard containers. Copies the interface of
// std::allocator and uses it internally:
// https://en.cppreference.com/w/cpp/memory/allocator
template <typename T>
struct no_init_alloc {
    using value_type = T;
    using size_type = std::size_t;
    using difference_type = std::ptrdiff_t;
    using propagate_on_container_move_assignment = std::true_type;
    using is_always_equal = std::true_type;

    constexpr no_init_alloc() noexcept = default;
    constexpr no_init_alloc(const no_init_alloc &) noexcept = default;
    template <class U>
    constexpr no_init_alloc(const no_init_alloc<U> &) noexcept : no_init_alloc()
    {
    }

    [[nodiscard]] constexpr T *allocate(std::size_t n)
    {
        return std::allocator<T>{}.allocate(n);
    }
    constexpr void deallocate(T *p, std::size_t n)
    {
        std::allocator<T>{}.deallocate(p, n);
    }

    template <class U, class... Args>
    void construct(U *p, Args &&...args)
    {
        if constexpr (sizeof...(Args) > 0u) {
            ::new (static_cast<void *>(p)) U(std::forward<Args>(args)...);
        } else {
            ::new (static_cast<void *>(p)) U;
        }
    }
};

template <typename T, typename U>
inline bool operator==(const no_init_alloc<T> &, const no_init_alloc<U> &)
{
    return true;
}

template <typename T, typename U>
inline bool operator!=(const no_init_alloc<T> &, const no_init_alloc<U> &)
{
    return false;
}

} // namespace detail

struct sim::sim_data {
    // The adaptive integrators.
    // NOTE: these are never used directly,
    // we just copy them as necessary to setup
    // the integrator caches below.
    heyoka::taylor_adaptive<double> s_ta;
    heyoka::taylor_adaptive_batch<double> b_ta;

    // The JIT data.
    heyoka::llvm_state state;
    using pta_t = double *(*)(double *, const double *, double) noexcept;
    pta_t pta = nullptr;
    using pssdiff3_t = void (*)(double *, const double *, const double *, const double *, const double *,
                                const double *, const double *) noexcept;
    pssdiff3_t pssdiff3 = nullptr;
    using fex_check_t = void (*)(const double *, const double *, const std::uint32_t *, std::uint32_t *) noexcept;
    fex_check_t fex_check = nullptr;
    using rtscc_t = void (*)(double *, double *, std::uint32_t *, const double *) noexcept;
    rtscc_t rtscc = nullptr;
    using pt1_t = void (*)(double *, const double *) noexcept;
    pt1_t pt1 = nullptr;

    // The superstep size and the number of chunks.
    // NOTE: these are set up at the beginning of each superstep.
    double delta_t = 0;
    unsigned nchunks = 0;
    // Helper to compute the begin and end of a chunk within
    // a superstep for a given collisional timestep.
    std::array<double, 2> get_chunk_begin_end(unsigned, double) const;

    // Buffers that will contain the state at the end of a superstep.
    std::vector<double> final_x, final_y, final_z, final_vx, final_vy, final_vz;

    // The integrator caches.
    // NOTE: the integrators in the caches are those
    // actually used in numerical propagations.
    oneapi::tbb::concurrent_queue<std::unique_ptr<heyoka::taylor_adaptive<double>>> s_ta_cache;
    oneapi::tbb::concurrent_queue<std::unique_ptr<heyoka::taylor_adaptive_batch<double>>> b_ta_cache;

    // The time coordinate.
    heyoka::detail::dfloat<double> time;

    // Particle substep data to be filled in at each superstep.
    struct step_data {
        // Taylor coefficients for the state variables.
        // Each vector contains data for multiple substeps.
        std::vector<double> tc_x, tc_y, tc_z, tc_vx, tc_vy, tc_vz, tc_r;
        // Time coordinates of the end of each substep.
        std::vector<heyoka::detail::dfloat<double>> tcoords;
    };
    std::vector<step_data> s_data;

    // Bounding box data and Morton codes for each particle.
    // NOTE: each vector contains the data for all chunks.
    std::vector<float> x_lb, y_lb, z_lb, r_lb;
    std::vector<float> x_ub, y_ub, z_ub, r_ub;
    std::vector<std::uint64_t> mcodes;

    // The global bounding boxes, one for each chunk.
    std::vector<std::array<float, 4>> global_lb;
    std::vector<std::array<float, 4>> global_ub;

    // The indices vectors for indirect sorting.
    std::vector<size_type> vidx;

    // Versions of AABBs and Morton codes sorted
    // according to vidx.
    std::vector<float> srt_x_lb, srt_y_lb, srt_z_lb, srt_r_lb;
    std::vector<float> srt_x_ub, srt_y_ub, srt_z_ub, srt_r_ub;
    std::vector<std::uint64_t> srt_mcodes;

    // The BVH node struct.
    // NOTE: all members left intentionally uninited
    // for performance reasons.
    struct bvh_node {
        // Particle range.
        std::uint32_t begin, end;
        // Pointers to parent and children nodes.
        std::int32_t parent, left, right;
        // AABB.
        std::array<float, 4> lb, ub;
        // Number of nodes in the current level.
        std::uint32_t nn_level;
        // NOTE: split_idx is used only during tree construction.
        // NOTE: perhaps it's worth it to store it in a separate
        // vector in order to improve performance during
        // tree traversal? Same goes for nn_level.
        int split_idx;
    };

    // The BVH trees, one for each chunk.
    using bvh_tree_t = std::vector<bvh_node, detail::no_init_alloc<bvh_node>>;
    std::vector<bvh_tree_t> bvh_trees;
    // Temporary buffer used in the construction of the BVH trees.
    std::vector<std::vector<std::uint32_t, detail::no_init_alloc<std::uint32_t>>> nc_buffer;
    std::vector<std::vector<std::uint32_t, detail::no_init_alloc<std::uint32_t>>> ps_buffer;
    std::vector<std::vector<std::uint32_t, detail::no_init_alloc<std::uint32_t>>> nplc_buffer;

    // Vectors of broad phase collisions between AABBs.
    std::vector<oneapi::tbb::concurrent_vector<std::pair<size_type, size_type>>> bp_coll;
    // Caches of collision vectors between AABBs.
    std::vector<oneapi::tbb::concurrent_queue<std::vector<std::pair<size_type, size_type>>>> bp_caches;
    // Caches of stacks for the tree traversal during broad
    // phase collision detection.
    std::vector<oneapi::tbb::concurrent_queue<std::vector<std::int32_t>>> stack_caches;

    // Struct for holding polynomial caches used during
    // narrow phase collision detection.
    struct np_data {
        // A RAII helper to extract polys from a cache and
        // return them to the cache upon destruction.
        struct pwrap {
            std::vector<std::vector<double>> &pc;
            std::vector<double> v;

            void back_to_cache();
            std::vector<double> get_poly_from_cache(std::uint32_t);

            explicit pwrap(std::vector<std::vector<double>> &, std::uint32_t);
            pwrap(pwrap &&) noexcept;
            pwrap &operator=(pwrap &&) noexcept;

            // Delete copy semantics.
            pwrap(const pwrap &) = delete;
            pwrap &operator=(const pwrap &) = delete;

            ~pwrap();
        };

        // The working list type used during real root isolation.
        using wlist_t = std::vector<std::tuple<double, double, pwrap>>;
        // The type used to store the list of isolating intervals.
        using isol_t = std::vector<std::tuple<double, double>>;

        // Polynomial buffers used in the construction
        // of the dist square polynomial.
        std::array<std::vector<double>, 7> dist2;
        // Polynomial cache for use during real root isolation.
        // NOTE: it is *really* important that this is declared
        // *before* wlist, because wlist will contain references
        // to and interact with r_iso_cache during destruction,
        // and we must be sure that wlist is destroyed *before*
        // r_iso_cache.
        std::vector<std::vector<double>> r_iso_cache;
        // The working list.
        wlist_t wlist;
        // The list of isolating intervals.
        isol_t isol;
    };
    // NOTE: indirect through a unique_ptr, because for some reason a std::vector of
    // concurrent_queue requires copy ctability of np_data, which is not available due to
    // pwrap semantics.
    std::vector<std::unique_ptr<oneapi::tbb::concurrent_queue<std::unique_ptr<np_data>>>> np_caches;
    // The global vector of collisions.
    // NOTE: use a concurrent vector for the time being,
    // in the assumption that collisions are infrequent.
    // We can later consider solutions with better concurrency
    // if needed (e.g., chunk local concurrent queues of collision vectors).
    oneapi::tbb::concurrent_vector<std::tuple<size_type, size_type, double>> coll_vec;

    // Structures to detect reentry or out-of-domain conditions.
    // NOTE: these cannot be chunk-local because they are written to
    // during the dynamical propagation, which is not happening
    // chunk-by-chunk.
    oneapi::tbb::concurrent_vector<std::tuple<size_type, double>> reentry_vec;
    oneapi::tbb::concurrent_vector<std::tuple<size_type, double>> exit_vec;

    // The callback functors for use when a particle exits the domain or
    // crashes onto the central object.
    struct exit_cb {
        mutable sim_data *sdata = nullptr;
        mutable size_type pidx = 0;

        void operator()(heyoka::taylor_adaptive<double> &, double, int) const;
    };
    struct exit_cb_batch {
        mutable sim_data *sdata = nullptr;
        mutable size_type pidx = 0;

        void operator()(heyoka::taylor_adaptive_batch<double> &, double, int, std::uint32_t) const;
    };
    struct reentry_cb {
        mutable sim_data *sdata = nullptr;
        mutable size_type pidx = 0;

        void operator()(heyoka::taylor_adaptive<double> &, double, int) const;
    };
    struct reentry_cb_batch {
        mutable sim_data *sdata = nullptr;
        mutable size_type pidx = 0;

        void operator()(heyoka::taylor_adaptive_batch<double> &, double, int, std::uint32_t) const;
    };
};

} // namespace cascade

#endif
