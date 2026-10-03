#pragma once

#include <array>
#include <cstddef>
#include <functional>
#include <stdexcept>
#include <vector>

namespace numeric_flow {

// Named indices keep the ODE state contiguous without magic array offsets.
enum StateComponent : std::size_t {
    position,
    velocity,
    position_costate,
    velocity_costate,
    accumulated_cost
};

using State = std::array<double, 5>;
using DerivativeFunction = std::function<State(double, const State &)>;
inline constexpr double default_target_position = -11.0 / 24.0;

class IllConditionedSwitchingError : public std::runtime_error {
  public:
    using std::runtime_error::runtime_error;
};

struct Options {
    double absolute_tolerance = 1e-11;
    double relative_tolerance = 1e-9;
    double initial_step_size = 0.01;
    double max_step_size = 0.05;
    std::size_t max_step_attempts = 100000;
    void validate() const;
};

struct Step {
    double time;
    State state;
};

struct Integration {
    std::vector<Step> steps;
    std::size_t rejected_steps = 0;
};

struct Point {
    double time;
    State state;
    double control; // Right-continuous at switching times.
};

struct Result {
    double costate_parameter;
    double cost_weight_parameter;
    std::vector<double> switch_times;
    std::vector<Point> trajectory;
    std::size_t rejected_steps = 0;
    int shooting_iterations = 0;
    const Point & final() const { return trajectory.back(); }
    std::size_t accepted_steps() const { return trajectory.size() - 1; }
};

// Forward integration of a smooth RHS. Split discontinuities before calling.
Integration integrate(const DerivativeFunction &derivative, const State &initial_state,
                      double start_time, double end_time, const Options &options = {});
std::vector<double> switching_times(double costate_parameter, double cost_weight_parameter = 11.0);
Result simulate(double costate_parameter, double cost_weight_parameter = 11.0,
                const Options &options = {});
Result solve(double cost_weight_parameter = 11.0, double target_position = default_target_position,
             const Options &options = {}, double boundary_tolerance = 1e-10);

} // namespace numeric_flow
