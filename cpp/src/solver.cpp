#include "numeric_flow.hpp"

#include <algorithm>
#include <cmath>
#include <limits>
#include <optional>
#include <stdexcept>

namespace numeric_flow {
namespace {

constexpr double root_time_tolerance = 2e-15;
constexpr double stage_coefficients[7][7] = {
    {},
    {1.0 / 5},
    {3.0 / 40, 9.0 / 40},
    {44.0 / 45, -56.0 / 15, 32.0 / 9},
    {19372.0 / 6561, -25360.0 / 2187, 64448.0 / 6561, -212.0 / 729},
    {9017.0 / 3168, -355.0 / 33, 46732.0 / 5247, 49.0 / 176, -5103.0 / 18656},
    {35.0 / 384, 0, 500.0 / 1113, 125.0 / 192, -2187.0 / 6784, 11.0 / 84}};
constexpr double stage_time_offsets[] = {0, 1.0 / 5, 3.0 / 10, 4.0 / 5, 8.0 / 9, 1, 1};
constexpr double fifth_order_weights[] = {35.0 / 384, 0, 500.0 / 1113, 125.0 / 192, -2187.0 / 6784,
                                          11.0 / 84,  0};
constexpr double fourth_order_weights[] = {
    5179.0 / 57600, 0, 7571.0 / 16695, 393.0 / 640, -92097.0 / 339200, 187.0 / 2100, 1.0 / 40};

bool has_finite_components(const State &state) {
    return std::all_of(state.begin(), state.end(),
                       [](double value) { return std::isfinite(value); });
}

void validate_parameters(double costate_parameter, double cost_weight_parameter) {
    if (!std::isfinite(costate_parameter) || costate_parameter < 0)
        throw std::invalid_argument("parameter must be finite and nonnegative");
    if (!std::isfinite(cost_weight_parameter) || cost_weight_parameter < 0)
        throw std::invalid_argument("alpha must be finite and nonnegative");
}

bool have_opposite_signs(double first_value, double second_value) {
    return (first_value < 0 && second_value > 0) || (second_value < 0 && first_value > 0);
}

double bisect_root(const std::function<double(double)> &function, double left, double right) {
    double left_value = function(left);
    const double right_value = function(right);
    if (left_value == 0)
        return left;
    if (right_value == 0)
        return right;
    if (!have_opposite_signs(left_value, right_value))
        throw std::runtime_error("root is not bracketed");
    for (int i = 0; i < 100; ++i) {
        const double midpoint = left + (right - left) / 2;
        const double midpoint_value = function(midpoint);
        if (midpoint_value == 0 || right - left <= root_time_tolerance || midpoint == left ||
            midpoint == right)
            return midpoint;
        if (have_opposite_signs(left_value, midpoint_value))
            right = midpoint;
        else {
            left = midpoint;
            left_value = midpoint_value;
        }
    }
    throw std::runtime_error("root bisection did not converge");
}

double cost_weight(double time, double cost_weight_parameter) {
    return 1 / (1 + cost_weight_parameter * std::pow(time, 4));
}

} // namespace

void Options::validate() const {
    if (!std::isfinite(absolute_tolerance) || absolute_tolerance <= 0)
        throw std::invalid_argument("atol must be finite and positive");
    if (!std::isfinite(relative_tolerance) || relative_tolerance < 0)
        throw std::invalid_argument("rtol must be finite and nonnegative");
    if (!std::isfinite(initial_step_size) || initial_step_size <= 0)
        throw std::invalid_argument("initial_step must be finite and positive");
    if (!std::isfinite(max_step_size) || max_step_size <= 0)
        throw std::invalid_argument("max_step must be finite and positive");
    if (max_step_attempts == 0)
        throw std::invalid_argument("max_steps must be positive");
}

Integration integrate(const DerivativeFunction &derivative, const State &initial_state,
                      double start_time, double end_time, const Options &options) {
    options.validate();
    if (!std::isfinite(start_time) || !std::isfinite(end_time) || end_time < start_time)
        throw std::invalid_argument("integration requires finite start <= end");
    if (!has_finite_components(initial_state))
        throw std::invalid_argument("initial state must be finite");
    State state = initial_state;
    double time = start_time;
    double step_size = std::min(options.initial_step_size, options.max_step_size);
    Integration result{{{time, state}}, 0};
    std::size_t step_attempts = 0;
    while (time < end_time) {
        if (step_attempts++ >= options.max_step_attempts)
            throw std::runtime_error("integration step budget exhausted");
        const double remaining_time = end_time - time;
        step_size = std::min({step_size, options.max_step_size, remaining_time});
        if (time + step_size == time)
            throw std::runtime_error("integration step underflow; relax tolerances");
        std::array<State, 7> stage_derivatives{};
        for (std::size_t stage_index = 0; stage_index < 7; ++stage_index) {
            State stage_state{};
            for (std::size_t component_index = 0; component_index < state.size();
                 ++component_index) {
                double weighted_derivative = 0;
                for (std::size_t previous_stage_index = 0; previous_stage_index < stage_index;
                     ++previous_stage_index)
                    weighted_derivative += stage_coefficients[stage_index][previous_stage_index] *
                                           stage_derivatives[previous_stage_index][component_index];
                stage_state[component_index] =
                    state[component_index] + step_size * weighted_derivative;
            }
            stage_derivatives[stage_index] =
                derivative(time + stage_time_offsets[stage_index] * step_size, stage_state);
            if (!has_finite_components(stage_derivatives[stage_index]))
                throw std::runtime_error("RHS returned a nonfinite state");
        }
        State next_state{}, error_estimate{};
        for (std::size_t component_index = 0; component_index < state.size(); ++component_index) {
            double weighted_derivative = 0, weighted_error = 0;
            for (std::size_t previous_stage_index = 0; previous_stage_index < 7;
                 ++previous_stage_index) {
                weighted_derivative += fifth_order_weights[previous_stage_index] *
                                       stage_derivatives[previous_stage_index][component_index];
                weighted_error += (fifth_order_weights[previous_stage_index] -
                                   fourth_order_weights[previous_stage_index]) *
                                  stage_derivatives[previous_stage_index][component_index];
            }
            next_state[component_index] = state[component_index] + step_size * weighted_derivative;
            error_estimate[component_index] = step_size * weighted_error;
        }
        if (!has_finite_components(next_state) || !has_finite_components(error_estimate))
            throw std::runtime_error("nonfinite Runge-Kutta step");
        double scaled_error = 0;
        for (std::size_t component_index = 0; component_index < state.size(); ++component_index) {
            const double error_scale =
                options.absolute_tolerance +
                options.relative_tolerance * std::max(std::abs(state[component_index]),
                                                      std::abs(next_state[component_index]));
            const double component_error = std::abs(error_estimate[component_index]) / error_scale;
            if (!std::isfinite(component_error))
                throw std::runtime_error("nonfinite error estimate");
            scaled_error = std::max(scaled_error, component_error);
        }
        if (scaled_error <= 1) {
            time = step_size == remaining_time ? end_time : time + step_size;
            state = next_state;
            result.steps.push_back({time, state});
        } else
            ++result.rejected_steps;
        const double step_multiplier =
            scaled_error == 0 ? 2.5 : std::clamp(0.9 * std::pow(scaled_error, -0.2), 0.1, 2.5);
        step_size *= step_multiplier;
    }
    return result;
}

std::vector<double> switching_times(double costate_parameter, double cost_weight_parameter) {
    validate_parameters(costate_parameter, cost_weight_parameter);
    // g=(1-t)*(1+alpha*t^4) has at most two stationary points.
    const auto derivative = [cost_weight_parameter](double time) {
        return cost_weight_parameter * (time * time * time * (4 - 5 * time)) - 1;
    };
    std::vector<double> critical_times{0};
    if (derivative(0.6) > 0) {
        critical_times.push_back(bisect_root(derivative, 0, 0.6));
        critical_times.push_back(bisect_root(derivative, 0.6, 0.8));
    } else if (std::abs(derivative(0.6)) <= 64 * std::numeric_limits<double>::epsilon()) {
        critical_times.push_back(0.6);
    }
    critical_times.push_back(1);
    const auto switching_function = [=](double time) {
        return costate_parameter * (1 - time) - cost_weight(time, cost_weight_parameter);
    };
    for (std::size_t i = 1; i + 1 < critical_times.size(); ++i) {
        const double time = critical_times[i];
        const double scale =
            std::abs(costate_parameter * (1 - time)) + cost_weight(time, cost_weight_parameter);
        if (std::abs(switching_function(time)) <=
            64 * std::numeric_limits<double>::epsilon() * scale)
            throw IllConditionedSwitchingError(
                "ill-conditioned switching roots: choose parameters farther from a tangency");
    }
    std::vector<double> candidate_times;
    for (std::size_t i = 1; i < critical_times.size(); ++i) {
        const double left = critical_times[i - 1], right = critical_times[i];
        const double left_value = switching_function(left), right_value = switching_function(right);
        if (left_value == 0 && left > 0 && left < 1)
            candidate_times.push_back(left);
        if (have_opposite_signs(left_value, right_value))
            candidate_times.push_back(bisect_root(switching_function, left, right));
        if (right_value == 0 && right > 0 && right < 1)
            candidate_times.push_back(right);
    }
    std::sort(candidate_times.begin(), candidate_times.end());
    candidate_times.erase(std::unique(candidate_times.begin(), candidate_times.end()),
                          candidate_times.end());
    std::vector<double> switch_times;
    for (std::size_t i = 0; i < candidate_times.size(); ++i) {
        const double time = candidate_times[i];
        const double left = i == 0 ? 0 : candidate_times[i - 1];
        const double right = i + 1 == candidate_times.size() ? 1 : candidate_times[i + 1];
        if (have_opposite_signs(switching_function((left + time) / 2),
                                switching_function((time + right) / 2)))
            switch_times.push_back(time);
    }
    return switch_times;
}

Result simulate(double costate_parameter, double cost_weight_parameter, const Options &options) {
    validate_parameters(costate_parameter, cost_weight_parameter);
    options.validate();
    Result result{costate_parameter,
                  cost_weight_parameter,
                  switching_times(costate_parameter, cost_weight_parameter),
                  {},
                  0,
                  0};
    std::vector<double> interval_boundaries{0};
    interval_boundaries.insert(interval_boundaries.end(), result.switch_times.begin(),
                               result.switch_times.end());
    interval_boundaries.push_back(1);
    State state{0, 0, costate_parameter, costate_parameter, 0};
    std::size_t step_attempts = 0;
    for (std::size_t i = 1; i < interval_boundaries.size(); ++i) {
        const double left = interval_boundaries[i - 1], right = interval_boundaries[i],
                     midpoint = (left + right) / 2;
        const double control =
            costate_parameter * (1 - midpoint) > cost_weight(midpoint, cost_weight_parameter) ? 1
                                                                                              : -1;
        if (result.trajectory.empty())
            result.trajectory.push_back({left, state, control});
        else
            result.trajectory.back() = {left, state, control};
        const auto derivative = [=](double time, const State &state_values) {
            return State{state_values[StateComponent::velocity], control, 0,
                         -state_values[StateComponent::position_costate],
                         control * cost_weight(time, cost_weight_parameter)};
        };
        if (step_attempts >= options.max_step_attempts)
            throw std::runtime_error("integration step budget exhausted");
        Options segment_options = options;
        segment_options.max_step_attempts -= step_attempts;
        const auto segment = integrate(derivative, state, left, right, segment_options);
        for (std::size_t j = 1; j < segment.steps.size(); ++j)
            result.trajectory.push_back({segment.steps[j].time, segment.steps[j].state, control});
        state = segment.steps.back().state;
        result.rejected_steps += segment.rejected_steps;
        step_attempts += segment.steps.size() - 1 + segment.rejected_steps;
    }
    return result;
}

Result solve(double cost_weight_parameter, double target_position, const Options &options,
             double boundary_tolerance) {
    validate_parameters(0, cost_weight_parameter);
    options.validate();
    if (!std::isfinite(target_position) || target_position <= -0.5 || target_position >= 0.5)
        throw std::invalid_argument("target must be finite and strictly between -0.5 and 0.5");
    if (!std::isfinite(boundary_tolerance) || boundary_tolerance <= 0)
        throw std::invalid_argument("boundary_tol must be finite and positive");
    double left = 0, right = 1.28 / (1 + cost_weight_parameter);
    bool is_bracketed = false;
    for (int i = 0; i < 1100; ++i) {
        try {
            if (simulate(right, cost_weight_parameter, options)
                    .final()
                    .state[StateComponent::position] >= target_position) {
                is_bracketed = true;
                break;
            }
        } catch (const IllConditionedSwitchingError &) {
            // A degenerate trial need not make the desired shot degenerate.
        }
        if (right > std::numeric_limits<double>::max() / 2)
            throw std::runtime_error("shooting bracket overflow");
        right *= 2;
    }
    if (!is_bracketed)
        throw std::runtime_error("could not bracket shooting parameter");
    for (int iteration = 1; iteration <= 100; ++iteration) {
        double trial_parameter = 0;
        std::optional<Result> trial_result;
        for (double trial_fraction : {0.5, 0.4, 0.6, 0.3, 0.7}) {
            trial_parameter = left + (right - left) * trial_fraction;
            try {
                trial_result = simulate(trial_parameter, cost_weight_parameter, options);
                break;
            } catch (const IllConditionedSwitchingError &) {
                continue;
            }
        }
        if (!trial_result)
            throw IllConditionedSwitchingError(
                "ill-conditioned switching roots throughout shooting bracket");
        const double residual =
            trial_result->final().state[StateComponent::position] - target_position;
        if (std::abs(residual) <= boundary_tolerance) {
            trial_result->shooting_iterations = iteration;
            return *trial_result;
        }
        if (trial_parameter == left || trial_parameter == right)
            break;
        if (residual < 0)
            left = trial_parameter;
        else
            right = trial_parameter;
    }
    throw std::runtime_error("shooting did not converge; relax boundary_tol");
}

} // namespace numeric_flow
