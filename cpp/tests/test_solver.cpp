#include "numeric_flow.hpp"

#include <cmath>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <string>

using namespace numeric_flow;

void check(bool condition, const std::string &message) {
    if (!condition)
        throw std::runtime_error(message);
}

void check_near(double actual, double expected, double tolerance, const std::string &message) {
    check(std::isfinite(actual) && std::abs(actual - expected) <= tolerance, message);
}

template <typename Function> void expect_exception(Function function, const std::string &message) {
    bool exception_thrown = false;
    try {
        function();
    } catch (const std::exception &) {
        exception_thrown = true;
    }
    check(exception_thrown, message);
}

int main() {
    try {
        const auto constant_control_result = simulate(0.0, 0.0);
        check(constant_control_result.switch_times.empty(), "constant control has no switches");
        for (const auto &point : constant_control_result.trajectory) {
            check_near(point.state[StateComponent::position], -point.time * point.time / 2, 1e-13,
                       "constant acceleration x");
            check_near(point.state[StateComponent::velocity], -point.time, 1e-13,
                       "constant acceleration y");
            check_near(point.state[StateComponent::accumulated_cost], -point.time, 1e-13,
                       "constant cost");
        }
        const auto analytic_result = solve(0.0);
        const double expected_switch_time = 1 - std::sqrt(23.0 / 24);
        check_near(analytic_result.costate_parameter, 1 / std::sqrt(23.0 / 24), 3e-10,
                   "analytic parameter");
        check(analytic_result.switch_times.size() == 1, "one switch for alpha=0");
        check_near(analytic_result.switch_times[0], expected_switch_time, 1e-10, "analytic switch");
        check_near(analytic_result.final().state[StateComponent::position], -11.0 / 24, 1e-10,
                   "analytic terminal x");
        check_near(analytic_result.final().state[StateComponent::accumulated_cost],
                   2 * expected_switch_time - 1, 2e-10, "analytic integral");
        check_near(analytic_result.final().state[StateComponent::velocity_costate], 0, 1e-13,
                   "terminal costate");

        Options large_step_options;
        large_step_options.initial_step_size = large_step_options.max_step_size = 1;
        const auto three_switch_result = simulate(1.05, 11, large_step_options);
        check(three_switch_result.switch_times.size() == 3, "all three switches found");
        for (std::size_t i = 1; i < three_switch_result.trajectory.size(); ++i) {
            const auto &previous_point = three_switch_result.trajectory[i - 1];
            const auto &current_point = three_switch_result.trajectory[i];
            const double step_size = current_point.time - previous_point.time;
            check(step_size > 0, "strictly increasing time");
            check_near(current_point.state[StateComponent::position],
                       previous_point.state[StateComponent::position] +
                           previous_point.state[StateComponent::velocity] * step_size +
                           previous_point.control * step_size * step_size / 2,
                       1e-12, "step does not cross a switch");
        }
        check(switching_times(1, 0).empty(), "endpoint root excluded");
        expect_exception([] { switching_times(1.5625, 125.0 / 27); },
                         "stationary inflection detected");
        expect_exception([] { switching_times(0.8922274495253677, 11); },
                         "unresolved close roots detected");
        const auto large_weight_result = solve(1e30);
        check_near(large_weight_result.final().state[StateComponent::position], -11.0 / 24, 1e-10,
                   "large alpha shooting scale");
        check_near(solve(1e30, 0.499999999).final().state[StateComponent::position], 0.499999999,
                   1e-10, "large alpha near endpoint target");

        const auto result = solve();
        check(result.switch_times.size() == 2, "default has two switches");
        check_near(result.final().state[StateComponent::position], -11.0 / 24, 1e-10,
                   "default terminal constraint");
        std::vector<double> boundaries{0};
        boundaries.insert(boundaries.end(), result.switch_times.begin(), result.switch_times.end());
        boundaries.push_back(1);
        double integral = 0;
        for (std::size_t j = 1; j < boundaries.size(); ++j) {
            const double interval_start = boundaries[j - 1], interval_end = boundaries[j];
            const double midpoint = (interval_start + interval_end) / 2;
            const double control =
                result.costate_parameter * (1 - midpoint) > 1 / (1 + 11 * std::pow(midpoint, 4))
                    ? 1
                    : -1;
            const auto cost_integrand = [control](double time) {
                return control / (1 + 11 * std::pow(time, 4));
            };
            const int quadrature_intervals = 2000;
            const double step_size = (interval_end - interval_start) / quadrature_intervals;
            double weighted_sum = cost_integrand(interval_start) + cost_integrand(interval_end);
            for (int i = 1; i < quadrature_intervals; ++i)
                weighted_sum += (i % 2 ? 4 : 2) * cost_integrand(interval_start + i * step_size);
            integral += step_size * weighted_sum / 3;
        }
        check_near(result.final().state[StateComponent::accumulated_cost], integral, 2e-9,
                   "independent Simpson integral");

        Options strict_options;
        strict_options.absolute_tolerance = strict_options.relative_tolerance = 1e-12;
        strict_options.initial_step_size = strict_options.max_step_size = 1;
        const auto exponential_result =
            integrate([](double, const State &state_values) { return state_values; },
                      State{1, 1, 1, 1, 1}, 0, 1, strict_options);
        check_near(exponential_result.steps.back().state[StateComponent::position], std::exp(1.0),
                   2e-11, "exponential ODE");
        check(exponential_result.rejected_steps > 0, "error controller rejects large step");
        const auto short_interval_result =
            integrate([](double, const State &) { return State{2, 0, 0, 0, 0}; }, State{}, 0, 1e-8);
        check_near(short_interval_result.steps.back().state[StateComponent::position], 2e-8, 1e-20,
                   "short final step");
        check(short_interval_result.steps.back().time == 1e-8, "exact final time");
        Options limited_options;
        limited_options.max_step_attempts = 2;
        expect_exception([&] { simulate(0, 11, limited_options); }, "step budget enforced");
        Options invalid_options;
        invalid_options.absolute_tolerance = 0;
        expect_exception([&] { simulate(0, 11, invalid_options); }, "zero atol rejected");
        expect_exception([] { solve(-1); }, "negative alpha rejected");
        expect_exception([] { solve(11, 0.5); }, "unreachable target rejected");
        expect_exception([] { simulate(std::numeric_limits<double>::infinity()); },
                         "infinity rejected");
        expect_exception(
            [] {
                integrate(
                    [](double, const State &) {
                        return State{std::numeric_limits<double>::infinity(), 0, 0, 0, 0};
                    },
                    State{}, 0, 1);
            },
            "nonfinite derivative rejected");
        std::cout << "All C++ numerical checks passed\n";
    } catch (const std::exception &error) {
        std::cerr << "Test failure: " << error.what() << '\n';
        return 1;
    }
}
