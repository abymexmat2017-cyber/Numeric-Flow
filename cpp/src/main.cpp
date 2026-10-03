#include "numeric_flow.hpp"

#include <cmath>
#include <cstdlib>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <locale>
#include <optional>
#include <stdexcept>
#include <string>

namespace {

using numeric_flow::StateComponent;

void print_help() {
    std::cout << "Usage: numeric-flow [options]\n"
              << "Bang-bang optimal control with Dormand-Prince 5(4).\n\n"
              << "  --alpha NUMBER         Cost weight parameter (default: 11)\n"
              << "  --target NUMBER        Terminal x in (-0.5, 0.5) (default: -11/24)\n"
              << "  --parameter NUMBER     Fixed costate parameter; skip shooting\n"
              << "  --atol NUMBER          Absolute local tolerance (default: 1e-11)\n"
              << "  --rtol NUMBER          Relative local tolerance (default: 1e-9)\n"
              << "  --max-step NUMBER      Maximum step (default: 0.05)\n"
              << "  --boundary-tol NUMBER  Terminal x tolerance (default: 1e-10)\n"
              << "  --output PATH          CSV path; parent directory must exist\n"
              << "  --help                 Show this help\n\n"
              << "JSON summary is written to stdout. Existing CSV files are replaced.\n";
}

double parse_finite_number(const std::string &option_value) {
    char *parse_end = nullptr;
    const double parsed_value = std::strtod(option_value.c_str(), &parse_end);
    if (parse_end == option_value.c_str() || *parse_end != '\0' || !std::isfinite(parsed_value))
        throw std::invalid_argument("expected a finite number: " + option_value);
    return parsed_value;
}

void write_csv(const numeric_flow::Result &result, const std::string &path) {
    std::ofstream stream;
    stream.exceptions(std::ios::failbit | std::ios::badbit);
    stream.open(path);
    stream.imbue(std::locale::classic());
    stream << std::setprecision(17) << "t,x,y,p_x,p_y,u,integral\n";
    for (const auto &point : result.trajectory)
        stream << point.time << ',' << point.state[StateComponent::position] << ','
               << point.state[StateComponent::velocity] << ','
               << point.state[StateComponent::position_costate] << ','
               << point.state[StateComponent::velocity_costate] << ',' << point.control << ','
               << point.state[StateComponent::accumulated_cost] << '\n';
    stream.close(); // Check errors from flushing as well as opening/writing.
}

void write_summary(const numeric_flow::Result &result, double target_position) {
    const auto &point = result.final();
    std::cout << std::setprecision(17) << "{\n  \"alpha\": " << result.cost_weight_parameter
              << ",\n  \"target\": " << target_position
              << ",\n  \"parameter\": " << result.costate_parameter << ",\n  \"switch_times\": [";
    for (std::size_t i = 0; i < result.switch_times.size(); ++i) {
        if (i)
            std::cout << ", ";
        std::cout << result.switch_times[i];
    }
    std::cout << "],\n  \"final\": {\"t\": " << point.time
              << ", \"x\": " << point.state[StateComponent::position]
              << ", \"y\": " << point.state[StateComponent::velocity]
              << ", \"p_x\": " << point.state[StateComponent::position_costate]
              << ", \"p_y\": " << point.state[StateComponent::velocity_costate]
              << ", \"integral\": " << point.state[StateComponent::accumulated_cost] << "},"
              << "\n  \"residual\": " << point.state[StateComponent::position] - target_position
              << ",\n  \"accepted_steps\": " << result.accepted_steps()
              << ",\n  \"rejected_steps\": " << result.rejected_steps
              << ",\n  \"shooting_iterations\": " << result.shooting_iterations << "\n}\n";
}

} // namespace

int main(int argc, char *argv[]) {
    try {
        std::cout.exceptions(std::ios::failbit | std::ios::badbit);
        std::cout.imbue(std::locale::classic());
        numeric_flow::Options options;
        double cost_weight_parameter = 11, target_position = numeric_flow::default_target_position,
               boundary_tolerance = 1e-10;
        std::optional<double> costate_parameter;
        std::optional<std::string> output_path;
        for (int i = 1; i < argc; ++i) {
            std::string option_name = argv[i], option_value;
            if (option_name == "--help" || option_name == "-h") {
                print_help();
                std::cout.flush();
                return 0;
            }
            const auto equals_position = option_name.find('=');
            if (equals_position != std::string::npos) {
                option_value = option_name.substr(equals_position + 1);
                option_name = option_name.substr(0, equals_position);
            } else {
                if (i + 1 >= argc)
                    throw std::invalid_argument("missing value for " + option_name);
                option_value = argv[++i];
            }
            if (option_name == "--output")
                output_path = option_value;
            else if (option_name == "--alpha")
                cost_weight_parameter = parse_finite_number(option_value);
            else if (option_name == "--target")
                target_position = parse_finite_number(option_value);
            else if (option_name == "--parameter")
                costate_parameter = parse_finite_number(option_value);
            else if (option_name == "--atol")
                options.absolute_tolerance = parse_finite_number(option_value);
            else if (option_name == "--rtol")
                options.relative_tolerance = parse_finite_number(option_value);
            else if (option_name == "--max-step")
                options.max_step_size = parse_finite_number(option_value);
            else if (option_name == "--boundary-tol")
                boundary_tolerance = parse_finite_number(option_value);
            else
                throw std::invalid_argument("unknown option: " + option_name);
        }
        if (target_position <= -0.5 || target_position >= 0.5)
            throw std::invalid_argument("target must be strictly between -0.5 and 0.5");
        if (boundary_tolerance <= 0)
            throw std::invalid_argument("boundary_tol must be positive");
        const auto result =
            costate_parameter
                ? numeric_flow::simulate(*costate_parameter, cost_weight_parameter, options)
                : numeric_flow::solve(cost_weight_parameter, target_position, options,
                                      boundary_tolerance);
        if (output_path)
            write_csv(result, *output_path);
        write_summary(result, target_position);
        std::cout.flush();
    } catch (const std::exception &error) {
        std::cerr << "numeric-flow: error: " << error.what() << '\n';
        return 2;
    }
}
