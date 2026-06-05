#ifndef CRUNCH_PLOTTER_H
#define CRUNCH_PLOTTER_H

#include <map>
#include <string>
#include <utility>
#include <vector>

struct Plot {
    explicit Plot(std::string  format, double step = 1.0) : format_(std::move(format)), values_(0), step_(step){}

    [[nodiscard]] std::string_view format() const { return std::string_view(format_); }

    Plot& operator+=(double value);

    [[nodiscard]] const std::vector<double>& xs() const { return ordinates_; }
    [[nodiscard]] const std::vector<double>& ys() const { return values_; }

private:
    std::string format_;
    std::vector<double> ordinates_;
    std::vector<double> values_;
    const double step_;
    double ordinate_ = 0.0;
};

class Plotter {
public:
    explicit Plotter(std::string title) : title_(std::move(title)) {}

    void add(const std::string& label, Plot plot) {
        plots_.emplace(label, std::move(plot));
    }

    void draw(const std::string& baseName);

    std::string xlabel;
    std::string ylabel;

private:
    const std::string title_;
    std::map<std::string, Plot> plots_;
};

#endif //CRUNCH_PLOTTER_H
