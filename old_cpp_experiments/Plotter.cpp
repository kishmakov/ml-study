#include "Plotter.h"

#include <matplot/matplot.h>

Plot& Plot::operator+=(double value) {
    ordinates_.push_back(ordinate_);
    values_.push_back(value);

    ordinate_ += step_;

    return *this;
}

void Plotter::draw(const std::string& baseName) {
    std::vector<std::string> legends;

    for (const auto& [label, plot] : plots_) {
        legends.push_back(label);
        matplot::plot(plot.xs(), plot.ys(), plot.format());
        matplot::hold(matplot::on);
    }

    matplot::legend(matplot::gca(), legends);
    matplot::ylabel(ylabel);
    matplot::xlabel(xlabel);
    matplot::title(title_);
    matplot::save("results/" + baseName + ".svg");
    matplot::cla();
}
