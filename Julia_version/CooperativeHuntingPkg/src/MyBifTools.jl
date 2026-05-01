module MyBifTools

using Plots

# plotting defaults
default(
    guidefontsize=22,
    tickfontsize=12,
    legendfontsize=14,
    linewidth=2,
    grid = false,
    fontfamily="Computer Modern"
)

export plot_segments!

    function plot_segments!(x_vec, y_vec, stability_vec;
        color = :black, plot_legend = true)
        """
        Plots line segments based on stability.

        # Arguments
        - `x_vec`: A vector of x-coordinates.
        - `y_vec`: A vector of y-coordinates.
        - `stability_vec`: A vector of booleans indicating stability for each point.
                        `true` means stable, `false` means unstable.

        # Returns
        - A plot with solid lines for stable segments and dashed lines for unstable segments.
        """
        #plt = plot()  # Initialize the plot

        # Initialize temporary vectors for segments
        x_segment = []
        y_segment = []
        current_stability = stability_vec[1]  # Start with the stability of the first point

        # Track whether the labels have been added to the legend
        stable_label_added = false
        unstable_label_added = false
        for i in 1:length(stability_vec)
            # Add the current point to the segment
            push!(x_segment, x_vec[i])
            push!(y_segment, y_vec[i])

            # Check if stability changes or if it's the last point
            if i == length(stability_vec) || stability_vec[i] != current_stability
                # Plot the current segment
                linestyle = current_stability ? :solid : :dash
                linewidth = current_stability ? 6 : 2
                plot!(x_segment, y_segment, linestyle=linestyle, linewidth = linewidth, label = "",
                        legend = false, color = color)

                # Start a new segment
                x_segment = [x_vec[i]]
                y_segment = [y_vec[i]]
                current_stability = stability_vec[i]
            end
        end
        if plot_legend
            # manually add legend
            plot!([], [], linestyle=:solid, color=:black, label="stable", legend = true)
            plot!([], [], linestyle=:dash, color=:black, label="unstable", legend = true)
        end

    end

end
