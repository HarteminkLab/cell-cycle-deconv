function[fig] = drawDeconvolved(model)
    fig = figure();
    f.Position = [100 100 1200 600];
    tiled_layout = tiledlayout(3, 3);
    tiled_layout.Padding = 'compact';
    tiled_layout.TileSpacing = 'compact';
    fig_title = sprintf("%s, gamma=%.g", model.genename, model.gm);
    title(tiled_layout, fig_title);

    plot_g = model.g;
    plot_pred_g = model.pred_g;
    plot_f = model.f;

    ylim_raw = max(plot_g)*1.1;
    ylim_deconv = max(plot_f)*1.25;
    ylim_low = -max(plot_g)*0.1;

    line_width = 3.5;

    % -------------------------------------------------------------------------------

    % WT1 Raw and fitted plot
    num_timepoints = size(model.timepoints, 2);
    nexttile;
    hold on;
    plot(model.timepoints(1, :), plot_g(1, 1:num_timepoints), ...
        '-', 'color', colorForName('raw'), ...
        'LineWidth', line_width);
    plot(model.timepoints(1, :), plot_pred_g(1:num_timepoints, 1), ...
        '-', 'color', colorForName('fit'),  ...
        'LineWidth', line_width);
    ylim([ylim_low  ylim_raw]);
    yticks([]);
    xticks([]);
    ylabel('WT 1');
    hold off;

    % -------------------------------------------------------------------------------

    % Initial plot

    initialTimepointsList = model.intervals.initialTimepointsList;
    f_initial_list = model.f_initial_list;

    initialTimepointsList

    nexttile;
    hold on;
    plot(initialTimepointsList{1}(1:end-1), plot_f(f_initial_list{1}(1:end)), '-', 'color',  ...
        colorForName('R'), 'LineWidth', line_width);
    plot(initialTimepointsList{2}(1:end-1), plot_f(f_initial_list{2}(1:end)), '-', 'color',  ...
        colorForName('CG1'), 'LineWidth', line_width);
    plot(initialTimepointsList{3}(1:end-1), plot_f(f_initial_list{3}(1:end)), '-', 'color',  ...
        colorForName('postG1'), 'LineWidth', line_width);
    ylim([ylim_low  ylim_deconv]);
    yticks([]);
    xticks([]);
    ylabel('Initial, f\_i');
    hold off;

    % -------------------------------------------------------------------------------

    % Mother plot

    topTimepointsList = model.intervals.topTimepointsList;
    f_top_list = model.f_top_list;

    nexttile;
    hold on;
    ylim([ylim_low  ylim_deconv]);
    yticks([]); 

    cg1_timepoints = topTimepointsList{1}(1:end-1);
    post_g1_timepoints = topTimepointsList{2}(1:end-1);

    plot(cg1_timepoints, plot_f([f_top_list{1}]), '-', 'color', colorForName('CG1'),  ...
        'LineWidth', line_width);
    plot(post_g1_timepoints, plot_f([f_top_list{2}]), '-', 'color', colorForName('postG1'), ...
        'LineWidth', line_width);
    ylabel('Top/Mother, f\_t');
    xlim([min(topTimepointsList{1}) max(topTimepointsList{2})]);
    hold off;

    cg1_tick = cg1_timepoints(length(cg1_timepoints)/2);
    post_g1_tick = post_g1_timepoints(round(length(post_g1_timepoints)/2));

    set(gca, 'xtick', [cg1_tick post_g1_tick]);
    set(gca, 'xticklabel', {"CG1" "postG1"});

    % -------------------------------------------------------------------------------

    % WT1 Raw and fitted plot
    num_timepoints = size(model.timepoints, 2);

    nexttile;
    hold on;
    plot(model.timepoints(1, :), plot_g(1, num_timepoints:end-1), ...
        '-', 'color', colorForName('raw'), ...
        'LineWidth', line_width);
    plot(model.timepoints(1, :), plot_pred_g(num_timepoints:end-1, 1), ...
        '-', 'color', colorForName('fit'),  ...
        'LineWidth', line_width);
    ylim([ylim_low  ylim_raw]);
    yticks([]);
    xticks([]);
    ylabel('WT 1');
    hold off;

    % f vector
    nexttile;
    hold on;
    ylim([ylim_low  ylim_deconv]);
    yticks([]);
    xticks([]);
    ylabel('f');

    cc_states = {"R" "CG1" "DG1" "postG1"};

    abs_indices = 1:1:size(model.H, 2);
    for i = 1:length(model.Hpos)
        indices = h_indices_for_name(model, cc_states{i});
        plot(abs_indices(indices), model.f(indices), '-', 'color', colorForName(cc_states{i}), ...
            'LineWidth', line_width);
    end
    xlim([1 size(model.H, 2)]);
    hold off;

    % -------------------------------------------------------------------------------

    % Daughter plot
    nexttile;
    hold on;

    bottomTimepointsList = model.intervals.bottomTimepointsList;
    f_bottom = model.f_bottom_list;

    dg1_timepoints = bottomTimepointsList{1}(1:end-1);
    post_g1_timepoints = bottomTimepointsList{2}(3:end);

    plot(dg1_timepoints, plot_f(f_bottom{1}), '-', 'color', colorForName('DG1'),  ...
        'LineWidth', line_width);
    plot(post_g1_timepoints, plot_f(f_bottom{2}(1:end-1)), '-', 'color',  ...
        colorForName('postG1'), 'LineWidth', line_width);
    ylim([ylim_low  ylim_deconv]);
    yticks([]);
    ylabel('Bottom/Daughter, f\_b');
    hold off;

    % -------------------------------------------------------------------------------

    nexttile;
    axis off;

    % H heatmap
    nexttile;
    hm = heatmap(model.H);
    hm.GridVisible = 'off';
    hm.ColorbarVisible = 'off';
    ylabel('H');
    Ax = gca;
    Ax.XDisplayLabels = nan(size(Ax.XDisplayData));
    Ax.YDisplayLabels = nan(size(Ax.YDisplayData));

    % -------------------------------------------------------------------------------

    % f vector
    nexttile;
    hold on;
    ylim([ylim_low  ylim_deconv]);
    yticks([]);
    xticks([]);
    ylabel('Single Cell Profile');
    
    cc_states = {"R", "CG1", "postG1", "DG1"};
    last = 0;
    for i = 1:length(cc_states)
        cc_state = cc_states{i};
        y = h_indices_for_name(model, cc_state);
        x = last+1:1:last+length(y);
        plot(x, model.f(y), ':', 'color', colorForName(cc_state),  ...
                'LineWidth', line_width);
        last = last+length(y);
    end
    xlim([1 max(x)]);
    hold off;

    return


function[h_indices] = h_indices_for_name(model, cc_state)
    % TODO: hard-coded the cc state names, refactor to read the model label names
    cc_states = ["R" "CG1" "DG1" "postG1"];
    h_pos_indices = [1 2 3 4];
    h_dict = dictionary(cc_states, h_pos_indices);
    h_index = h_dict(cc_state);
    h_indices = model.Hpos{h_index}(1):1:model.Hpos{h_index}(2);
    return


function[color] = colorForName(color_name)

    color_names = ["raw" "fit" "R" "CG1" "DG1" "postG1"];
    colors = {[158 50 50]/255., ...
         [145 180 98]/255., ...
         [199 148 144]/255., ...
         [147 168 198]/255., ...
         [165 197 204]/255., ...
         [223 192 158]/255., ...
        };
    color_mapping = dictionary(color_names, colors);
    color = color_mapping(color_name);
    color = color{1};
    return
