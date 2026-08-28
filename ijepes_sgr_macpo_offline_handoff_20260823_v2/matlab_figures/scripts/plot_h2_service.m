function plot_h2_service()
%PLOT_H2_SERVICE Two-panel physical-unit hydrogen-service comparison.

s = paper_style();
scriptDir = fileparts(mfilename('fullpath'));
figureDir = fileparts(scriptDir);
dataPath = fullfile(figureDir, 'data', 'hydrogen_service_days.csv');
outputDir = fullfile(figureDir, 'output');
if ~exist(outputDir, 'dir'), mkdir(outputDir); end

t = readtable(dataPath, 'TextType', 'string');
methodKeys = ["GRU-MAPPO", "Fixed-penalty GRU-MAPPO", "SGR-MACPO"];
colors = s.colors([1, 2, 4], :);

fig = figure('Color', 'w', 'Units', 'centimeters', ...
    'Position', [2, 2, s.doubleColumnCm, 7.4]);
layout = tiledlayout(fig, 1, 2, 'TileSpacing', 'compact', 'Padding', 'compact');

ax1 = nexttile(layout);
plot_panel(ax1, t, methodKeys, colors, {'planned_mwh', 'emergency_mwh'}, ...
    {'Planned order', 'Emergency supply'}, '(a) Procurement');
ylabel(ax1, 'Hydrogen energy (MWh_{H2}/day)', 'FontSize', s.labelFontSize);

ax2 = nexttile(layout);
bars = plot_panel(ax2, t, methodKeys, colors, {'late_mwh', 'undelivered_mwh'}, ...
    {'Late order', 'Terminal undelivered'}, '(b) Delivery risk');

lgd = legend(ax2, bars, s.methodLabels, 'Orientation', 'horizontal', ...
    'NumColumns', 3, 'Box', 'off');
lgd.Layout.Tile = 'north';

exportgraphics(fig, fullfile(outputDir, 'fig_hydrogen_service.pdf'), ...
    'ContentType', 'vector');
exportgraphics(fig, fullfile(outputDir, 'fig_hydrogen_service.png'), ...
    'Resolution', 600);
close(fig);

fig = figure('Color', 'w', 'Units', 'centimeters', ...
    'Position', [2, 2, s.singleColumnCm, 10.8]);
layout = tiledlayout(fig, 2, 1, 'TileSpacing', 'compact', 'Padding', 'compact');
ax1 = nexttile(layout);
plot_panel(ax1, t, methodKeys, colors, {'planned_mwh', 'emergency_mwh'}, ...
    {'Planned order', 'Emergency supply'}, '(a) Procurement');
ylabel(ax1, 'Hydrogen energy (MWh_{H2}/day)', 'FontSize', s.labelFontSize);
ax2 = nexttile(layout);
bars = plot_panel(ax2, t, methodKeys, colors, {'late_mwh', 'undelivered_mwh'}, ...
    {'Late order', 'Terminal undelivered'}, '(b) Delivery risk');
ylabel(ax2, 'Hydrogen energy (MWh_{H2}/day)', 'FontSize', s.labelFontSize);
lgd = legend(ax2, bars, {'GRU-MAPPO', 'Fixed-penalty', 'SGR-MACPO'}, ...
    'Orientation', 'horizontal', 'NumColumns', 2, 'Box', 'off');
lgd.Layout.Tile = 'north';
exportgraphics(fig, fullfile(outputDir, 'fig_hydrogen_service_singlecol.pdf'), ...
    'ContentType', 'vector');
exportgraphics(fig, fullfile(outputDir, 'fig_hydrogen_service_singlecol.png'), ...
    'Resolution', 600);
close(fig);
end

function bars = plot_panel(ax, t, methodKeys, colors, fields, labels, panelTitle)
hold(ax, 'on');
means = zeros(2, 3);
values = cell(2, 3);
for category = 1:2
    for method = 1:3
        rows = t.method == methodKeys(method);
        y = t.(fields{category})(rows);
        means(category, method) = mean(y);
        values{category, method} = y;
    end
end

bars = bar(ax, means, 'grouped', 'BarWidth', 0.82);
for method = 1:3
    bars(method).FaceColor = colors(method, :);
    bars(method).EdgeColor = 'none';
    bars(method).FaceAlpha = 0.86;
end

markers = {'o', 's', '^'};
for category = 1:2
    for method = 1:3
        x = bars(method).XEndPoints(category);
        y = values{category, method};
        jitter = [-0.035; 0; 0.035];
        scatter(ax, x + jitter, y, 18, 'Marker', markers{method}, ...
            'MarkerFaceColor', 'w', 'MarkerEdgeColor', colors(method, :), ...
            'LineWidth', 0.75, 'HandleVisibility', 'off');
    end
end

set(ax, 'XTick', 1:2, 'XTickLabel', labels);
title(ax, panelTitle, 'FontWeight', 'normal', 'FontSize', 8.5);
grid(ax, 'on'); ax.GridAlpha = 0.12; ax.XGrid = 'off';
ylim(ax, [0, max(means, [], 'all') * 1.18 + eps]);
end
