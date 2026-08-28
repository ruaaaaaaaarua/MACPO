function plot_h2_delay_case_seed32()
%PLOT_H2_DELAY_CASE_SEED32 Same-policy delayed versus instant H2 delivery.

s = paper_style();
scriptDir = fileparts(mfilename('fullpath'));
figureDir = fileparts(scriptDir);
dataPath = fullfile(figureDir, 'data', 'seed32_delivery_counterfactual.csv');
outputDir = fullfile(figureDir, 'output');
if ~exist(outputDir, 'dir'), mkdir(outputDir); end

t = readtable(dataPath, 'TextType', 'string');
scenarios = ["Delayed delivery", "Instant delivery"];
scenarioColors = [0.230, 0.260, 0.650; 0.250, 0.250, 0.250];
styles = {'-', '--'};

fig = figure('Color', 'w', 'Units', 'centimeters', ...
    'Position', [2, 2, s.doubleColumnCm, 10.8]);
layout = tiledlayout(fig, 2, 2, 'TileSpacing', 'compact', 'Padding', 'compact');

ax1 = nexttile(layout); hold(ax1, 'on');
h = gobjects(2, 1);
for k = 1:2
    rows = t.scenario == scenarios(k);
    h(k) = plot(ax1, t.hour(rows), t.tank_energy_mwh(rows), ...
        'Color', scenarioColors(k, :), 'LineStyle', styles{k}, ...
        'LineWidth', s.lineWidth);
end
format_panel(ax1, '(a)', 'Stored H_2 energy (MWh_{H2})', false);

ax2 = nexttile(layout); hold(ax2, 'on');
for k = 1:2
    rows = t.scenario == scenarios(k);
    stairs(ax2, t.hour(rows), t.pending_energy_mwh(rows), ...
        'Color', scenarioColors(k, :), 'LineStyle', styles{k}, ...
        'LineWidth', s.lineWidth);
end
format_panel(ax2, '(b)', 'Pending H_2 energy (MWh_{H2})', false);

ax3 = nexttile(layout); hold(ax3, 'on');
for k = 1:2
    rows = t.scenario == scenarios(k);
    y = cumsum(t.planned_order_mwh(rows));
    plot(ax3, t.hour(rows), y, 'Color', scenarioColors(k, :), ...
        'LineStyle', styles{k}, 'LineWidth', s.lineWidth);
    text(ax3, 24.2, y(end), sprintf('%.1f', y(end)), ...
        'Color', scenarioColors(k, :), 'FontSize', 7.5, ...
        'HorizontalAlignment', 'left', 'VerticalAlignment', 'middle');
end
format_panel(ax3, '(c)', 'Cumulative planned order (MWh_{H2})', true);
xlim(ax3, [1 25.5]);

ax4 = nexttile(layout); hold(ax4, 'on');
for k = 1:2
    rows = t.scenario == scenarios(k);
    y = cumsum(t.emergency_supply_mwh(rows));
    plot(ax4, t.hour(rows), y, 'Color', scenarioColors(k, :), ...
        'LineStyle', styles{k}, 'LineWidth', s.lineWidth);
    text(ax4, 24.2, y(end), sprintf('%.1f', y(end)), ...
        'Color', scenarioColors(k, :), 'FontSize', 7.5, ...
        'HorizontalAlignment', 'left', 'VerticalAlignment', 'middle');
end
format_panel(ax4, '(d)', 'Cumulative emergency supply (MWh_{H2})', true);
xlim(ax4, [1 25.5]);

lgd = legend(ax1, h, scenarios, 'Orientation', 'horizontal', ...
    'NumColumns', 2, 'Box', 'off');
lgd.Layout.Tile = 'north';

exportgraphics(fig, fullfile(outputDir, 'fig_h2_delay_case_seed32.pdf'), ...
    'ContentType', 'vector');
exportgraphics(fig, fullfile(outputDir, 'fig_h2_delay_case_seed32.png'), ...
    'Resolution', 600);
close(fig);
end

function format_panel(ax, panelLabel, yLabelText, showXLabel)
xlim(ax, [1 24]); xticks(ax, 2:2:24);
if showXLabel
    xlabel(ax, 'Hour');
else
    xticklabels(ax, []);
end
ylabel(ax, yLabelText);
grid(ax, 'on'); ax.GridAlpha = 0.13;
text(ax, 0.02, 0.92, panelLabel, 'Units', 'normalized', 'FontWeight', 'bold');
end
