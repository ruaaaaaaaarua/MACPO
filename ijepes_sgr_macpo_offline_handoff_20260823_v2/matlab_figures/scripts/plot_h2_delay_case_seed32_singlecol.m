function plot_h2_delay_case_seed32_singlecol()
%PLOT_H2_DELAY_CASE_SEED32_SINGLECOL H2 CDA-to-delivery mechanism.

s = paper_style();
scriptDir = fileparts(mfilename('fullpath'));
figureDir = fileparts(scriptDir);
dataPath = fullfile(figureDir, 'data', 'seed32_h2_cda_delivery_rollouts.csv');
outputDir = fullfile(figureDir, 'output');
t = readtable(dataPath, 'TextType', 'string');

scenarios = ["Delayed delivery", "Instant delivery"];
colors = [0.230, 0.260, 0.650; 0.250, 0.250, 0.250];
styles = {'-', '--'};
delayedRows = t.scenario == scenarios(1);
instantRows = t.scenario == scenarios(2);
hours = t.hour(delayedRows);

fig = figure('Color', 'w', 'Units', 'centimeters', ...
    'Position', [2, 2, s.singleColumnCm, 13.0]);
layout = tiledlayout(fig, 4, 1, 'TileSpacing', 'compact', 'Padding', 'compact');

ax1 = nexttile(layout); hold(ax1, 'on');
hCda = stairs(ax1, hours, t.h2_cda_mwh(delayedRows), ':', ...
    'Color', [0.050, 0.500, 0.620], 'LineWidth', 1.2 * s.lineWidth);
hPlanDelayed = stairs(ax1, hours, t.planned_external_order_mwh(delayedRows), ...
    'Color', colors(1, :), 'LineStyle', styles{1}, 'LineWidth', s.lineWidth);
hPlanInstant = stairs(ax1, hours, t.planned_external_order_mwh(instantRows), ...
    'Color', colors(2, :), 'LineStyle', styles{2}, 'LineWidth', s.lineWidth);
format_time_panel(ax1, '(a)', {'Cleared/scheduled H_2';'energy (MWh_{H2})'}, false);

ax2 = nexttile(layout); hold(ax2, 'on');
hDelayed = stairs(ax2, hours, t.pending_energy_mwh(delayedRows), ...
    'Color', colors(1, :), 'LineStyle', styles{1}, 'LineWidth', s.lineWidth);
hInstant = stairs(ax2, hours, t.pending_energy_mwh(instantRows), ...
    'Color', colors(2, :), 'LineStyle', styles{2}, 'LineWidth', s.lineWidth);
format_time_panel(ax2, '(b)', 'Pending H_2 (MWh_{H2})', false);

ax3 = nexttile(layout); hold(ax3, 'on');
fill_between(ax3, hours, t.tank_energy_mwh(delayedRows), ...
    t.tank_energy_mwh(instantRows), colors(1, :));
for k = 1:2
    rows = t.scenario == scenarios(k);
    plot(ax3, t.hour(rows), t.tank_energy_mwh(rows), ...
        'Color', colors(k, :), 'LineStyle', styles{k}, 'LineWidth', s.lineWidth);
end
format_time_panel(ax3, '(c)', 'Stored H_2 (MWh_{H2})', false);

ax4 = nexttile(layout); hold(ax4, 'on');
fill_between(ax4, hours, t.electrolyzer_power_mw(delayedRows), ...
    t.electrolyzer_power_mw(instantRows), colors(1, :));
for k = 1:2
    rows = t.scenario == scenarios(k);
    plot(ax4, t.hour(rows), t.electrolyzer_power_mw(rows), ...
        'Color', colors(k, :), 'LineStyle', styles{k}, 'LineWidth', s.lineWidth);
end
format_time_panel(ax4, '(d)', 'Electrolyzer power (MW)', true);

lgdMarket = legend(ax1, [hCda, hPlanDelayed, hPlanInstant], ...
    {'Internal H_2 CDA', 'External order: delayed', 'External order: instant'}, ...
    'Orientation', 'horizontal', 'NumColumns', 2, 'Box', 'off');
lgdMarket.Layout.Tile = 'north';
legend(ax2, [hDelayed, hInstant], scenarios, 'Location', 'northeast', ...
    'Box', 'off', 'FontSize', 6.5);

linkaxes([ax1, ax2, ax3, ax4], 'x');
exportgraphics(fig, fullfile(outputDir, 'fig_h2_delay_case_seed32_singlecol.pdf'), ...
    'ContentType', 'vector');
exportgraphics(fig, fullfile(outputDir, 'fig_h2_delay_case_seed32_singlecol.png'), ...
    'Resolution', 600);
close(fig);
end

function fill_between(ax, x, y1, y2, color)
fill(ax, [x; flipud(x)], [y1; flipud(y2)], color, ...
    'FaceAlpha', 0.10, 'EdgeColor', 'none', 'HandleVisibility', 'off');
end

function format_time_panel(ax, panelLabel, yLabelText, showXLabel)
xlim(ax, [1 24]); xticks(ax, 2:2:24);
if showXLabel
    xlabel(ax, 'Hour');
else
    xticklabels(ax, []);
end
ylabel(ax, yLabelText);
grid(ax, 'on'); ax.GridAlpha = 0.13;
text(ax, 0.02, 0.90, panelLabel, 'Units', 'normalized', 'FontWeight', 'bold');
end
