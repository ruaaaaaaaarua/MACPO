function plot_voltage_case_seed32()
%PLOT_VOLTAGE_CASE_SEED32 CDA-to-network mechanism for evaluation day 32.

s = paper_style();
scriptDir = fileparts(mfilename('fullpath'));
figureDir = fileparts(scriptDir);
dataPath = fullfile(figureDir, 'data', 'seed32_cda_market_rollouts.csv');
outputDir = fullfile(figureDir, 'output');
if ~exist(outputDir, 'dir'), mkdir(outputDir); end

t = readtable(dataPath, 'TextType', 'string');
methodKeys = ["MAPPO", "Fixed-penalty MAPPO", "SGR-MACPO"];
colors = s.colors([1, 2, 4], :);
styles = {'-', '--', '-.'};

fig = figure('Color', 'w', 'Units', 'centimeters', ...
    'Position', [2, 2, s.singleColumnCm, 13.0]);
layout = tiledlayout(fig, 4, 1, 'TileSpacing', 'compact', 'Padding', 'compact');

ax1 = nexttile(layout);
hold(ax1, 'on');
h = gobjects(3, 1);
for k = 1:3
    rows = t.method == methodKeys(k);
    h(k) = stairs(ax1, t.hour(rows), t.electricity_cda_mwh(rows), ...
        'Color', colors(k, :), 'LineStyle', styles{k}, 'LineWidth', s.lineWidth);
end
ylabel(ax1, {'Internal electricity';'CDA trade (MWh)'}, ...
    'FontSize', s.labelFontSize);
xlim(ax1, [1 24]); xticks(ax1, 1:2:24); xticklabels(ax1, []);
grid(ax1, 'on'); ax1.GridAlpha = 0.13; ax1.MinorGridAlpha = 0.06;
text(ax1, 0.01, 0.92, '(a)', 'Units', 'normalized', 'FontWeight', 'bold');

ax2 = nexttile(layout);
hold(ax2, 'on');
for k = 1:3
    rows = t.method == methodKeys(k);
    residualExternal = t.electricity_external_buy_mwh(rows) ...
        - t.electricity_external_sell_mwh(rows);
    stairs(ax2, t.hour(rows), residualExternal, ...
        'Color', colors(k, :), 'LineStyle', styles{k}, 'LineWidth', s.lineWidth);
end
yline(ax2, 0, ':', 'Color', [0.35 0.35 0.35], 'LineWidth', 0.8);
ylabel(ax2, {'Residual external';'balance (MWh)'}, ...
    'FontSize', s.labelFontSize);
xlim(ax2, [1 24]); xticks(ax2, 1:2:24); xticklabels(ax2, []);
grid(ax2, 'on'); ax2.GridAlpha = 0.13;
text(ax2, 0.01, 0.92, '(b)', 'Units', 'normalized', 'FontWeight', 'bold');

ax3 = nexttile(layout);
hold(ax3, 'on');
for k = 1:3
    rows = t.method == methodKeys(k);
    plot(ax3, t.hour(rows), t.minimum_voltage_pu(rows), ...
        'Color', colors(k, :), 'LineStyle', styles{k}, 'LineWidth', s.lineWidth);
end
yline(ax3, 0.95, '--', '0.95 p.u. limit', 'Color', [0.28 0.28 0.28], ...
    'LineWidth', 0.9, 'LabelHorizontalAlignment', 'left');
mappo = t.method == methodKeys(1);
viol = mappo & t.minimum_voltage_pu < 0.95;
plot(ax3, t.hour(viol), t.minimum_voltage_pu(viol), 'o', ...
    'Color', colors(1, :), 'MarkerFaceColor', 'w', ...
    'MarkerSize', s.markerSize, 'LineWidth', 0.9);
ylabel(ax3, {'Minimum bus';'voltage (p.u.)'}, 'FontSize', s.labelFontSize);
xlim(ax3, [1 24]); ylim(ax3, [0.925 1.005]);
xticks(ax3, 1:2:24); xticklabels(ax3, []);
grid(ax3, 'on'); ax3.GridAlpha = 0.13;
text(ax3, 0.01, 0.92, '(c)', 'Units', 'normalized', 'FontWeight', 'bold');

ax4 = nexttile(layout);
hold(ax4, 'on');
for k = 1:3
    rows = t.method == methodKeys(k);
    stairs(ax4, t.hour(rows), t.raw_voltage_cost(rows), ...
        'Color', colors(k, :), 'LineStyle', styles{k}, 'LineWidth', s.lineWidth);
end
ylabel(ax4, {'Hourly raw';'voltage cost'}, 'FontSize', s.labelFontSize);
xlabel(ax4, 'Hour', 'FontSize', s.labelFontSize);
xlim(ax4, [1 24]); ylim(ax4, [0 0.46]); xticks(ax4, 2:2:24);
grid(ax4, 'on'); ax4.GridAlpha = 0.13;
text(ax4, 0.01, 0.90, '(d)', 'Units', 'normalized', 'FontWeight', 'bold');

lgd = legend(ax1, h, {'MAPPO', 'Fixed-penalty MAPPO', 'SGR-MACPO'}, ...
    'Orientation', 'horizontal', 'Location', 'northoutside', ...
    'NumColumns', 2, 'Box', 'off');
lgd.Layout.Tile = 'north';

linkaxes([ax1, ax2, ax3, ax4], 'x');
exportgraphics(fig, fullfile(outputDir, 'fig_voltage_case_seed32.pdf'), ...
    'ContentType', 'vector');
exportgraphics(fig, fullfile(outputDir, 'fig_voltage_case_seed32.png'), ...
    'Resolution', 600);
close(fig);
end
