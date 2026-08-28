function plot_recurrent_ablation_seed30()
%PLOT_RECURRENT_ABLATION_SEED30 Same-checkpoint recurrent-information ablation.

s = paper_style();
scriptDir = fileparts(mfilename('fullpath'));
figureDir = fileparts(scriptDir);
dataPath = fullfile(figureDir, 'data', 'recurrent_ablation_seed30.csv');
outputDir = fullfile(figureDir, 'output');
if ~exist(outputDir, 'dir'), mkdir(outputDir); end

t = readtable(dataPath, 'TextType', 'string');
fig = figure('Color', 'w', 'Units', 'centimeters', ...
    'Position', [2, 2, s.singleColumnCm, 5.8]);
ax = axes(fig); hold(ax, 'on');

barColors = [0.180, 0.560, 0.300; 0.430, 0.430, 0.430; 0.180, 0.180, 0.180];
bars = barh(ax, 1:height(t), t.economic_cost_m_cny, 0.58, ...
    'FaceColor', 'flat', 'EdgeColor', 'none');
bars.CData = barColors;

for k = 1:height(t)
    text(ax, t.economic_cost_m_cny(k) + 0.12, k, ...
        sprintf('%.2f', t.economic_cost_m_cny(k)), ...
        'HorizontalAlignment', 'left', 'VerticalAlignment', 'middle', ...
        'FontName', s.fontName, 'FontSize', s.fontSize);
end

set(ax, 'YTick', 1:height(t), 'YTickLabel', ...
    {'Full SGR--MACPO', 'GRU hidden state removed', 'Previous action removed'}, ...
    'YDir', 'reverse');
xlabel(ax, 'Economic cost (M CNY)', 'FontSize', s.labelFontSize);
xlim(ax, [0, 8.8]); ylim(ax, [0.4, 3.6]);
grid(ax, 'on'); ax.GridAlpha = 0.13; ax.YGrid = 'off';

exportgraphics(fig, fullfile(outputDir, 'fig_recurrent_ablation_seed30.pdf'), ...
    'ContentType', 'vector');
exportgraphics(fig, fullfile(outputDir, 'fig_recurrent_ablation_seed30.png'), ...
    'Resolution', 600);
close(fig);
end
