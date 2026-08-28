function split_authoritative_convergence()
%SPLIT_AUTHORITATIVE_CONVERGENCE Preserve and separate the paper curves.
%
% The underlying curve arrays are unavailable.  This script deliberately crops
% the current 600-dpi manuscript image instead of substituting server logs.

scriptDir = fileparts(mfilename('fullpath'));
figureDir = fileparts(scriptDir);
paperDir = fileparts(figureDir);
outputDir = fullfile(figureDir, 'output');
if ~exist(outputDir, 'dir'), mkdir(outputDir); end

sourcePath = fullfile(paperDir, 'fig_convergence.png');
img = imread(sourcePath);

% Source is 2023 x 2141 px.  The top crop keeps the shared legend; the lower
% crop is paired with a copy of that legend so each standalone panel is usable.
reward = img(1:1065, :, :);
legendBand = img(850:1135, :, :);
costPanel = img(1136:end, :, :);
cost = [legendBand; costPanel];

write_crop_with_optional_xlabel(reward, ...
    fullfile(outputDir, 'fig_convergence_reward'), true);
write_crop_with_optional_xlabel(cost, ...
    fullfile(outputDir, 'fig_convergence_voltage_cost'), false);
end

function write_crop_with_optional_xlabel(img, outputStem, addUpdateLabel)
widthCm = 8.6;
heightCm = widthCm * size(img, 1) / size(img, 2);
fig = figure('Color', 'w', 'Units', 'centimeters', ...
    'Position', [2, 2, widthCm, heightCm]);
ax = axes(fig, 'Position', [0 0 1 1]);
image(ax, img);
axis(ax, 'image');
axis(ax, 'off');
if addUpdateLabel
    annotation(fig, 'textbox', [0.42 0.125 0.16 0.045], ...
        'String', 'Update', 'FontName', 'Times New Roman', 'FontSize', 8, ...
        'HorizontalAlignment', 'center', 'VerticalAlignment', 'middle', ...
        'EdgeColor', 'none', 'Margin', 0);
end
exportgraphics(fig, [outputStem '.png'], 'Resolution', 600);
exportgraphics(fig, [outputStem '.pdf'], 'ContentType', 'image', 'Resolution', 600);
close(fig);
end
