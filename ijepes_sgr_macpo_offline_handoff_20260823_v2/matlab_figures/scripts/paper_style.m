function s = paper_style()
%PAPER_STYLE Shared IJEPES-oriented MATLAB figure style.

s.fontName = 'Times New Roman';
s.fontSize = 8;
s.labelFontSize = 8.5;
s.lineWidth = 1.35;
s.markerSize = 4.5;
s.singleColumnCm = 8.6;
s.doubleColumnCm = 17.8;
s.colors = [ ...
    0.0000, 0.4470, 0.7410; ... % MAPPO blue
    0.8500, 0.3250, 0.0980; ... % fixed-penalty orange
    0.4940, 0.1840, 0.5560; ... % Lagrangian purple
    0.1800, 0.5600, 0.3000];    % SGR-MACPO green
s.lineStyles = {'-', '--', ':', '-.'};
s.methodLabels = {'GRU-MAPPO', 'Fixed-penalty GRU-MAPPO', 'SGR-MACPO'};

set(groot, 'defaultAxesFontName', s.fontName);
set(groot, 'defaultTextFontName', s.fontName);
set(groot, 'defaultAxesFontSize', s.fontSize);
set(groot, 'defaultAxesLineWidth', 0.65);
set(groot, 'defaultAxesTickDir', 'out');
set(groot, 'defaultAxesBox', 'on');
set(groot, 'defaultLegendFontName', s.fontName);
set(groot, 'defaultLegendFontSize', s.fontSize);
end
