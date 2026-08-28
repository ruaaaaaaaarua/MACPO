function run_all_figures()
%RUN_ALL_FIGURES Generate all figures that do not require new server rollout.

scriptDir = fileparts(mfilename('fullpath'));
addpath(scriptDir);
plot_voltage_case_seed32();
plot_h2_service();
plot_recurrent_ablation_seed30();
if isfile(fullfile(fileparts(scriptDir), 'data', 'seed32_delivery_counterfactual.csv'))
    plot_h2_delay_case_seed32();
    plot_h2_delay_case_seed32_singlecol();
end
split_authoritative_convergence();
end
