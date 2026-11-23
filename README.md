Evaluating SHAP-based fault localization against documented root causes and simple baselines in the Tennessee Eastman Process

https://doi.org/10.5281/zenodo.24409648

Code to reproduce the results of the paper "Evaluating SHAP-Based Fault Localization Against Documented Root Causes and Simple Baselines in the Tennessee Eastman Process" (Imran, Zaidi, ).

The study scores SHAP channel rankings of five fault classifiers against the documented root-cause channels of 12 Tennessee Eastman Process (TEP) faults, against an exact chance model and against rankings computed without a classifier, and repeats the evaluation on a second TEP simulator and on a simulated jacketed reactor.

Layout
tep/        analysis of the two TEP datasets (data preparation, classifiers, rankings, statistics)
reactor/    simulation and evaluation of the jacketed reactor (second process)
figures/    scripts that draw the paper's figures from the stored results
data/       instructions for obtaining the data

Results are written to a dataset root: a folder with data/tep_clean.npz and out/. By default the scripts use runs/revised; set TEP_ROOT for scripts without a root argument, or pass the root as the first argument where the usage line asks for it. Run the TEP scripts from tep/ and the reactor scripts from reactor/.

Installation
bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

The results were produced with Python 3.13 on CPU. Seeds are fixed in every script.

Data

See data/README.md. In short:

bash
cd tep
python prepare_data.py /path/to/matlab_data_1year.csv ../runs/revised
python prepare_data.py /path/to/python_data_1year.csv ../runs/original
Reproducing the results

Seeds 0-9 are the ten episode-level splits. Commands are shown for the revised dataset; for the original dataset use TEP_ROOT=../runs/original, the root ../runs/original, and --no-xgb-perm for run_mlp_xgboost.py.

bash
cd tep
S="0 1 2 3 4 5 6 7 8 9"; R=../runs/revised
# classifiers, SHAP rankings and main baselines
python run_mlp_xgboost.py $S
python run_lagged_mlp.py $S
python run_sequence_models.py cnn $S; python run_sequence_models.py gru $S
python baseline_pca_rbc.py $R/data $R/out; python baseline_pca_rbc_parallel.py $R/data $R/out
for p in classfree mlprand ig_cnn ig_gru; do python baselines_extra.py $R $p $S; done
python conditional_shap.py $R $S
for a in cnn gru; do python sequence_deviation.py $R $a $S; done
# ablation, fairness, robustness, sensitivity
python ablation.py $R mlp $S; python ablation.py $R xgb $S
python fairness_test_set.py $R $S; python fairness_matched_samples.py $R $S
python tuning_mlp.py $R $S; for p in xgb cnn gru; do python tuning_xgb_sequence.py $R $p $S; done
python robustness_test.py ../runs/revised ../runs/original $S; python robustness_train.py $R $S
python sensitivity_pipeline.py $R $S; python sensitivity_baselines.py $R $S; python sensitivity_onset.py $R $S
python deviation_all_samples.py $R; python monte_carlo_mlp.py $R 0 1 2
for a in cnn gru; do python monte_carlo_sequence.py $R $a 0 1 2; done
python bootstrap_stability.py; python residual_variability.py; python confusion_matrix_model.py
OMP_NUM_THREADS=1 python computational_cost.py $R
# aggregation and statistics
python aggregate_main.py; python aggregate_all.py $R
python statistics_main.py $R "Mean shift"          # original dataset: "Mean/var", and "Mean shift" stats2_meanshift.json
python statistics_extra_rankings.py $R; python statistics_top5_subset.py $R; python power_top5.py $R
python ablation_stats.py $R; python hit_agreement_kappa.py $R; python monte_carlo_sequence_summary.py
python robustness_aggregate.py $R; python sensitivity_evaluation.py $R; python sensitivity_aggregate.py $R
python sensitivity_pca_specificity.py $R; python sensitivity_onset_aggregate.py $R
python metric_robustness.py $R

# reactor (second process)
cd ../reactor
python simulate.py
python run_reactor.py 0 1 2 3 4 5 6 7 8 9
python evaluate_reactor.py; python evaluate_reactor_mv_convention.py; python sensor_fault_magnitude.py

# figures
cd ../figures
python make_figures.py; python figure_beeswarm_fault14.py

The sequence models, Integrated Gradients and the permutation explainer on XGBoost take the longest (Supplementary Table S9 gives run times on one CPU thread).

Scripts and the paper content they produce
Script	Paper content
tep/prepare_data.py, data_split.py, fault_metadata.py	data preparation, episode split, documented channels (Section 3, Table 2)
tep/run_mlp_xgboost.py, run_lagged_mlp.py, run_sequence_models.py	classifiers and SHAP rankings (Table 3)
tep/baseline_pca_rbc.py, baselines_extra.py	PCA contribution, KS, Granger, untrained MLP, Integrated Gradients (Tables 3-5, Table S8)
tep/baseline_pca_rbc_parallel.py, conditional_shap.py, sequence_deviation.py	parallel-analysis PCA, conditional SHAP, sequence-model deviation (Sections 4.3-4.4, Table S8)
tep/aggregate_main.py, aggregate_all.py	hit rates, MRR, chance tests, paired comparisons (Tables 3-4, Figs. 2, S2)
tep/statistics_main.py	bootstrap intervals, permutation and sign-flip tests (Table 4, Section 4.2, Table S2)
tep/statistics_extra_rankings.py, statistics_top5_subset.py, power_top5.py, fixed_list_bound.py	comparisons of the added rankings; best top-five baseline and direct/strong subset; power of the paired test (Sections 4.3, 4.5, Limitations)
tep/ablation.py, ablation_stats.py, deviation_all_samples.py, hit_agreement_kappa.py	ablation and deviation analyses (Table 5, Section 4.4)
tep/fairness_test_set.py, fairness_matched_samples.py, tuning_mlp.py, tuning_xgb_sequence.py	fair comparison and tuning (Section 3.5, Section S5)
tep/robustness_test.py, robustness_train.py, robustness_aggregate.py	robustness (Table 6, Section S7)
tep/sensitivity_*.py	sensitivity analysis and cut-off dependence (Table 6, Fig. 3, Section S8)
tep/monte_carlo_mlp.py, monte_carlo_sequence.py, monte_carlo_sequence_summary.py	Monte Carlo error of the explainers (Limitations, Sections S4, S12)
tep/metric_robustness.py, computational_cost.py	metric robustness and run times (Section S9, Table S9, Discussion)
tep/bootstrap_stability.py, residual_variability.py, confusion_matrix_model.py	Fig. S3, Section S10, Fig. S1
reactor/simulate.py, run_reactor.py, evaluate_reactor.py	second process (Section 4.7, Section S12, Table S10)
reactor/evaluate_reactor_mv_convention.py, sensor_fault_magnitude.py	sensor-fault checks on the reactor (Section 4.7, Section S12)
figures/make_figures.py, figure_beeswarm_fault14.py	Figs. 1-3 and S1-S5
Data source

Oliveira e Souza, A. C., de Souza Jr., M. B., & da Silva, F. V. (2024). Enhancing fault detection and diagnosis systems for a chemical process: a study on convolutional neural networks and transfer learning. Evolving Systems, 15(2), 611-633. https://doi.org/10.1007/s12530-023-09523-y

Citation

If you use this code, please cite the paper (reference to be added on publication) and the archived code:

Imran, M., He, Z., Zaidi, F. S., Anwar, S. M., & Hu, X. (2026). Code for "Evaluating SHAP-Based Fault Localization Against Documented Root Causes and Simple Baselines in the Tennessee Eastman Process" Zenodo. https://doi.org/10.5281/zenodo.24409648

License

MIT License; see LICENSE.
