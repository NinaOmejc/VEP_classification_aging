The scrips contain the workflo of the ERP classification task based on two types of features: across time and statistical ERP parameters.

1. The data are first extracted by two matlab scripts "extract_features_across_time" and "extract_features_stat_params".
2. Most informative subset of features were selected by calling "feature_selection" function from the "classify_main" script. 
	The feature selection results were merged and plotted by "merge_feature_selection" script.
3. Binary classification tasks were ran by "classify_main" script, where the classification_type had to be chosen between "classify_bysub", "classify_allsubsage" and "classify_allsubs",
	where the first one classified all subjects independently(target = stimulus type), the second one all subjects together but with only rare trials included (target=age) and the last
	classified again all participants together but with four-class target: stim (rare/freq) x age (young/older)
4. Results of the classifiers are merged by "merge results" script.
5. Results are analysed and mainly plotted by "analyse_results" and a helper script "analyse_results_plotting".

Shell document was used to send jobs of (a bit modified) classify_main script to the HPC (supercomputer).

Code was not designed for common usage so it can be sloppy and have some mistakes. It was put online to present the general workflow of the ERP analysis & classification protocol used in the paper:
"On the influence of aging on classification performance in visual EEG oddball paradigm using statistical and temporal features".

Written by Nina Omejc, 10.12.2022. 
For info please contact: nina.omejc@ijs.si


	
