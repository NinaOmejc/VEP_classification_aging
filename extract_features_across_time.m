% The script extracts the temporal features from the participant's
% preprocessed set files. 
% 8.12.2022, Nina Omejc

clear all
close all
eeglab nogui; % start EEGLAB under Matlab 

data_version = 'v0';
path_main = pwd;
path_dataIn = [path_main + '\data_eeglab\']; % path to set files
path_dataOut = [path_main + '\data_for_classification\'];
cd(path_dataIn);
rawDataFiles = dir('*ar.set');

elec_clusts_idx = [30:32; 24:26; 15:17; 6:8];
elec_clusts_names = {'occipital'; 'parietal'; 'central'; 'frontal'};
n_trials = 147; % 23 rare + 124 freq
n_trials_threshold = 100; % if participant has less good trials then this threshold, remove complete dataset of the participant
noise_threshold = 100; % absolute threshold above which the trials are removed

y_age = {};
y_stim = {};
y = [];
X = [];
IDs = [];
isub = 0;

for idataFile = 1:length(rawDataFiles)
    isub = isub + 1;
    loadName = rawDataFiles(idataFile).name;
    dataName = loadName(1:end-4);
    EEG = pop_loadset('filename', loadName,'filepath', path_dataIn);
    disp(['Loaded subject data:', num2str(isub)])

    % remove double events
    ie = 1;
    while ie < size(EEG.event, 2) 
       if ~(strcmp(EEG.event(ie).codelabel, 'white') || strcmp(EEG.event(ie).codelabel, 'einstein'))
          EEG.event(ie) = [];
       end
       if sum([EEG.event.bepoch] == EEG.event(ie).bepoch) == 2
          epochIdx = find([EEG.event.bepoch] == EEG.event(ie).bepoch);
          EEG.event(epochIdx(2)) = [];
       end
       ie = ie + 1;
    end
    
    % remove artifacts with absolute value above the threshold
    bad_trials = zeros(1, EEG.trials);
    for itrial = 1:EEG.trials
        data_to_check = EEG.data(sort(reshape(elec_clusts_idx, [], 1)), :, itrial);
        [bad_elects, bad_tps] = find(data_to_check > noise_threshold | data_to_check < -noise_threshold);
        if any(bad_elects)
            bad_trials(itrial) = 1;
        end
    end
    disp(['Number of bad trials detected: ' num2str(sum(bad_trials))])
    EEG.data = EEG.data(:, :, ~bad_trials);
    EEG.event(find(bad_trials)) = [];
    EEG.trials = size(EEG.event, 2);

    % remove events if too many or remove subjects if it doesnt have enough events
    if size(EEG.event, 2) > n_trials
        EEG.event(148:end) = [];
        EEG.trials = size(EEG.event, 2);
        EEG.data = EEG.data(:, :, 1:n_trials);
    elseif size(EEG.event, 2) < n_trials_threshold
        disp(['skipping subject number ', num2str(isub), ' due to low number of events left.'])
        continue
    end

    % extract event labels
    y_stim = {EEG.event.codelabel}';
    if any(strcmp(y_stim, 'white') | strcmp(y_stim, 'einstein'))
        y_stim = y_stim(strcmp(y_stim, 'white') | strcmp(y_stim, 'einstein'));
    end

    % extract age labels
    if EEG.setname(1) == 'G'
        ilabel = 'older';
    else
        ilabel = 'young';
    end
    y_age = repmat({ilabel}, EEG.trials, 1);
    
    % get amplitudes
    X_amp = cell(length(elec_clusts_names), 2);

    % time freq analysis
    X_tf = cell(length(elec_clusts_names), 2);
    num_freqs = 9;
    paddingBegin = 16;
    paddingEnd = 16;

    for ichanclust = 1:length(elec_clusts_names)

        ichan_timefreq = zeros(EEG.trials, EEG.pnts, num_freqs);
        EEG_ichan_data = squeeze(mean(EEG.data(elec_clusts_idx(ichanclust, :),:,:)));
        X_amp{ichanclust, 1} = EEG_ichan_data;
        X_amp{ichanclust, 2} = elec_clusts_names(ichanclust);
        disp(['Time freq analysis for: ', elec_clusts_names{ichanclust}])
        
        for it = 1:EEG.trials
       
            [ersp, ~, ~, times, freqs] = newtimef(EEG_ichan_data(:, it), ...
                                                  EEG.pnts, ... 
                                                  [EEG.xmin  EEG.xmax]*1000, ...
                                                  EEG.srate, ... 
                                                  [0] , ... % use FFT Hann window, fixed
                                                  'timesout', EEG.times, ...
                                                  'freqs', [4 36], ...
                                                  'nfreqs', 9, ...
                                                  'padratio', 2, ...
                                                  'plotphase', 'off', ...
                                                  'plotersp', 'off', ...
                                                  'plotitc', 'off', ...
                                                  'verbose', 'on');
            
            ersp_padded = [nan(length(freqs), paddingBegin) ersp nan(length(freqs), paddingEnd)]';
            ichan_timefreq(it, :, :) = ersp_padded;

        end 
        X_tf{ichanclust, 1} = ichan_timefreq;
        X_tf{ichanclust, 2} = elec_clusts_names(ichanclust);
    end


    % merge labels and data 
    y = cat(1, y, cat(2, y_age, y_stim));

    X = cat(1, X, cat(3, X_amp{1, 1}', X_amp{2, 1}',X_amp{3, 1}',X_amp{4, 1}', ...
                         X_tf{1, 1}, X_tf{2, 1}, X_tf{3, 1}, X_tf{4, 1}));

    X_ID =  repmat({isub}, EEG.trials, 1);
    X_ID = [string(X_ID), cat(2, y_age, y_stim)];
    IDs = [IDs; X_ID];
end

y = strrep(y, 'white', 'freq');
y = strrep(y, 'einstein', 'rare');
y = categorical(join(y, '-'));
y_cats = categories(y);
y = cellstr(y);
yi = grp2idx(y);
IDs = cellstr(IDs);

X_features = {'X_ampO', 'X_ampP', 'X_ampC', 'X_ampF', ...
              'X_tfO4', 'X_tfO8', 'X_tfO12', 'X_tfO16', 'X_tfO20', 'X_tfO24', 'X_tfO28', 'X_tfO32', 'X_tfO36',...            
              'X_tfP4', 'X_tfP8', 'X_tfP12', 'X_tfP16', 'X_tfP20', 'X_tfP24', 'X_tfP28', 'X_tfP32', 'X_tfP36',...             
              'X_tfC4', 'X_tfC8', 'X_tfC12', 'X_tfC16', 'X_tfC20', 'X_tfC24', 'X_tfC28', 'X_tfC32', 'X_tfC36',...              
              'X_tfF4', 'X_tfF8', 'X_tfF12', 'X_tfF16', 'X_tfF20', 'X_tfF24', 'X_tfF28', 'X_tfF32', 'X_tfF36'};

time_info_orig = EEG.times;
time_info_reduced = times;
chan_info = [{EEG.chanlocs.labels}]';

save([path_dataOut 'data_' data_version '.mat'], "X", "IDs", "y", "yi", "y_cats", "X_features", "time_info_orig", "time_info_reduced", "chan_info", "elec_clusts_idx", "elec_clusts_names")

% choose specific features only:
data_version = 'v2';
chosen_features = [1:2, 4, 6, 13, 15];
X_features = X_features(chosen_features);
X = X(:, :, chosen_features);
save([path_dataOut 'data_' data_version '.mat'], "X", "IDs", "y", "yi", "y_cats", "X_features", "time_info_orig", "time_info_reduced", "chan_info", "elec_clusts_idx", "elec_clusts_names")

data_version = 'v6';
chosen_features = [1:4, 12, 21, 24 30];
X_features = X_features(chosen_features);
X = X(:, :, chosen_features);
save([path_dataOut 'data_' data_version '.mat'], "X", "IDs", "y", "yi", "y_cats", "X_features", "time_info_orig", "time_info_reduced", "chan_info", "elec_clusts_idx", "elec_clusts_names")

data_version = 'v7';
chosen_features = [1, 3, 5, 6, 23, 24, 25, 32];
X_features = X_features(chosen_features);
X = X(:, :, chosen_features);
save([path_dataOut 'data_' data_version '.mat'], "X", "IDs", "y", "yi", "y_cats", "X_features", "time_info_orig", "time_info_reduced", "chan_info", "elec_clusts_idx", "elec_clusts_names")


