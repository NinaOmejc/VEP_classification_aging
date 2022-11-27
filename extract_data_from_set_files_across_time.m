clear all
close all
eeglab nogui; % start EEGLAB under Matlab 

data_version = 'v3';
%filePathIn = ['N:\GIBKOP\EEG\EEGLAB\2_intermediate_analyis_data\pre\']; 
%filePathOut = ['N:\MPŠ\DataMining\Seminar\data_erp_mat\'];
%filePathIn = ['D:\MPŠ\DataMining\Seminar\data_eeglab'];
filePathIn = ['N:\MPŠ\DataMining\Seminar\data_eeglab\']; 
filePathOut = ['N:\SloMoBIL\classification_paper\data\data_for_classification\matlab\'];
cd(filePathIn);
rawDataFiles = dir('*ar.set');

elec_clusts = [30:32; 24:26; 15:17; 6:8];
elec_clusts_names = {'occipital'; 'parietal'; 'central'; 'frontal'};
n_trials = 147; % 23 rare + 124 freq
n_trials_threshold = 100;

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
    EEG = pop_loadset('filename', loadName,'filepath', filePathIn);
    disp(['Loaded subject data:', num2str(isub)])

    % remove double events
    ie = 1;
    while ie < size(EEG.event, 2) 
       if ~(strcmp(EEG.event(ie).codelabel, 'white') | strcmp(EEG.event(ie).codelabel, 'einstein'))
          EEG.event(ie) = [];
       end
       if sum([EEG.event.bepoch] == EEG.event(ie).bepoch) == 2
          epochIdx = find([EEG.event.bepoch] == EEG.event(ie).bepoch);
          EEG.event(epochIdx(2)) = [];
       end
       ie = ie + 1;
    end
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
        EEG_ichan_data = squeeze(mean(EEG.data(elec_clusts(ichanclust, :),:,:)));
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
    
    % save subject ID
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

time_info = EEG.times;
chan_info = [{EEG.chanlocs.labels}]';

save([filePathOut 'data_' data_version '.mat'], "X", "IDs", "y", "yi", "y_cats", "X_features", "time_info", "chan_info", "elec_clusts", "elec_clusts_names")


% Remove subject 25
%
% idx25 = [];
% for id = 1:length(IDs)
%     if IDs{id} == '25'
%         idx25 = [idx25; id];
%     end
% end
% 
% idx = ismember(1:length(IDs),idx25);
% IDs = IDs(~idx, :);
% X = X(~idx, :, :);
% y = y(~idx);
% yi = yi(~idx);    
% 
% filePathOut = 'D:\Experiments\erp_classification_study\data\data_for_classification\matlab\data_v2.mat'
% save(filePathOut, "X", "IDs", "y", "yi", "y_cats", "X_features", "time_info", "chan_info")

