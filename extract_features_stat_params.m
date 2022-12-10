% The script extracts the statistical ERP features from the participant's
% preprocessed set files. 
% 8.12.2022, Nina Omejc

clear all
close all
eeglab nogui; % start EEGLAB under Matlab 

data_version = 'v0';
path_main = pwd;
path_dataIn = [path_main + '\data_eeglab\']; % path to set files
path_dataOut = [path_main + '\data_for_classification\'];
cd(filePathIn);
rawDataFiles = dir('*ar.set');

elec_clusts_idx = [30:32; 15:17];
elec_clusts_names = {'occ'; 'cen'};
comps = {'P1'; 'N1'; 'P2'; 'P3'};
comp_locs = {'occ'; 'occ'; 'occ'; 'cen'};
comp_times = {50:150; 100:200; 200:325; 250:500}; % time windows were chosen based on zero-crossings in the group averaged ERP 
comp_timeidx = {65:90; 78:103; 103:135; 116:180};
params = {'PA', 'MA', 'PL', 'FL'};
n_trials = 147; % 23 rare + 124 freq
n_trials_threshold = 100; % if participant has less good trials then this threshold, remove complete dataset of the participant
noise_threshold = 100; % absolute threshold above which the trials are removed

y_age = {};
y_stim = {};
X_P1 = [];
X_N1 = [];
X_P2 = [];
X_P3 = [];
y = [];
X = [];
IDs = [];
isub = 0;

for idataFile = 1:length(rawDataFiles)
    isub = isub + 1;
    disp(['isub ', num2str(isub), ', idatafile ', num2str(idataFile)])
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

    % remove artifacts
    bad_trials = zeros(1, EEG.trials);
    for itrial = 1:EEG.trials
        data_to_check = EEG.data(sort(reshape(elec_clusts, [], 1)), :, itrial);
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
    
    % extract average amplitudes
    occ_amp = squeeze(mean(EEG.data(elec_clusts(1, :),:,:)));
    cen_amp = squeeze(mean(EEG.data(elec_clusts(2, :),:,:)));
    disp(['max amp occ: ', num2str(max(max(occ_amp))), '  ', num2str(max(max(cen_amp)))])
    pause(0.5)
    iX_P1 = zeros(length(comps), n_trials)';
    iX_N1 = zeros(length(comps), n_trials)';
    iX_P2 = zeros(length(comps), n_trials)';
    iX_P3 = zeros(length(comps), n_trials)';
    iX = {iX_P1, iX_N1, iX_P2, iX_P3};

    for i = 1:length(comps)
        icomp = comps{i};
        icomp_loc = comp_locs{i};
        icomp_times = comp_times{i};
        icomp_timeidx = comp_timeidx{i};
        if strcmp(icomp_loc, 'occ')
            erps = occ_amp;
        else
            erps = cen_amp;
        end
        
        % find max or min peak inside measurement window        
        if contains(icomp, 'P') 
            [iPA, iPLidx] = max(erps(icomp_timeidx, :), [], 1);
        else % if negative polarity 
            [iPA, iPLidx] = min(erps(icomp_timeidx, :), [], 1);
        end
        iPL = EEG.times(icomp_timeidx(iPLidx));
        
        % find mean amplitude
        iMA = mean(erps(icomp_timeidx, :));

        % find fractional peak latency
        iFL = zeros(1, EEG.trials);
        for itrial = 1:EEG.trials

            if contains(icomp, 'P') 
                iFL_itrial = erps(1:icomp_timeidx(iPLidx(itrial)), itrial) < iPA(itrial)/2;
            else
                iFL_itrial = erps(1:icomp_timeidx(iPLidx(itrial)), itrial) > iPA(itrial)/2;
            end

            firstOverIdx = find(iFL_itrial, 1, 'last');

            if ~isempty(firstOverIdx)
                iFL(itrial) = EEG.times(firstOverIdx)';
            else
                iFL(itrial) = nan;
            end
        end
        
        iX{i} = [iPA; iMA; iPL; iFL]';

    end

    % save subject ID
    subID =  repmat({isub}, EEG.trials, 1);
    subID = [string(subID), cat(2, y_age, y_stim)];
    IDs = [IDs; subID];

    % merge main labels and data
    y = cat(1, y, cat(2, y_age, y_stim));
    X = cat(1, X, [iX{1}, iX{2}, iX{3}, iX{4}]);

end


y = strrep(y, 'white', 'freq');
y = strrep(y, 'einstein', 'rare');
y = categorical(join(y, '-'));
y_cats = categories(y);
y = cellstr(y);
yi = grp2idx(y);
IDs = cellstr(IDs);

X_features = {'X_P1_PA', 'X_P1_MA', 'X_P1_PL', 'X_P1_FL', ...
              'X_N1_PA', 'X_N1_MA', 'X_N1_PL', 'X_N1_FL', ...
              'X_P2_PA', 'X_P2_MA', 'X_P2_PL', 'X_P2_FL', ...
              'X_P3_PA', 'X_P3_MA', 'X_P3_PL', 'X_P3_FL'};

time_info_orig = EEG.times;
time_info_reduced = EEG.times;
chan_info = [{EEG.chanlocs.labels}]';

save([filePathOut 'data_' data_version '_notime.mat'], "X", "IDs", "y", "yi", "y_cats", "X_features", "time_info_orig", "time_info_reduced", "chan_info", "elec_clusts_idx", "elec_clusts_names")

% choose specific features only:
data_version = 'v00';
chosen_features = [1, 3, 5, 7, 9, 11, 13, 15];
X_features = X_features(chosen_features);
X = X(:, :, chosen_features);
save([path_dataOut 'data_' data_version '_notime.mat'], "X", "IDs", "y", "yi", "y_cats", "X_features", "time_info_orig", "time_info_reduced", "chan_info", "elec_clusts_idx", "elec_clusts_names")



