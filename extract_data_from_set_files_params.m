clear all
close all
eeglab nogui; % start EEGLAB under Matlab 

data_version = 'v1';
filePathIn = ['N:\SloMoBIL\classification_paper\data\raw_data\']; 
filePathOut = ['N:\SloMoBIL\classification_paper\data\data_for_classification\matlab\'];
cd(filePathIn);
rawDataFiles = dir('*ar.set');

elec_clusts = [30:32; 15:17];
elec_clusts_names = {'occ'; 'cen'};
comps = {'P1'; 'N1'; 'P2'; 'P3'};
comp_locs = {'occ'; 'occ'; 'occ'; 'cen'};
comp_times = {50:100; 100:200; 200:350; 250:500};
comp_timeidx = {65:78; 78:103; 103:141; 116:180};
params = {'PA', 'MA', 'PL', 'FL'};
n_trials = 147; % 23 rare + 124 freq
n_trials_threshold = 100;

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

time_info = EEG.times;
chan_info = [{EEG.chanlocs.labels}]';

%save([filePathOut 'data_' data_version '_notime.mat'], "X", "IDs", "y", "yi", "y_cats", "X_features", "time_info", "chan_info", "elec_clusts", "elec_clusts_names")
