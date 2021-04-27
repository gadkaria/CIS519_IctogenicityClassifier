%% CIS 519 Final Project
%Contributors:
%Georgios Mentzelopoulos
%Ameya Gadkari
%Alejandro Resendiz

%Requirements to run code:
% 1. A valid account for IEEG.org (open source data repository for epilepsy
%    research
% 2. IEEG matlab toolbox (available to download through IEEG.org)

%% Download the HFOs and artifacts from the opensource IEEG Portal
% Replace the 'gment_ieeglogin.bin' file with your .bin file that contains
% your username and password on ieeg.org
session = IEEGSession('I521_A0004_D001', 'gment', 'gment_ieeglogin.bin');

[trainEvents, train_timesUSec, train_channels] = getAnnotations(session.data,'Training windows');
count_artifact = 0;
count_HFO = 0;
for i = 1:length(train_channels)
    if trainEvents(1,i).description == '2'
        count_HFO = count_HFO + 1;
    elseif trainEvents(1,i).description == '1'
        count_artifact = count_artifact + 1;
    else
        assert(false)
    end
end

%% Find the first occurence of a valid HFO and artifact and plot them as a sanity check
%Find index of first HFO
for i = 1:length(train_channels)
    if trainEvents(1,i).description == '2'
        train_HFO_index_1 = i;
        break
    end
end

%Find index of first artifact
for i = 1:length(train_channels)
    if trainEvents(1,i).description == '1'
        train_artifact_index_1 = i;
        break
    end
end

Sampling_rate = session.data.sampleRate; 

%Get data of first HFO
my_channel = 2;
timeStartUsec = trainEvents(1, train_HFO_index_1).start;
timeEndUsec =  trainEvents(1, train_HFO_index_1).stop;
blocklengthUsec = timeEndUsec-timeStartUsec;
first_HFO = session.data.getvalues(0, blocklengthUsec,my_channel);

first_HFO_time = linspace(timeStartUsec,timeEndUsec,length(first_HFO));


%Get data of first Artifact
my_channel = 2;
timeStartUsec = trainEvents(1, train_artifact_index_1).start;
timeEndUsec =  trainEvents(1, train_artifact_index_1).stop;
blocklengthUsec = timeEndUsec-timeStartUsec;
first_artifact = session.data.getvalues(timeStartUsec, blocklengthUsec,my_channel);

%Plot the data
first_artifact_time = linspace(timeStartUsec,timeEndUsec,length(first_artifact));

figure()
subplot(1,2,1)
plot(first_HFO_time*10^-3,first_HFO);
legend('First HFO');
xlabel('Time (msec)');
set(gca,'YTick', []);
subplot(1,2,2)
plot(first_artifact_time*10^-3, first_artifact);
legend('First Artifact');
xlabel('Time (msec)');
set(gca,'YTick', []);

%% Filter the data to isolate the frequencies of interest (80-520 Hz)
    d = designfilt('bandpassfir','FilterOrder',100, ...
        'CutoffFrequency1',80,'CutoffFrequency2',520, ...
        'SampleRate',Sampling_rate);
%     fvtool(d, 'Fs', Sampling_rate)

 first_HFO_filtered = filtfilt(d,first_HFO);
 first_artifact_filtered = filtfilt(d,first_artifact);
 % Plot the HFO and artifact after filtering
figure()
subplot(1,2,1)
hold on;
plot(first_HFO_time*10^-3,first_HFO);
plot(first_HFO_time*10^-3,first_HFO_filtered);
title('First HFO');
legend('No-Filter', 'Filter');
xlabel('Time (msec)');
set(gca,'YTick', []);
hold off;
subplot(1,2,2)
hold on;
plot(first_artifact_time*10^-3, first_artifact);
plot(first_artifact_time*10^-3, first_artifact_filtered);
title('First Artifact');
legend('No-Filter', 'Filter');
xlabel('Time (msec)');
set(gca,'YTick', []);
hold off;

%% Define the function that will be used for feature extraction

LLFn = @(x) sum(abs(diff(x))); % Define Line Length

A = @(x) sum(abs(x)); %Define Area

E = @(x) sum(x.^2); % Define Energy

ZX = @(x) sum(diff(sign(x-mean(x))) ~=0); % Define Zero Crossings


[testEvents, test_timesUSec, test_channels] = getAnnotations(session.data,'Testing windows');

trainFeats = -1*ones(length(trainEvents),2);

testFeats = -1*ones(length(testEvents),2);

train_HFO_index = [];
train_aftifact_index = [];
test_HFO_index = [];
test_artifact_index = [];

%Find HFO index in TrainEvents
for i = 1:length(train_channels)
    if trainEvents(1,i).description == '2'
        train_HFO_index(end+1) = i;
    end
end

%Find Artifact Index in TrainEvents
for i = 1:length(train_channels)
    if trainEvents(1,i).description == '1'
        train_aftifact_index(end+1) = i;
    end
end

%Find HFO index in testEvents
for i = 1:length(test_channels)
    if testEvents(1,i).description == '2'
        test_HFO_index(end+1) = i;
    end
end

%Find Artifact Index in TrainEvents
for i = 1:length(test_channels)
    if testEvents(1,i).description == '1'
        test_artifact_index(end+1) = i;
    end
end
%% Extract Features 

%Extract Features from Training Set
for i = 1:length(trainFeats(:,1))
    my_channel = 2;
    timeStartUsec = trainEvents(1, i).start;
    timeEndUsec =  trainEvents(1, i).stop;
    blocklengthUsec = timeEndUsec-timeStartUsec;
    HFO = session.data.getvalues(timeStartUsec, blocklengthUsec,my_channel);
    HFO_LL = LLFn(HFO);
    HFO_A = A(HFO);
    HFO_E = E(HFO);
    HFO_ZX = ZX(HFO);
    trainFeats(i,1) = HFO_LL;
    trainFeats(i,2) = HFO_A;
    trainFeats(i,3) = HFO_E;
    trainFeats(i,4) = HFO_ZX;
end

trainFeats(end,:) = trainFeats(end-1,:);

%Extract Features from Testing Set
for i = 1:length(testFeats(:,1))
    my_channel = 1;
    timeStartUsec = testEvents(1, i).start;
    timeEndUsec =  testEvents(1, i).stop;
    blocklengthUsec = timeEndUsec-timeStartUsec;
    artifact = session.data.getvalues(timeStartUsec, blocklengthUsec,my_channel);
    artifact_LL = LLFn(artifact);
    artifact_A = A(artifact);
    artifact_E = E(artifact);
    artifact_ZX = ZX(artifact);
    testFeats(i,1) = artifact_LL;
    testFeats(i,2) = artifact_A;
    testFeats(i,3) = artifact_E;
    testFeats(i,4) = artifact_ZX;
end
%% Plot the artifact and HFOs in Feature space to see whether they are  
%  separable
figure()
hold on;
for i = 1:length(trainFeats(:,1))
    if any(train_HFO_index(:) == i)
        scatter(trainFeats(i,1),trainFeats(i,2), 'b');
    elseif any(train_aftifact_index(:) == i)
        scatter(trainFeats(i,1),trainFeats(i,2), 'r');
    end
end
xlabel('Line Length');
ylabel('Area');
legend('Artifact', 'HFO')
title('Scatter Plot of the Features of the Training Set')

%% Combine the training and testing features into a single feature matrix
% This matrix can be used to train and validate different classifiers in
% python 
feature_matrix = [trainFeats; testFeats];

%save matrix into excel format to be uploaded in google collab
xlswrite('Feature_matrix.xlsx', feature_matrix)


