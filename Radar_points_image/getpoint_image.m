function key = getpoint_image(file_path) %  Usage Input: mmwave .bin file Output: xxx.png
%% 全局变量 Global variable

    file_name = 'southner_data_Raw_0'; % my mmwave file name
    light_v = 3e8;  % velocity of light
    f_0 = 77e9;     % start frequency
    total_chirp_num = 80*3; 
    sample_num =256;
    tx_num = 3; 
    rx_num = 4;
    chirp_num = total_chirp_num / tx_num;
    %lane_num = 4;
    lamda = light_v / f_0;
    sample_rate = 6800e3;
    chirp_ramp_time = 83e-6;
    frequency_slope = 47.990e12;  
    B=frequency_slope *chirp_ramp_time;
    frame_num = 450;
    frame_period = 50e-3;%数据啥也不影响就是了 
    IDLE_TIME = 7e-6;
    chirp_period = chirp_ramp_time+IDLE_TIME;
    range_theorical=light_v/2/B;
    range_res = sample_rate * light_v / sample_num / frequency_slope / 2;
    % Range_real is the Reliable resolution
    range_real=max([range_theorical,range_res]);
    range_max = sample_rate * light_v / frequency_slope / 2/2;
    velocity_theorical = lamda / chirp_period / chirp_num / 2;
    velocity_max = lamda / chirp_period /4;


%% Trick 开关
% 是否进行1443天线板固有相位偏移补偿 
% is_1443_comps: Compensation for inherent phase deviation of antenna board. Your antenna doesn't need this. 
is_1443_comps = true;
% is_rm_empty
% 是否减去 empty 状态信号
% is_rm_empty = false;
% % 是否减去前一帧信号以消除杂波
% is_rm_forward = false;
% % 是否进行mmMesh静态杂波消除
% is_clutter_removal = true;
% % 是否进行加窗
% is_win = true;
% % 是否将 max_range 米之外的信号舍弃
% is_drop_far = false;
% % 是否将 min_range 米之内的信号置零 
% is_drop_close = false;
% % 是否进行CA-CFAR
% is_cfar = false;
% % 是否进行相移补偿
is_phase_comps = true;
% % 是否执行帧合并
% is_merge_set = false;
% % 是否每帧仅保留设定个数点
% is_point_res = false;
% % 是否画出三维散点图
% is_scatter_point = true;
% % 是否画出动态 doppler 热图
% is_figure_doppler = false;

%% Trick 部分数据设定
% 1443天线板固有相位偏移常数
att_phase_shift_1443 = [[0, 0.2443, 0.2560, 0.3664]; [0.7842, 1.0626, 1.0997, 1.1946]; [0.8840, 1.1286, 1.1252, 1.2505]]; % 暂定

% 数据转换方案 
% 0: 原始方案，3DFFT结果取两个峰值
% 1: wx取fft峰值来计算，wz取wx的+-5范围之内的峰值，若无峰值，wz取第一个值
% 2: wx取fft峰值来计算，wz取原始4个数据除以对应wx递增量后再取均值
plan = 0;
% 坐标显示限制
% axis_set = [-3, 3, 4, 9, -1, 1.5];
% 每帧对应一个点集
point_res_num = 128;
point_set_num = frame_num;
point_set = zeros(point_set_num, 6, point_res_num); % x,y,z coordinate，距离 range，速度 velocity，信号强度 density
point_num = zeros(1, point_set_num);
% 合并帧设定
% if is_merge_set
%     merge_num = 3; % 将连续 merge_num 个帧合并在一个点集之中
% else
%     merge_num = 1;
% end
%merge_set_num = frame_num / merge_num; % 合并帧点集数
% 每个合并帧保留的点个数

%point_res_set = zeros(merge_set_num, 6, point_res_num);
% 选择显示某个合并帧
%merge_set_index = 1;
% 3-D FFT填充个数
angle_padded_num = 32;
% 窗函数设定
% r_win = reshape(hamming(sample_num), 1, 1, 1, 1, []);
% v_win = reshape(hamming(chirp_num), 1, 1, 1, [], 1);
%r_win = reshape(blackman(sample_num), 1, 1, 1, 1, []);
%v_win = reshape(hann(chirp_num), 1, 1, 1, [], 1);
% a_win = hamming(angle_padded_num);
% 舍弃范围在 max_range 米之外的信号
% if is_drop_far
%     max_range = 8;
%     max_r_i = floor(max_range * 2 * frequency_slope * sample_num / sample_rate / light_v) + 1;
% else
%     max_r_i = sample_num;
% end
% % 将 min_range 米之内的信号置零
% if is_drop_close
%     min_range = 2;
%     min_r_i = floor(min_range * 2 * frequency_slope * sample_num / sample_rate / light_v) + 1;
% else
%     min_r_i = 0;
% end

% %% CA-CFAR 相关变量
% cfar_gd_size = [1, 1]; % guard部分大小
% cfar_tr_size = [1, 1]; % train部分大小
% cfar_pfa = 0.5; % 过滤点百分比
% %cfar2D = phased.CFARDetector2D('GuardBandSize',cfar_gd_size,'TrainingBandSize',cfar_tr_size, 'ProbabilityFalseAlarm',cfar_pfa);
% row_beg = cfar_tr_size(1) + cfar_gd_size(1) + 1;
% col_beg = cfar_tr_size(2) + cfar_gd_size(2) + 1;
% row_end = chirp_num + cfar_tr_size(1) + cfar_gd_size(1);
% col_end = max_r_i + cfar_tr_size(2) + cfar_gd_size(2);
% 
% cutidx = zeros(2, chirp_num * max_r_i);
% index = 1;
% for m = col_beg : col_end
%     for n = row_beg : row_end
%         cutidx(1, index) = n;
%         cutidx(2, index) = m;
%         index = index + 1;
%     end
% end

%% 载入数据并处理 Load mmwave .bin data
fid = fopen([[file_path,'\'], file_name, '.bin'],'r');
data = fread(fid,frame_num* total_chirp_num * sample_num *2*rx_num, 'int16');
data = reshape(data, rx_num * 2, []);
data = data([1,2,3,4],:) + sqrt(-1) * data([5,6,7,8],:);
data = reshape(data, rx_num, sample_num, tx_num, chirp_num, frame_num);
data = permute(data, [5, 3, 1, 4, 2]);
fclose(fid);
% 6843 处理
% temp_data = zeros(1, squeeze(size(data)) / 2);
% counter = 1;
% for i=1:4:fileSize-1
%     temp_data(1,counter) = data(i) + sqrt(-1)*data(i+2); 
%     temp_data(1,counter+1) = data(i+1)+sqrt(-1)*data(i+3); 
%     counter = counter + 2;
% end
% data = reshape(temp_data, sample_num, rx_num, tx_num, chirp_num, frame_num);
% data = permute(data, [5, 3, 2, 4, 1]);

% raw_data = data;


%% 减去前一帧信号以消除静态杂波 Subtract the previous frame signal to eliminate static clutter
% if is_rm_forward
%     temp_data = data(2:end, : ,: , :, :) - data(1:end-1, :, :, : ,:);
%     data(2:end, : ,: , :, :) = temp_data;
% end

%% 减去 empty 状态信号 Subtract empty status signal
% if is_rm_empty && isfile([file_path, 'empty.bin'])
%     fid = fopen([file_path, 'empty.bin'],'r');
%     empty_data = fread(fid, frame_num * total_chirp_num * sample_num * rx_num * 2, 'int16');
%     empty_data = reshape(empty_data, rx_num * 2, []);
%     empty_data = empty_data([1,2,3,4],:) + sqrt(-1) * empty_data([5,6,7,8],:);
%     empty_data = reshape(empty_data, rx_num, sample_num, tx_num, chirp_num, frame_num);
%     empty_data = permute(empty_data, [5, 3, 1, 4, 2]);
%     data = data - empty_data;
% end

%% 对信号进行天线板固有相位偏移补偿 Compensation for inherent phase deviation of antenna board.
if is_1443_comps
    att_phase_shift_1443 = reshape(att_phase_shift_1443, 1, 3, 4, 1, 1);
    data = data .* exp(-1 * sqrt(-1) * att_phase_shift_1443);
end


%% 距离fft Range-fft
% 对距离fft进行加窗
data = fft(data, [], 5);
% figure;
% plot(squeeze(abs(data(150, 1, 1, 1, :))));
% title('range fft');
%% mmMesh版静态杂波消除
mean_temp = mean(data, 4);
for i = 1 : chirp_num
    data(:, :, :, i, :) = data(:, :, :, i, :) - mean_temp;
end
%% Draw if needed
%     clutter_rm_frame = data;
%     figure;
%     plot(squeeze(abs(data(150, 1, 1, 1, :))));
%     title('range fft after clutter removal');
%% 多普勒 Doppler-fft
% 对多普勒fft进行加窗 Doppler-fft Windowing
% if 0
%     data = data .* v_win;
% end
data = fftshift(fft(data, [], 4), 4);
%% Draw if needed
%data(:,:,:,32:34,:)=0;
% figure;
% surf(squeeze(abs(data(27, 1, 1, :, :))));
% title('doppler fft');

%% 进行由于分时发射引起的相移补偿 % Compensation for phase shift caused by time-sharing emission
if is_phase_comps
    for i = 1 : chirp_num
        for j = 1 : tx_num
            data(:, j, :, i, :) = data(:, j, :, i, :) * exp(-sqrt(-1) * (j - 1) * 2 * pi / 3 * (i - chirp_num / 2 - 1) / chirp_num);
        end
    end
    figure;
    surf(squeeze(abs(data(150, 1, 1, :, :))));
    title('doppler fft after phase compensation');
end

%% 对doppler热图中的高能量点进行筛选 Screening for high-energy points in the Doppler heat map
doppler_sum = squeeze(sum(data, [2, 3]));  % 将12个虚拟通道的信号对应全部叠加 Overlay all signals corresponding to 12 virtual channels
% doppler_db = log10(abs(doppler_sum));
doppler_db = abs(doppler_sum);
rss_sorted = sort(reshape(doppler_db, frame_num, []), 2, 'descend');
rss_thr = squeeze(rss_sorted(:, point_res_num));
doppler_filter = doppler_db >= rss_thr;

%% 3D fft (angle fft)
azimuth_data = cat(2, squeeze(data(:, 1, :, :, :)), squeeze(data(:, 2, :, :, :)));
elevation_data = squeeze(data(:, 3, :, : ,:));
azimuth_data = fftshift(fft(azimuth_data, angle_padded_num, 2), 2);
elevation_data = fftshift(fft(elevation_data, 8, 2), 2);

% azimuth_data2=zeros(frame_num,64,256);
% for j=1:frame_num
% for i =2:chirp_num
%    
%     azimuth_data2(j,:,:)=reshape((azimuth_data(j,:,i,:)),[1,64,256])+azimuth_data2(j,:,:);
% end
% figure;
% imagesc(squeeze(abs(azimuth_data2( j,:, :))));
% title('angle fft of azimuth data');
% end
% imagesc(squeeze(abs(azimuth_data2( :, :))));
% title('angle fft of azimuth data');
% figure;
% plot(squeeze(abs(elevation_data(150, :, 33,166))));
% title('angle fft of elevation data');
% figure;
% plot(squeeze(abs(azimuth_data(150, :, 33,166))));
% title('angle fft of azimuth data');
%% 将信号数据转换成各种物理量 Convert signal data into various physical quantities
for i = 1 : frame_num
    for k =  1 : sample_num
        for j = 1 : chirp_num
            if doppler_filter(i, j, k)
                range = (k - 1) * range_res;
                velocity = (j - 1 - chirp_num / 2) * velocity_theorical;
                rss = doppler_db(i, j, k);
                if rss ~= 1
                    switch plan
                        case 0 % 原始方案，3DFFT结果取两个峰值
                            [rss_a, a_i] = max(abs(azimuth_data(i, : , j, k)));
                            [rss_e, e_i] = max(abs(elevation_data(i, :, j, k)));
                            wx = (a_i - 1 - angle_padded_num / 2) / angle_padded_num * 2 * pi;
                            wz = angle(azimuth_data(i, a_i, j, k) * conj(elevation_data(i, e_i, j, k)) * exp(sqrt(-1) * 2 * wx));
                        case 1 % wx取fft峰值来计算，wz取wx的+-5范围之内的峰值，若无峰值，wz取第一个值
                            [rss_a, a_i] = max(abs(azimuth_data(i, : , j, k)));
                            [peaks_2, locs_2] = findpeaks(abs(elevation_data(i, :, j, k)));
                            e_i = 1;
                            temp_db = 0;
                            for l_i = 1 : length(locs_2)
                                if abs(locs_2(l_i) - a_i) <= 5 && temp_db < abs(elevation_data(i, locs_2(l_i), j, k))
                                    temp_db = abs(elevation_data(i, locs_2(l_i), j, k));
                                    e_i = locs_2(l_i);
                                end
                            end
                            wx = (a_i - 1 - angle_padded_num / 2) / angle_padded_num * 2 * pi;
                            wz = angle(azimuth_data(i, a_i, j, k) * conj(elevation_data(i, e_i, j, k)) * exp(sqrt(-1) * 2 * wx));
                        case 2 % wx取fft峰值来计算，wz取原始4个数据除以对应wx递增量后再取均值
                            [rss_a, a_i] = max(abs(azimuth_data(i, : , j, k)));
                            elevation_s = squeeze(elevation_data(i, :, j, k));
                            wx = (a_i - 1 - angle_padded_num / 2) / angle_padded_num * 2 * pi;
                            for l_i = 1 : 4
                                elevation_s(i) = elevation_s(i) .* exp(sqrt(-1) * (i - 1) * wx);
                            end
                            elevation_s = mean(elevation_s);
                            wz = angle(azimuth_data(i, a_i, j, k) * conj(elevation_s) * exp(sqrt(-1) * 2 * wx));
                    end

%                     rss = squeeze(log(abs(sum(data(i, :, :, j, k), [2, 3]))));
                    p_x = range * wx / pi;
                    p_z = range * wz / pi;
                    p_y = sqrt(range^2 - p_x^2- p_z^2); 
                    if real(p_y) ~= 0 && p_z>=-0.95 && p_z<=1 
%                     
                          point_num(i) = point_num(i) + 1;
%                           if(point_num(i)>point_res_num)
%                               disp(point_num(i))
%                           %error("???")
%                           end
                          point_set(i, :, point_num(i)) = [p_x; p_y; p_z; range; velocity; rss];
%                     
                    end
                end
            end
        end
    end
end

%% 画出多帧三维散点图 Draw multiple frames of 3D scatter plots
close all
if 0
    figure;
    for i = 1 : 1
           %figure;

            % 画 3D 图
            subplot(2, 2, 1);
            scatter3(squeeze(point_set(i, 1, 1 : point_num(i))), squeeze(point_set(i, 2, 1 : point_num(i))), squeeze(point_set(i, 3, 1 : point_num(i))), 15, abs(squeeze(point_set(i, 6, 1 : point_num(i)))));
            xlabel('x');
            ylabel('y');
            zlabel('z');
            title('3D点云');
            colormap jet;
            colorbar;
            xlim([-4,4]);
            ylim([0,7.5]);
            zlim([-4,4]);
%             axis(axis_set);

            % 画正视图
            subplot(2, 2, 2);
            scatter(squeeze(point_set(i, 1, :)), squeeze(point_set(i, 3, :)), 15, abs(squeeze(point_set(i, 6, :))));
            xlabel('x');
            ylabel('z');
            title('正视图');
            colormap jet;
            colorbar;
%             axis(axis_set([1, 2, 5, 6]));
            
            % 画俯视图
            subplot(2, 2, 3);
            scatter(squeeze(point_set(i, 1, :)), squeeze(point_set(i, 2, :)), 15, abs(squeeze(point_set(i, 6, :))));
            xlabel('x');
            ylabel('y');
            title('俯视图');
            colormap jet;
            colorbar;
%             axis(axis_set(1 : 4));

            % 画侧视图
            subplot(2, 2, 4);
            scatter(squeeze(point_set(i, 2, :)), squeeze(point_set(i, 3, :)), 15, abs(squeeze(point_set(i, 6, :))));
            xlabel('y');
            ylabel('z');
            title('侧视图');
            colormap jet;
            colorbar;
%             axis(axis_set([3 : 6]));
            pause(1);
            %saveas(gcf,[savepath,int2str(i)],'png');
            
    end
end
%% Until now we have already get the 6d points_set(x,y,z), range, velocity and Signal strength
point_to_image(file_path,point_set)
clear
key=0;
end
