function Verify_Saved_IQ()
    %% 1. 选择要验证的 .mat 文件
    [filename, pathname] = uigetfile('*.mat', '请选择保存的 IQ_Data 文件');
    if isequal(filename, 0)
        disp('用户取消了选择。');
        return;
    end
    mat_filepath = fullfile(pathname, filename);
    disp(['正在加载: ', mat_filepath]);
    
    %% 2. 加载数据
    try
        load(mat_filepath, 'local_data_H', 'local_data_V');
    catch
        error('无法加载文件，或者文件中没有 local_data_H 和 local_data_V 变量。');
    end
    
    [num_pulses, num_gates] = size(local_data_H);
    fprintf('成功加载数据: 脉冲数 = %d, 距离库数 = %d\n', num_pulses, num_gates);

    %% 3. 尝试自动读取对应的 .txt 标签文件获取坐标信息
    txt_filename = strrep(filename, '.mat', '.txt');
    txt_filepath = fullfile(pathname, '..', 'Labels', txt_filename);
    
    center_dist = [];
    target_vel = [];
    azimuth = [];
    beam_layer = [];
    
    if exist(txt_filepath, 'file')
        fid = fopen(txt_filepath, 'r');
        while ~feof(fid)
            line = fgetl(fid);
            if contains(line, 'Distance(m):')
                center_dist = sscanf(line, 'Distance(m): %f');
            elseif contains(line, 'Velocity(m/s):')
                target_vel = sscanf(line, 'Velocity(m/s): %f');
            elseif contains(line, 'Azimuth(deg):')
                azimuth = sscanf(line, 'Azimuth(deg): %f');
            elseif contains(line, 'Beam_Layer:')
                beam_layer = sscanf(line, 'Beam_Layer: %d');
            end
        end
        fclose(fid);
        disp('成功读取对应的标签文件，坐标轴将还原为真实物理量。');
    else
        disp('未找到对应的 .txt 标签文件，将使用默认物理坐标轴。');
    end

    %% 4. 重建坐标轴雷达参数 (与提取程序保持一致)
    f_c = 9300e6; 
    c = 3e8; 
    lambda = c / f_c;
    PRF = 2900/2; 
    range_res = 30;

    % 重建速度轴
    doppler_bins = (-num_pulses/2 : num_pulses/2-1)';
    velocity_axis = -doppler_bins * (lambda * PRF) / (2 * num_pulses);

    % 重建距离轴
    % 保存的是某个方位上的完整距离切片，因此直接按真实距离库恢复
    range_axis = (1:num_gates) * range_res;

    if ~isempty(center_dist)
        title_info = sprintf('第 %d 层 | 方位: %.2f° | 目标距离: %.1fm', ...
            beam_layer, azimuth, center_dist);
    else
        title_info = '验证 RD 图';
    end

    %% 5. 计算 H 通道和 V 通道的 RD 图
    win = hanning(num_pulses);
    
    RD_H = squeeze(20*log10(abs(fftshift(fft(local_data_H .* win, [], 1), 1))));
    RD_V = squeeze(20*log10(abs(fftshift(fft(local_data_V .* win, [], 1), 1))));

    %% 6. 绘图展示
    fig = figure('Name', '已保存数据的双通道 RD 验证图', ...
        'NumberTitle', 'off', 'Position',[100, 100, 1200, 550]);
    
    % H 通道 RD 图
    ax1 = subplot(1, 2, 1);
    imagesc(ax1, range_axis, velocity_axis, RD_H);
    set(ax1, 'YDir', 'normal'); 
    colormap(ax1, jet);
    colorbar(ax1);
    xlabel(ax1, '距离 (m)');
    ylabel(ax1, '速度 (m/s)');
    title(ax1, {'【H 通道】 RD 能量图', title_info}, ...
        'FontSize', 12, 'FontWeight', 'bold');
    
    % V 通道 RD 图
    ax2 = subplot(1, 2, 2);
    imagesc(ax2, range_axis, velocity_axis, RD_V);
    set(ax2, 'YDir', 'normal'); 
    colormap(ax2, jet);
    colorbar(ax2);
    xlabel(ax2, '距离 (m)');
    ylabel(ax2, '速度 (m/s)');
    title(ax2, {'【V 通道】 RD 能量图', title_info}, ...
        'FontSize', 12, 'FontWeight', 'bold');
    
    % 如果有记录目标速度，在图上打红色十字
    if ~isempty(target_vel) && ~isempty(center_dist)
        [~, rg_idx] = min(abs(range_axis - center_dist));
        exact_dist = range_axis(rg_idx);

        hold(ax1, 'on');
        plot(ax1, exact_dist, target_vel, 'r+', 'MarkerSize', 15, 'LineWidth', 2);

        hold(ax2, 'on');
        plot(ax2, exact_dist, target_vel, 'r+', 'MarkerSize', 15, 'LineWidth', 2);
    end

    linkaxes([ax1, ax2], 'xy');
end
