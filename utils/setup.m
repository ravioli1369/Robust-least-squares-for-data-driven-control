function y = setup_plots()
        %% IEEE Standard Figure Configuration - Version 1.0
    % run this code before the plot command
    %%
    % According to the standard of IEEE Transactions and Journals: 
    % Times New Roman is the suggested font in labels. 
    % For a singlepart figure, labels should be in 8 to 10 points,
    % whereas for a multipart figure, labels should be in 8 points.
    % Width: column width: 8.8 cm; page width: 18.1 cm.
    %% width & hight of the figure
    k_scaling = 4;          % scaling factor of the figure
    % (You need to plot a figure which has a width of (8.8 * k_scaling)
    % in MATLAB, so that when you paste it into your paper, the width will be
    % scalled down to 8.8 cm  which can guarantee a preferred clearness.
    k_width_height = 1.5;      % width:hight ratio of the figure
    width = 8.8 * k_scaling;
    height = width / k_width_height;
    %% figure margins
    top = 2;  % normalized top margin
    bottom = 3;	% normalized bottom margin
    left = 3;	% normalized left margin
    right = 0.5;  % normalized right margin
    %% set default figure configurations
    set(0,'defaultFigureUnits','centimeters');
    set(0,'defaultFigurePosition',[0 0 width height]);
    set(0,'defaultLineLineWidth',0.8*k_scaling);
    set(0,'defaultAxesLineWidth',0.25*k_scaling);
    set(0,'defaultAxesGridLineStyle','-');
    set(0,'defaultAxesYGrid','on');
    set(0,'defaultAxesXGrid','on');
    set(0,'defaultAxesFontName', 'Arial');
    set(0,'defaultAxesFontSize',4*k_scaling);
    set(0,'defaultTextFontName','Times New Roman');
    set(0,'defaultTextFontSize',4*k_scaling);
    set(0,'defaultLegendFontName','Times New Roman');
    set(0,'defaultLegendFontSize',8*k_scaling);
    set(0,'defaultAxesUnits','normalized');
    set(0,'defaultAxesPosition',[left/width bottom/height (width-left-right)/width  (height-bottom-top)/height]);
    % set(0,'defaultAxesColorOrder',[0 0 0]);
    set(0,'defaultAxesTickDir','out');
    set(0,'defaultFigurePaperPositionMode','auto');
    % you can change the Legend Location to whatever as you wish
    set(0,'defaultLegendLocation','best');
    set(0,'defaultLegendBox','on');
    set(0,'defaultLegendOrientation','vertical');
    set(0, 'defaulttextInterpreter', 'latex');
end