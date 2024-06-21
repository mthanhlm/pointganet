function helperLabelColorbar(ax,classNames)
% Colormap for the original classes.
cmap = [[0 0 255];
    [0 255 0];
    [255 192 203];
    [255 255 0];
    [255 0 255];
    [255 165 0];
    [139 0 150];
    [255 0 0]];
cmap = cmap./255;
cmap = cmap(1:numel(classNames),:);
colormap(ax,cmap);

% Add colorbar to current figure.
c = colorbar(ax);
c.Color = 'w';

% Center tick labels and use class names for tick marks.
numClasses = size(classNames,1);
c.Ticks = 1:1:numClasses;
c.TickLabels = classNames;

% Remove tick mark.
c.TickLength = 0;
end