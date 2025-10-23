function cut_concentration_rectangles(varargin)
    %% microPAD Colorimetric Analysis — Concentration Region Cutting Tool
    %% Interactively select and save polygonal concentration regions from strip images
    %% Author: Veysel Y. Yilmaz
    %
    % Inputs (Name-Value pairs):
    % - 'numSquares': number of regions to capture per strip
    % - 'aspectRatio': width/height of the reference strip
    % - 'coverage': fraction of image width to fill (default 0.995)
    % - 'gapPercent': gap as percent of region width (0..1 or 0..100)
    % - 'inputFolder' | 'outputFolder': override default I/O folders
    % - 'preserveFormat' | 'jpegQuality' | 'saveCoordinates': output behavior
    %
    % Outputs/Side effects:
    % - Writes polygon crops to 3_concentration_rectangles/[phone]/con_*/
    % - Writes consolidated coordinates.txt at phone level (atomic, no duplicate rows per image)
    %
    % Behavior:
    % - Cuts N region crops and saves into con_0..con_(N-1) subfolders for each strip
    % - All polygon coordinates written to single phone-level coordinates.txt

%% ========================================================================
    %% EXPERIMENT CONFIGURATION CONSTANTS - Modify here for different setups
    %% ========================================================================
    % Validate varargin structure to prevent silent failures
    if mod(length(varargin), 2) ~= 0
        error('concentration:invalid_args', 'Parameters must be provided as name-value pairs');
    end

    % === DATASET AND FOLDER STRUCTURE ===
    INPUT_FOLDER = '2_micropad_papers';     % Source rectangular crop folder
    OUTPUT_FOLDER = '3_concentration_rectangles';  % Output concentration regions folder

    % === OUTPUT FORMATTING ===
    PRESERVE_FORMAT = true;                         % Keep original image format when saving
    JPEG_QUALITY = 100;                             % JPEG quality when writing JPEGs
    SAVE_COORDINATES = true;                        % Save polygon vertices to coordinates.txt

    % === DEFAULT GEOMETRY / SELECTION ===
    DEFAULT_NUM_SQUARES = 7;                        % Number of concentration regions per strip
    DEFAULT_ASPECT_RATIO = 7.6;                     % Reference strip width/height
    DEFAULT_COVERAGE = 0.995;                       % Coverage fraction (how much of image width to use)
    DEFAULT_GAP_PERCENT = 0.2;                      % Gap as fraction of each region width

    % === CAMERA / PROJECTION DEFAULTS ===
    DEFAULT_FOCAL_LENGTH = 1.0;                     % Normalized focal length
    DEFAULT_CAMERA_HEIGHT = 2.0;                    % Camera Z at center view
    DEFAULT_MAX_ANGLE_DEG = 60;                     % Max camera tilt
    DEFAULT_X_RANGE = [-1, 1];                      % Horizontal camera range
    DEFAULT_Y_RANGE = [-1, 1];                      % Vertical camera range
    DEFAULT_Z_RANGE = [0.7, 6.0];                   % Camera distance range

    % === VISUAL / UI BEHAVIOR ===
    DEFAULT_DIM_FACTOR = 0.3;                       % UI dimming factor

    % === NAMING / FILE CONSTANTS ===
    COORDINATE_FILENAME = 'coordinates.txt';        % Coordinates log filename
    SUPPORTED_FORMATS = {'.jpg','.jpeg','.png','.bmp','.tif','.tiff'}; % Preserve formats
    ALLOWED_IMAGE_EXTENSIONS = {'*.jpg','*.jpeg','*.png','*.bmp','*.tiff','*.tif'}; % Input patterns
    CONC_FOLDER_PREFIX = 'con_';                    % Concentration subfolder prefix (configurable)

    % === UI CONSTANTS ===
    UI_CONST = struct();
    UI_CONST.fontSize = struct(...
        'title', 16, ...
        'path', 12, ...
        'button', 12, ...
        'label', 11, ...
        'value', 11, ...
        'info', 10, ...
        'instruction', 10, ...
        'preview', 14);
    UI_CONST.colors = struct(...
        'background', 'black', ...
        'foreground', 'white', ...
        'panel', [0.1 0.1 0.1], ...
        'stop', [0.8 0.2 0.2], ...
        'accept', [0.2 0.7 0.2], ...
        'retry', [0.8 0.8 0.2], ...
        'skip', [0.7 0.2 0.2], ...
        'polygon', 'red', ...
        'info', 'yellow', ...
        'path', [0.7 0.7 0.7], ...
        'apply', [0.2 0.4 0.8]);
    UI_CONST.positions = struct(...
        'figure', [0 0 1 1], ...
        'title', [0.1 0.93 0.8 0.04], ...
        'pathDisplay', [0.1 0.89 0.8 0.03], ...
        'image', [0.02 0.16 0.96 0.72], ...
        'controlPanel', [0.02 0.02 0.50 0.10], ...
        'cutButtonPanel', [0.55 0.02 0.43 0.10], ...
        'previewPanel', [0.25 0.02 0.50 0.10], ...
        'stopButton', [0.02 0.93 0.06 0.05], ...
        'instructions', [0.02 0.125 0.96 0.025], ...
        'previewLeft', [0.02 0.16 0.47 0.72], ...
        'previewRight', [0.51 0.16 0.47 0.72]);
    UI_CONST.polygon = struct(...
        'lineWidth', 3, ...
        'borderWidth', 2);

    %% Build configuration from constants and overrides
    cfg = createConfiguration(INPUT_FOLDER, OUTPUT_FOLDER, PRESERVE_FORMAT, JPEG_QUALITY, SAVE_COORDINATES, ...
                              DEFAULT_NUM_SQUARES, ...
                              DEFAULT_ASPECT_RATIO, DEFAULT_COVERAGE, DEFAULT_GAP_PERCENT, ...
                              DEFAULT_FOCAL_LENGTH, DEFAULT_CAMERA_HEIGHT, DEFAULT_MAX_ANGLE_DEG, DEFAULT_X_RANGE, DEFAULT_Y_RANGE, DEFAULT_Z_RANGE, ...
                              DEFAULT_DIM_FACTOR, COORDINATE_FILENAME, SUPPORTED_FORMATS, ALLOWED_IMAGE_EXTENSIONS, varargin{:});
    
    % Inject UI config
    cfg.ui = createUIConfiguration(UI_CONST);
    % Inject concentration folder prefix for naming consistency
    cfg.concFolderPrefix = CONC_FOLDER_PREFIX;

    try
        processAllFolders(cfg);
        fprintf('>> Concentration region cutting completed successfully!\n');
    catch ME
        handleError(ME);
    end
end

%% -----------------------------------------------------------------------
function cfg = createConfiguration(inputFolder, outputFolder, preserveFormat, jpegQuality, saveCoordinates, ...
                                   defaultNumSquares, ...
                                   defaultAspectRatio, defaultCoverage, defaultGapPercent, ...
                                   defaultFocalLength, defaultCameraHeight, defaultMaxAngleDeg, defaultXRange, defaultYRange, defaultZRange, ...
                                   defaultDimFactor, coordinateFileName, supportedFormats, allowedImageExtensions, varargin)
    parser = inputParser;
    parser.addParameter('numSquares', defaultNumSquares, @(x) validateattributes(x, {'numeric'}, {'scalar','integer','>=',1,'<=',20}));
    % Allow overriding default folders and output settings
    validateFolder = @(s) validateattributes(s, {'char', 'string'}, {'nonempty', 'scalartext'});
    parser.addParameter('inputFolder', inputFolder, validateFolder);
    parser.addParameter('outputFolder', outputFolder, validateFolder);
    parser.addParameter('preserveFormat', preserveFormat, @(x) islogical(x));
    parser.addParameter('jpegQuality', jpegQuality, @(x) validateattributes(x, {'numeric'}, {'scalar','>=',1,'<=',100}));
    parser.addParameter('saveCoordinates', saveCoordinates, @(x) islogical(x));
    % Geometry / reference rectangle parameters
    parser.addParameter('aspectRatio', defaultAspectRatio, @(x) validateattributes(x, {'numeric'}, {'scalar','>',0}));
    parser.addParameter('coverage', defaultCoverage, @(x) validateattributes(x, {'numeric'}, {'scalar','>',0,'<=',1}));
    % Gap percent of region width (world space).
    parser.addParameter('gapPercent', defaultGapPercent, @(x) isnumeric(x) && isscalar(x) && x>=0);
    parser.parse(varargin{:});

    cfg.numSquares = parser.Results.numSquares;

    % Validate numSquares for practical limits
    if cfg.numSquares > 15
        warning('concentration:many_squares', 'Large numSquares (%d) may cause UI layout issues and small regions', cfg.numSquares);
    end

    % Path configuration - Dynamic path resolution using parsed values
    cfg = addPathConfiguration(cfg, parser.Results.inputFolder, parser.Results.outputFolder);

    % Output configuration using parsed values
    cfg.output.preserveFormat = parser.Results.preserveFormat;
    cfg.output.jpegQuality = parser.Results.jpegQuality;
    cfg.output.saveCoordinates = parser.Results.saveCoordinates;
    cfg.output.supportedFormats = supportedFormats;
    cfg.allowedImageExtensions = allowedImageExtensions;

    % Coordinate file naming (centralized)
    cfg.coordinateFileName = coordinateFileName;

    % Geometry is driven by a fixed reference rectangle and coverage fraction.
    cfg.geometry = struct();
    cfg.geometry.aspectRatio = parser.Results.aspectRatio;     % width/height
    cfg.geometry.useFixedReference = true;                     % enforce reference rectangle based polygons
    % Gap percent of region width in world space; accepts 0..1 or 0..100.
    gp = parser.Results.gapPercent;
    if gp > 1
        gp = gp / 100;
    end
    cfg.geometry.gapPercentWidth = gp;

    % Coverage fraction (how much of image width to fill)
    cfg.coverage = parser.Results.coverage;

    % 3D perspective camera configuration
    cameraCfg = struct();
    cameraCfg.usePerspectiveModel = true;
    cameraCfg.f = defaultFocalLength;
    cameraCfg.z = defaultCameraHeight;
    cameraCfg.maxAngleDeg = defaultMaxAngleDeg;
    cameraCfg.xRange = defaultXRange;
    cameraCfg.yRange = defaultYRange;
    cameraCfg.zRange = defaultZRange;
    % Default initial viewpoint (centered, straight-on view)
    cameraCfg.defaultViewpoints = struct();
    cameraCfg.defaultViewpoints.centerX = 0;                  % Default X position (centered)
    cameraCfg.defaultViewpoints.centerY = 0;                  % Default Y position (centered)
    cameraCfg.defaultViewpoints.centerZ = defaultCameraHeight;% Default camera height
    cfg.camera = cameraCfg;

    % Preview/visual configuration
    cfg.dimFactor = defaultDimFactor;
end

function uiCfg = createUIConfiguration(UI_CONST)
    uiCfg = UI_CONST;
end

%% Main Processing Functions
function processAllFolders(cfg)
    validatePaths(cfg);
    phoneList = getSubFolders(cfg.inputPath);

    if isempty(phoneList)
        error('concentration:no_phones', 'No phone folders found in: %s', cfg.inputPath);
    end

    fprintf('Found %d phone(s): %s\n', numel(phoneList), strjoin(phoneList, ', '));
    
    % Iterate explicitly to avoid cellfun side-effect warnings
    executeInFolder(cfg.inputPath, @processPhones);

    function processPhones()
        for i = 1:numel(phoneList)
            processPhone(phoneList{i}, cfg);
        end
    end
end

function processPhone(phoneName, cfg)
    fprintf('\r\n=== Processing Phone: %s ===\r\n', phoneName);
    if ~isfolder(phoneName)
        warning('concentration:missing_phone', 'Phone folder missing: %s', phoneName);
        return;
    end
    executeInFolder(phoneName, @() processImagesInPhone(phoneName, cfg));
end

function processImagesInPhone(phoneName, cfg)
    imageList = getImageFiles('.', cfg.allowedImageExtensions);
    if isempty(imageList)
        warning('concentration:no_images', 'No images found for phone folder: %s', phoneName);
        return;
    end

    fprintf('Found %d images\n', numel(imageList));

    outputDir = createOutputDirectory(cfg.outputPath, phoneName, cfg.projectRoot, cfg.numSquares, cfg.concFolderPrefix);

    memory = initializeMemory();
    persistentFig = [];

    try
        for idx = 1:numel(imageList)
            if ~isempty(persistentFig) && ~isvalid(persistentFig)
                persistentFig = [];
            end
            [memory, success, persistentFig] = processOneImageWithPersistentWindow(imageList{idx}, outputDir, cfg, memory, idx == 1, persistentFig, phoneName);
            logProcessingResult(success, idx == 1);
        end
    catch ME
        if ~isempty(persistentFig) && isvalid(persistentFig)
            close(persistentFig);
        end
        rethrow(ME);
    end

    if ~isempty(persistentFig) && isvalid(persistentFig)
        close(persistentFig);
    end

    fprintf('Completed: %s\n', composePathSegments(phoneName));
end



function [updatedMemory, success, fig] = processOneImageWithPersistentWindow(imageName, outputDir, cfg, memory, isFirst, fig, phoneName)
    updatedMemory = memory;
    success = false;
    
    % Load image
    [img, isValid] = loadImage(imageName);
    if ~isValid, return; end
    
    % Display processing info
    displayProcessingInfo(imageName, memory, isFirst);
    
    % Interactive region selection with persistent window
    [polygonParams, finalSliders, fig] = showInteractiveGUIWithPersistentWindow(img, imageName, phoneName, cfg, memory, isFirst, fig);
    
    if ~isempty(polygonParams)
        saveCroppedRegions(img, imageName, polygonParams, outputDir, cfg);
        updatedMemory = updateMemory(memory, polygonParams, img, finalSliders);
        success = true;
    else
        fprintf('  Image skipped by user\n');
    end
end

function [polygonParams, finalSliders, fig] = showInteractiveGUIWithPersistentWindow(img, imageName, phoneName, cfg, memory, isFirst, fig)
    % Show interactive GUI with persistent window and retry loop for preview
    polygonParams = [];
    finalSliders = struct('left', 0, 'right', 0);
    
    % Create figure if needed
    if isempty(fig) || ~isvalid(fig)
        fig = createFigure(imageName, phoneName, cfg);
    end
    
    while true
        % Clear and rebuild UI for concentration region editing mode
        clearAndRebuildUI(fig, 'editing', img, imageName, phoneName, cfg, memory, isFirst);
        
        [action, userPolygons, userSliders] = waitForUserAction(fig);
        
        switch action
            case 'skip'
                return;
            case 'stop'
                close(fig);
                error('User stopped execution');
            case 'accept'
                polygonParams = userPolygons;
                finalSliders = userSliders;
                
                % Clear and rebuild UI for preview mode
                clearAndRebuildUI(fig, 'preview', img, imageName, phoneName, cfg, memory, isFirst, polygonParams, finalSliders);
                
                [prevAction, ~, ~] = waitForUserAction(fig);
                
                switch prevAction
                    case 'accept'
                        return;
                    case {'skip', 'stop'}
                        if strcmp(prevAction, 'stop')
                            close(fig);
                            error('User stopped execution');
                        end
                        polygonParams = [];
                        return;
                    case 'retry'
                        % Continue loop to redraw editing interface
                        continue;
                end
        end
    end
end

%% Unified UI Clear and Rebuild Function
function clearAndRebuildUI(fig, mode, img, imageName, phoneName, cfg, memory, isFirst, polygonParams, finalSliders)
    % Modes: 'editing' (interactive polygon adjustment), 'preview' (final confirmation with crops)

    % Get existing GUI data
    guiData = get(fig, 'UserData');
    
    clearAllUIElements(fig, guiData);
    
    % Rebuild UI based on mode
    switch mode
        case 'editing'
            buildConcentrationEditUI(fig, img, imageName, phoneName, cfg, memory, isFirst);
        case 'preview'
            % Pass polygonParams and finalSliders to preview mode if provided
            if nargin >= 10 && exist('polygonParams', 'var') && exist('finalSliders', 'var')
                buildPreviewUI(fig, img, imageName, phoneName, cfg, polygonParams, finalSliders);
            else
                error('concentration:preview_args', 'Preview mode requires polygonParams and finalSliders');
            end
    end
end

function polys = collectValidPolygons(guiData)
    % Extract valid polygon objects from GUI data

    polys = [];
    if isempty(guiData) || ~isstruct(guiData) || ~isfield(guiData, 'polygons')
        return;
    end
    if ~iscell(guiData.polygons)
        return;
    end

    validMask = cellfun(@isvalid, guiData.polygons);
    if any(validMask)
        polys = [guiData.polygons{validMask}]';
    end
end

function clearAllUIElements(fig, guiData)
    % Delete all UI controls, panels, axes, and polygon ROIs from figure

    % Collect all graphics objects
    allObjects = findall(fig);
    if isempty(allObjects)
        set(fig, 'UserData', []);
        return;
    end

    % Filter by type using logical indexing
    objTypes = get(allObjects, 'Type');
    if ~iscell(objTypes), objTypes = {objTypes}; end

    isControl = strcmp(objTypes, 'uicontrol');
    isPanel = strcmp(objTypes, 'uipanel');
    isAxes = strcmp(objTypes, 'axes');

    toDelete = allObjects(isControl | isPanel | isAxes);

    % Add polygon ROIs from guiData
    validPolys = collectValidPolygons(guiData);
    if ~isempty(validPolys)
        toDelete = [toDelete; validPolys];
    end

    % Bulk delete all valid objects
    if ~isempty(toDelete)
        validMask = arrayfun(@isvalid, toDelete);
        delete(toDelete(validMask));
    end

    % Cleanup any remaining ROI polygon objects
    rois = findobj(fig, '-isa', 'images.roi.Polygon');
    if ~isempty(rois)
        validRois = rois(arrayfun(@isvalid, rois));
        if ~isempty(validRois)
            delete(validRois);
        end
    end

    % Reset UserData
    set(fig, 'UserData', []);
end

function buildConcentrationEditUI(fig, img, imageName, phoneName, cfg, memory, isFirst)
    % Update figure name
    set(fig, 'Name', sprintf('Concentration Region Cutter - %s - %s', composePathSegments(phoneName), imageName));
    
    % Create GUI data
    guiData = struct();
    guiData.mode = 'editing';
    
    % Create UI components
    guiData.titleHandle = createTitle(fig, phoneName, imageName, isFirst, cfg);
    guiData.pathHandle = createPathDisplay(fig, phoneName, imageName, cfg);
    
    % Display image
    guiData.imgAxes = createImageAxes(fig, img, cfg);
    
    % Calculate parameters and create polygons
    [imageHeight, imageWidth, ~] = size(img);
    % Always use perspective camera driven mode (x, y, z)
    viewParams = calculateViewSliderParameters(cfg);
    initialSliders = getInitialViewValues(memory, isFirst, viewParams, cfg);
    guiData.polygons = createPolygons(img, initialSliders, cfg, memory, isFirst);
    guiData.sliderHandles = createViewControlPanel(fig, viewParams, initialSliders, guiData.polygons, cfg);
    guiData.cutButtonPanel = createConcentrationEditButtonPanel(fig, cfg);
    guiData.stopButton = createStopButton(fig, cfg);
    guiData.instructionText = createInstructions(fig, memory, isFirst, cfg);
    
    % Initialize GUI data
    guiData = initializeGUIData(guiData, imageWidth, imageHeight, initialSliders);
    guiData.action = '';
    
    set(fig, 'UserData', guiData);
end


function buildPreviewUI(fig, img, imageName, phoneName, cfg, polygonParams, finalSliders)
    % Update figure name
    set(fig, 'Name', sprintf('PREVIEW - %s - %s', composePathSegments(phoneName), imageName));
    
    % Create GUI data
    guiData = struct();
    guiData.mode = 'preview';
    guiData.savedPolygonParams = polygonParams;
    guiData.savedSliders = finalSliders;
    
    % Create preview title and path
    titleText = sprintf('PREVIEW: %s - %s', composePathSegments(phoneName), imageName);
    guiData.titleHandle = uicontrol('Parent', fig, 'Style', 'text', 'String', titleText, ...
                                   'Units', 'normalized', 'Position', cfg.ui.positions.title, ...
                                   'FontSize', cfg.ui.fontSize.title, 'FontWeight', 'bold', ...
                                   'ForegroundColor', cfg.ui.colors.foreground, ...
                                   'BackgroundColor', cfg.ui.colors.background, ...
                                   'HorizontalAlignment', 'center');
    
    pathText = sprintf('PREVIEW - Path: %s | Image: %s', composePathSegments(phoneName), imageName);
    guiData.pathHandle = uicontrol('Parent', fig, 'Style', 'text', 'String', pathText, ...
                                  'Units', 'normalized', 'Position', cfg.ui.positions.pathDisplay, ...
                                  'FontSize', cfg.ui.fontSize.path, 'FontWeight', 'normal', ...
                                  'ForegroundColor', cfg.ui.colors.path, ...
                                  'BackgroundColor', cfg.ui.colors.background, ...
                                  'HorizontalAlignment', 'center');
    
    % Create preview axes
    [guiData.leftAxes, guiData.rightAxes] = createPreviewAxes(fig, img, polygonParams, cfg);
    
    % Create buttons
    guiData.stopButton = createStopButton(fig, cfg);
    guiData.buttonPanel = createPreviewButtons(fig, cfg);
    
    guiData.action = '';
    
    set(fig, 'UserData', guiData);
end

%% Figure and UI Component Creation
function fig = createFigure(imageName, phoneName, cfg)
    titleText = sprintf('Concentration Region Cutter - %s - %s', composePathSegments(phoneName), imageName);
    fig = figure('Name', titleText, ...
                'Units', 'normalized', 'Position', cfg.ui.positions.figure, ...
                'MenuBar', 'none', 'ToolBar', 'none', ...
                'Color', cfg.ui.colors.background, 'KeyPressFcn', @keyPressHandler);
    
    % Ensure window focus and maximized window state
    drawnow limitrate;
    pause(0.05);
    set(fig, 'WindowState', 'maximized');
    figure(fig);
    drawnow limitrate;
end

%% UI Component Functions
function titleHandle = createTitle(fig, phoneName, imageName, isFirst, cfg)
    if isFirst
        titleText = sprintf('First Image: %s - %s (Set region positions)', composePathSegments(phoneName), imageName);
    else
        titleText = sprintf('Using Memory: %s - %s (Adjust if needed)', composePathSegments(phoneName), imageName);
    end
    
    titleHandle = uicontrol('Parent', fig, 'Style', 'text', 'String', titleText, ...
                           'Units', 'normalized', 'Position', cfg.ui.positions.title, ...
                           'FontSize', cfg.ui.fontSize.title, 'FontWeight', 'bold', ...
                           'ForegroundColor', cfg.ui.colors.foreground, ...
                           'BackgroundColor', cfg.ui.colors.background, ...
                           'HorizontalAlignment', 'center');
end

function pathHandle = createPathDisplay(fig, phoneName, imageName, cfg)
    pathText = sprintf('Path: %s | Image: %s', composePathSegments(phoneName), imageName);
    pathHandle = uicontrol('Parent', fig, 'Style', 'text', 'String', pathText, ...
                          'Units', 'normalized', 'Position', cfg.ui.positions.pathDisplay, ...
                          'FontSize', cfg.ui.fontSize.path, 'FontWeight', 'normal', ...
                          'ForegroundColor', cfg.ui.colors.path, ...
                          'BackgroundColor', cfg.ui.colors.background, ...
                          'HorizontalAlignment', 'center');
end

function imgAxes = createImageAxes(fig, img, cfg)
    imgAxes = axes('Parent', fig, 'Units', 'normalized', 'Position', cfg.ui.positions.image);
    imshow(img, 'Parent', imgAxes, 'InitialMagnification', 'fit');
    axis(imgAxes, 'image');
    axis(imgAxes, 'tight');
    hold(imgAxes, 'on');
end

function stopButton = createStopButton(fig, cfg)
    stopButton = uicontrol('Parent', fig, 'Style', 'pushbutton', ...
                          'String', 'STOP', 'FontSize', cfg.ui.fontSize.button, 'FontWeight', 'bold', ...
                          'Units', 'normalized', 'Position', cfg.ui.positions.stopButton, ...
                          'BackgroundColor', cfg.ui.colors.stop, 'ForegroundColor', cfg.ui.colors.foreground, ...
                          'Callback', @(~,~) stopExecution(fig));
end

function updateViewDisplays(sliderHandles, viewX, viewY, viewZ, cfg)
    % Update perspective view slider value displays
    degX = viewX * cfg.camera.maxAngleDeg;
    degY = viewY * cfg.camera.maxAngleDeg;
    if isfield(sliderHandles, 'leftValue')
        set(sliderHandles.leftValue, 'String', sprintf('x=%.2f (%.1f°)', viewX, degX));
    end
    if isfield(sliderHandles, 'rightValue')
        set(sliderHandles.rightValue, 'String', sprintf('y=%.2f (%.1f°)', viewY, degY));
    end
    if isfield(sliderHandles, 'zValue')
        set(sliderHandles.zValue, 'String', sprintf('z=%.2f', viewZ));
    end
end

%% UI Component Creators
function label = createLabel(parent, text, position, cfg)
    label = uicontrol('Parent', parent, 'Style', 'text', 'String', text, ...
                     'Units', 'normalized', 'Position', position, ...
                     'FontSize', cfg.ui.fontSize.label, 'FontWeight', 'bold', ...
                     'ForegroundColor', cfg.ui.colors.foreground, ...
                     'BackgroundColor', cfg.ui.colors.panel, ...
                     'HorizontalAlignment', 'center');
end

function polygons = createPolygons(img, initialSliders, cfg, memory, isFirst)
    polygonVertices = getInitialPolygons(img, memory, isFirst, cfg, initialSliders);
    polygons = cell(1, size(polygonVertices, 1));
    
    % Create polygons with consistent styling
    for i = 1:size(polygonVertices, 1)
        polygons{i} = drawpolygon('Position', squeeze(polygonVertices(i,:,:)), ...
                                 'Color', cfg.ui.colors.polygon, ...
                                 'LineWidth', cfg.ui.polygon.lineWidth, ...
                                 'MarkerSize', 8, ...
                                 'Selected', false);
        
        % Add listeners for quadrilateral enforcement and real-time feedback
        addlistener(polygons{i}, 'ROIMoved', @(~,~) enforceQuadrilateral(polygons{i}));
        % Editing view: no textual labels on the original image
    end
end

function sliderHandles = createViewControlPanel(fig, viewParams, initialView, polygons, cfg)
    panel = uipanel('Parent', fig, 'Units', 'normalized', ...
                   'Position', cfg.ui.positions.controlPanel, ...
                   'BackgroundColor', cfg.ui.colors.panel, ...
                   'BorderType', 'etchedin', 'HighlightColor', cfg.ui.colors.foreground);

    uicontrol('Parent', panel, 'Style', 'text', 'String', 'Viewpoint Control', ...
             'Units', 'normalized', 'Position', [0.02 0.75 0.96 0.20], ...
             'FontSize', cfg.ui.fontSize.label, 'FontWeight', 'bold', ...
             'ForegroundColor', cfg.ui.colors.foreground, ...
             'BackgroundColor', cfg.ui.colors.panel, ...
             'HorizontalAlignment', 'center');

    createLabel(panel, 'X (left/right):', [0.03 0.50 0.28 0.20], cfg);
    leftSlider = uicontrol('Parent', panel, 'Style', 'slider', ...
                          'Min', viewParams.minX, 'Max', viewParams.maxX, ...
                          'Value', initialView.left, 'Units', 'normalized', ...
                          'Position', [0.03 0.30 0.28 0.15], ...
                          'Callback', @(~, ~) updatePolygonsFromView([], polygons, cfg, fig));
    leftValue = uicontrol('Parent', panel, 'Style', 'text', 'String', 'x=0', ...
                          'Units', 'normalized', 'Position', [0.03 0.05 0.28 0.20], ...
                          'FontSize', cfg.ui.fontSize.value, 'FontWeight', 'bold', ...
                          'ForegroundColor', cfg.ui.colors.info, 'BackgroundColor', cfg.ui.colors.panel, ...
                          'HorizontalAlignment', 'center');

    createLabel(panel, 'Y (up/down):', [0.365 0.50 0.28 0.20], cfg);
    rightSlider = uicontrol('Parent', panel, 'Style', 'slider', ...
                          'Min', viewParams.minY, 'Max', viewParams.maxY, ...
                          'Value', initialView.right, 'Units', 'normalized', ...
                          'Position', [0.365 0.30 0.28 0.15], ...
                          'Callback', @(~, ~) updatePolygonsFromView([], polygons, cfg, fig));
    rightValue = uicontrol('Parent', panel, 'Style', 'text', 'String', 'y=0', ...
                          'Units', 'normalized', 'Position', [0.365 0.05 0.28 0.20], ...
                          'FontSize', cfg.ui.fontSize.value, 'FontWeight', 'bold', ...
                          'ForegroundColor', cfg.ui.colors.info, 'BackgroundColor', cfg.ui.colors.panel, ...
                          'HorizontalAlignment', 'center');

    createLabel(panel, 'Z (height):', [0.70 0.50 0.28 0.20], cfg);
    zSlider = uicontrol('Parent', panel, 'Style', 'slider', ...
                          'Min', viewParams.minZ, 'Max', viewParams.maxZ, ...
                          'Value', initialView.z, 'Units', 'normalized', ...
                          'Position', [0.70 0.30 0.28 0.15], ...
                          'Callback', @(~, ~) updatePolygonsFromView([], polygons, cfg, fig));
    zValue = uicontrol('Parent', panel, 'Style', 'text', 'String', 'z=0', ...
                          'Units', 'normalized', 'Position', [0.70 0.05 0.28 0.20], ...
                          'FontSize', cfg.ui.fontSize.value, 'FontWeight', 'bold', ...
                          'ForegroundColor', cfg.ui.colors.info, 'BackgroundColor', cfg.ui.colors.panel, ...
                          'HorizontalAlignment', 'center');

    sliderHandles = struct('leftSlider', leftSlider, 'leftValue', leftValue, ...
                          'rightSlider', rightSlider, 'rightValue', rightValue, ...
                          'zSlider', zSlider, 'zValue', zValue);
    % Initialize value strings
    updateViewDisplays(sliderHandles, initialView.left, initialView.right, initialView.z, cfg);
end

function cutButtonPanel = createConcentrationEditButtonPanel(fig, cfg)
    % Create panel for concentration editing mode buttons
    cutButtonPanel = uipanel('Parent', fig, 'Units', 'normalized', ...
                             'Position', cfg.ui.positions.cutButtonPanel, ...
                             'BackgroundColor', cfg.ui.colors.panel, ...
                             'BorderType', 'etchedin', 'HighlightColor', cfg.ui.colors.foreground);
    
    % APPLY button
    uicontrol('Parent', cutButtonPanel, 'Style', 'pushbutton', ...
             'String', 'APPLY', 'FontSize', cfg.ui.fontSize.button, 'FontWeight', 'bold', ...
             'Units', 'normalized', 'Position', [0.15 0.35 0.30 0.35], ...
             'BackgroundColor', cfg.ui.colors.apply, 'ForegroundColor', cfg.ui.colors.foreground, ...
             'Callback', @(~,~) setConcentrationEditAction(fig, 'accept'));
    
    % SKIP button
    uicontrol('Parent', cutButtonPanel, 'Style', 'pushbutton', ...
             'String', 'SKIP', 'FontSize', cfg.ui.fontSize.button, 'FontWeight', 'bold', ...
             'Units', 'normalized', 'Position', [0.55 0.35 0.30 0.35], ...
             'BackgroundColor', cfg.ui.colors.skip, 'ForegroundColor', cfg.ui.colors.foreground, ...
             'Callback', @(~,~) setConcentrationEditAction(fig, 'skip'));
end

function instructionText = createInstructions(fig, memory, isFirst, cfg)
    % Create instruction text with memory state information
    memoryText = '';
    if memory.hasSettings && ~isFirst
        if isfield(memory, 'zSlider') && ~isempty(memory.zSlider)
            memoryText = sprintf(' | Loaded: %d regions, P=(%.2f, %.2f, %.2f)', ...
                               memory.numRegions, memory.leftSlider, memory.rightSlider, memory.zSlider);
        else
            memoryText = sprintf(' | Loaded: %d regions, L=%.0f R=%.0f px', ...
                               memory.numRegions, memory.leftSlider, memory.rightSlider);
        end
        if isfield(memory, 'timestamp') && ~isempty(memory.timestamp)
            try
                timeStr = string(memory.timestamp, 'HH:mm');
            catch
                % Handle unexpected timestamp types
                timeStr = string(datetime('now'), 'HH:mm');
            end
            memoryText = [memoryText sprintf(' (%s)', timeStr)];
        end
    elseif isFirst
        memoryText = ' | First image - will establish settings for folder';
    else
        memoryText = ' | No memory - using default view settings';
    end
    
    sliderHint = 'Sliders = View (x,y,z)';
    instructionString = ['Mouse = Drag Vertices | ' sliderHint ' | APPLY = Save & Continue | SKIP = Skip | STOP = Exit | ' memoryText];
    
    instructionText = uicontrol('Parent', fig, 'Style', 'text', 'String', instructionString, ...
             'Units', 'normalized', 'Position', cfg.ui.positions.instructions, ...
             'FontSize', cfg.ui.fontSize.instruction, 'ForegroundColor', cfg.ui.colors.foreground, ...
             'BackgroundColor', cfg.ui.colors.background, 'HorizontalAlignment', 'center');
end

function buttonPanel = createPreviewButtons(fig, cfg)
    buttonPanel = uipanel('Parent', fig, 'Units', 'normalized', ...
                         'Position', cfg.ui.positions.previewPanel, ...
                         'BackgroundColor', cfg.ui.colors.panel, ...
                         'BorderType', 'etchedin', 'HighlightColor', cfg.ui.colors.foreground, ...
                         'BorderWidth', cfg.ui.polygon.borderWidth);
    
    % Create buttons
    buttons = {'ACCEPT', 'RETRY', 'SKIP'};
    positions = {[0.05 0.25 0.25 0.50], [0.375 0.25 0.25 0.50], [0.70 0.25 0.25 0.50]};
    colors = {cfg.ui.colors.accept, cfg.ui.colors.retry, cfg.ui.colors.skip};
    actions = {'accept', 'retry', 'skip'};
    
    for i = 1:numel(buttons)
        uicontrol('Parent', buttonPanel, 'Style', 'pushbutton', ...
                 'String', buttons{i}, 'FontSize', cfg.ui.fontSize.preview, 'FontWeight', 'bold', ...
                 'Units', 'normalized', 'Position', positions{i}, ...
                 'BackgroundColor', colors{i}, 'ForegroundColor', cfg.ui.colors.foreground, ...
                 'Callback', @(~,~) setPreviewAction(fig, actions{i}));
    end
end

function [leftAxes, rightAxes] = createPreviewAxes(fig, img, polygonParams, cfg)
    % Original with region overlays
    leftAxes = axes('Parent', fig, 'Units', 'normalized', 'Position', cfg.ui.positions.previewLeft);
    imshow(img, 'Parent', leftAxes, 'InitialMagnification', 'fit');
    axis(leftAxes, 'image');
    axis(leftAxes, 'tight');
    title(leftAxes, sprintf('Original with %d Concentration Regions', size(polygonParams, 1)), ...
          'Color', cfg.ui.colors.foreground, 'FontSize', cfg.ui.fontSize.button, 'FontWeight', 'bold');
    hold(leftAxes, 'on');
    
    % Draw polygon overlays with labels
    for i = 1:size(polygonParams, 1)
        poly = squeeze(polygonParams(i,:,:));
        if size(poly, 1) >= 3
            % Draw polygon outline
            plot(leftAxes, [poly(:,1); poly(1,1)], [poly(:,2); poly(1,2)], ...
                 'Color', cfg.ui.colors.polygon, 'LineWidth', cfg.ui.polygon.lineWidth);
            
            % Add label
            centerX = mean(poly(:,1));
            centerY = mean(poly(:,2));
            text(leftAxes, centerX, centerY, sprintf('C%d', i-1), ...
                 'HorizontalAlignment', 'center', 'VerticalAlignment', 'middle', ...
                 'FontSize', cfg.ui.fontSize.info, 'FontWeight', 'bold', ...
                 'Color', cfg.ui.colors.info, 'BackgroundColor', [0 0 0], ...
                 'EdgeColor', 'none');
        end
    end
    hold(leftAxes, 'off');
    
    % Preview of region organization
    rightAxes = axes('Parent', fig, 'Units', 'normalized', 'Position', cfg.ui.positions.previewRight);
    
    % Create a visualization showing region organization
    maskedImg = createMaskedPreview(img, polygonParams, cfg);
    imshow(maskedImg, 'Parent', rightAxes, 'InitialMagnification', 'fit');
    axis(rightAxes, 'image');
    axis(rightAxes, 'tight');
    title(rightAxes, 'Highlighted Concentration Regions', ...
          'Color', cfg.ui.colors.foreground, 'FontSize', cfg.ui.fontSize.button, 'FontWeight', 'bold');
end

function maskedImg = createMaskedPreview(img, polygonParams, cfg)
    % Create masked image showing concentration regions with dimmed background
    [height, width, ~] = size(img);
    totalMask = false(height, width);

    % Create combined mask from all polygon regions
    numRegions = size(polygonParams, 1);
    for i = 1:numRegions
        poly = squeeze(polygonParams(i,:,:));
        if size(poly, 1) >= 3
            regionMask = poly2mask(poly(:,1), poly(:,2), height, width);
            totalMask = totalMask | regionMask;
        end
    end

    % Apply vectorized dimming using implicit expansion
    dimFactor = cfg.dimFactor;
    maskedImg = double(img);

    % Create dimming multiplier (1 inside mask, dimFactor outside)
    dimMultiplier = double(totalMask) + (1 - double(totalMask)) * dimFactor;

    % Apply to all channels using implicit expansion
    maskedImg = maskedImg .* dimMultiplier;

    maskedImg = uint8(maskedImg);
end

%% User Interaction Functions
function setConcentrationEditAction(fig, action)
    guiData = get(fig, 'UserData');
    
    % Capture final values before accepting
    if strcmp(action, 'accept') && strcmp(guiData.mode, 'editing') && isfield(guiData, 'sliderHandles')
        guiData.finalLeft = get(guiData.sliderHandles.leftSlider, 'Value');
        guiData.finalRight = get(guiData.sliderHandles.rightSlider, 'Value');
        if isfield(guiData.sliderHandles, 'zSlider')
            guiData.finalZ = get(guiData.sliderHandles.zSlider, 'Value');
        end
    end
    
    guiData.action = action;
    set(fig, 'UserData', guiData);
    uiresume(fig);
end

function setPreviewAction(fig, action)
    guiData = get(fig, 'UserData');
    guiData.action = action;
    set(fig, 'UserData', guiData);
    uiresume(fig);
end

function stopExecution(fig)
    guiData = get(fig, 'UserData');
    guiData.action = 'stop';
    set(fig, 'UserData', guiData);
    uiresume(fig);
end

function keyPressHandler(src, event)
    guiData = get(src, 'UserData');
    
    switch event.Key
        case 'space'
            % Capture final values before accepting
            if strcmp(guiData.mode, 'editing') && isfield(guiData, 'sliderHandles')
                guiData.finalLeft = get(guiData.sliderHandles.leftSlider, 'Value');
                guiData.finalRight = get(guiData.sliderHandles.rightSlider, 'Value');
                if isfield(guiData.sliderHandles, 'zSlider')
                    guiData.finalZ = get(guiData.sliderHandles.zSlider, 'Value');
                end
            end
            guiData.action = 'accept';
        case 'escape'
            guiData.action = 'skip';
        otherwise
            return;
    end
    
    set(src, 'UserData', guiData);
    uiresume(src);
end

function [action, polygonParams, finalSliders] = waitForUserAction(fig)
    uiwait(fig);
    
    [action, polygonParams, finalSliders] = deal('', [], struct('left', 0, 'right', 0, 'z', 0));
    
    if isvalid(fig)
        guiData = get(fig, 'UserData');
        action = guiData.action;
        
        if strcmp(action, 'accept')
            if strcmp(guiData.mode, 'preview')
                polygonParams = guiData.savedPolygonParams;
                finalSliders = guiData.savedSliders;
            elseif strcmp(guiData.mode, 'editing')
                polygonParams = extractPolygonParameters(guiData);
                if isfield(guiData, 'finalLeft') && isfield(guiData, 'finalRight')
                    finalSliders.left = guiData.finalLeft;
                    finalSliders.right = guiData.finalRight;
                    if isfield(guiData, 'finalZ')
                        finalSliders.z = guiData.finalZ;
                    end
                end
                
                if isempty(polygonParams)
                    action = 'skip';
                end
            end
        end
    end
end

function polygonParams = extractPolygonParameters(guiData)
    polygonParams = [];
    if isfield(guiData, 'polygons') && iscell(guiData.polygons)
        numPolygons = numel(guiData.polygons);
        polygonParams = zeros(numPolygons, 4, 2);
        for i = 1:numPolygons
            if isvalid(guiData.polygons{i})
                polygonParams(i,:,:) = guiData.polygons{i}.Position;
            end
        end
    end
end

%% Polygon Generation and Memory Management
function polygonVertices = getInitialPolygons(img, memory, isFirst, cfg, sliders)
    [imageHeight, imageWidth, ~] = size(img);
    if memory.hasSettings && ~isFirst && ~isempty(memory.polygons)
        polygonVertices = scalePolygonsToNewDimensions(memory.polygons, memory.imageSize, [imageWidth, imageHeight]);
    else
        % Always use perspective model with fixed reference geometry
        % sliders.left -> viewX, sliders.right -> viewY, sliders.z -> viewZ
        viewZ = cfg.camera.z; if isfield(sliders, 'z'), viewZ = sliders.z; end
        polygonVertices = calculatePolygonsFromView(imageWidth, imageHeight, sliders.left, sliders.right, viewZ, cfg);
    end
end



function viewParams = calculateViewSliderParameters(cfg)
    viewParams.minX = cfg.camera.xRange(1);
    viewParams.maxX = cfg.camera.xRange(2);
    viewParams.minY = cfg.camera.yRange(1);
    viewParams.maxY = cfg.camera.yRange(2);
    viewParams.minZ = cfg.camera.zRange(1);
    viewParams.maxZ = cfg.camera.zRange(2);
end

function initialSliders = getInitialViewValues(memory, isFirst, viewParams, cfg)
    if memory.hasSettings && ~isFirst
        vx = clampToRange(memory.leftSlider, viewParams.minX, viewParams.maxX);
        vy = clampToRange(memory.rightSlider, viewParams.minY, viewParams.maxY);
        if isfield(memory, 'zSlider') && ~isempty(memory.zSlider)
            vz = clampToRange(memory.zSlider, viewParams.minZ, viewParams.maxZ);
        else
            vz = clampToRange(cfg.camera.defaultViewpoints.centerZ, viewParams.minZ, viewParams.maxZ);
        end
    else
        viewpoints = cfg.camera.defaultViewpoints;
        vx = clampToRange(viewpoints.centerX, viewParams.minX, viewParams.maxX);
        vy = clampToRange(viewpoints.centerY, viewParams.minY, viewParams.maxY);
        vz = clampToRange(viewpoints.centerZ, viewParams.minZ, viewParams.maxZ);
    end
    initialSliders = struct('left', vx, 'right', vy, 'z', vz);
end

function value = clampToRange(value, minVal, maxVal)
    value = max(minVal, min(value, maxVal));
end

function rad = localDeg2Rad(deg)
    % Convert degrees to radians
    rad = (pi/180) * deg;
end

function [worldCorners, rectGeometry] = buildWorldCoordinates(cfg, n)
    % Compute 3D world coordinates for all concentration region corners

    aspect = cfg.geometry.aspectRatio;
    aspect = max(aspect, eps);
    totalGridWidth = 1.0;
    rectHeightWorld = 1.0 / aspect;
    gridCenterX = -totalGridWidth / 2;
    gridCenterY = -rectHeightWorld / 2;

    % Compute gap size
    gp = 0.10;
    if isfield(cfg, 'geometry') && isfield(cfg.geometry, 'gapPercentWidth') && ~isempty(cfg.geometry.gapPercentWidth)
        gp = max(cfg.geometry.gapPercentWidth, 0);
    end
    denom = n + max(n-1, 0) * gp;
    if denom <= 0
        denom = max(n, 1);
    end
    w = totalGridWidth / denom;
    widthsWorld = w * ones(1, n);
    gapSizeWorld = gp * w;

    % Build world corner coordinates for each region
    worldCorners = zeros(n, 4, 2);
    xi = gridCenterX;
    for i = 1:n
        w = widthsWorld(min(i, numel(widthsWorld)));
        worldCorners(i, :, :) = [
            xi,       gridCenterY;
            xi + w,   gridCenterY;
            xi + w,   gridCenterY + rectHeightWorld;
            xi,       gridCenterY + rectHeightWorld
        ];
        xi = xi + w + gapSizeWorld;
    end

    rectGeometry = struct('gridCenterY', gridCenterY, 'rectHeightWorld', rectHeightWorld);
end

function projectedCorners = projectToCamera(worldCorners, viewX, viewY, viewZ, cfg)
    % Apply perspective projection from 3D world to 2D camera coordinates

    n = size(worldCorners, 1);

    % Compute rotation matrices
    yaw = localDeg2Rad(viewX * cfg.camera.maxAngleDeg);
    pitch = localDeg2Rad(viewY * cfg.camera.maxAngleDeg);
    Ry = [cos(yaw)  0  sin(yaw); 0 1 0; -sin(yaw) 0 cos(yaw)];
    Rx = [1 0 0; 0 cos(pitch) -sin(pitch); 0 sin(pitch) cos(pitch)];
    R = Rx * Ry;

    focalLength = cfg.camera.f;

    % Project each region's corners
    projectedCorners = zeros(n, 4, 2);
    for i = 1:n
        cornerWorldCoords = squeeze(worldCorners(i, :, :));
        worldX = cornerWorldCoords(:, 1).';
        worldY = cornerWorldCoords(:, 2).';

        cameraX = R(1,1) .* worldX + R(1,2) .* worldY;
        cameraY = R(2,1) .* worldX + R(2,2) .* worldY;
        cameraZ = R(3,1) .* worldX + R(3,2) .* worldY + viewZ;

        projectedU = (focalLength .* cameraX) ./ cameraZ;
        projectedV = (focalLength .* cameraY) ./ cameraZ;

        projectedCorners(i, :, :) = [projectedU(:) projectedV(:)];
    end
end

function polygons = scaleAndCenterPolygons(projectedCorners, imageWidth, imageHeight, cfg)
    % Scale projected coordinates to image dimensions and center within bounds

    n = size(projectedCorners, 1);

    % Compute bounds in projected space
    allU = projectedCorners(:, :, 1);
    allV = projectedCorners(:, :, 2);
    umin = min(allU(:));
    umax = max(allU(:));
    vmin = min(allV(:));
    vmax = max(allV(:));

    % Apply coverage fraction
    cov = 0.98;
    if isfield(cfg, 'coverage')
        cov = cfg.coverage;
    end
    cov = max(0.01, min(1.0, cov));

    % Width-based scaling
    denomU = max(umax - umin, eps);
    targetWidthPx = cov * imageWidth;
    scale = targetWidthPx / denomU;

    % Center within image bounds
    offX = (imageWidth - scale * (umax - umin)) / 2 - scale * umin;
    offY = (imageHeight - scale * (vmax - vmin)) / 2 - scale * vmin;

    % Clamp offsets to keep polygons within image
    offXmin = 0 - scale * umin;
    offXmax = imageWidth - scale * umax;
    offYmin = 0 - scale * vmin;
    offYmax = imageHeight - scale * vmax;
    offX = min(max(offX, offXmin), offXmax);
    offY = min(max(offY, offYmin), offYmax);

    % Apply scaling and offset to all polygons
    polygons = zeros(n, 4, 2);
    for i = 1:n
        uv = squeeze(projectedCorners(i, :, :));
        px = offX + scale * uv(:, 1);
        py = offY + scale * uv(:, 2);
        polygons(i, :, :) = [px py];
    end
end

function polygons = calculatePolygonsFromView(imageWidth, imageHeight, viewX, viewY, viewZ, cfg)
    % Generate polygon coordinates for concentration regions from camera viewpoint
    % Applies perspective projection and scales to fit image bounds

    n = getNumRegions(cfg);

    % Step 1: Build 3D world coordinates
    [worldCorners, ~] = buildWorldCoordinates(cfg, n);

    % Step 2: Project to camera view
    projectedCorners = projectToCamera(worldCorners, viewX, viewY, viewZ, cfg);

    % Step 3: Scale and center within image bounds
    polygons = scaleAndCenterPolygons(projectedCorners, imageWidth, imageHeight, cfg);
end

function updatePolygonsFromView(~, polygons, cfg, fig)
    guiData = get(fig, 'UserData');
    if ~strcmp(guiData.mode, 'editing'), return; end
    vx = get(guiData.sliderHandles.leftSlider, 'Value');
    vy = get(guiData.sliderHandles.rightSlider, 'Value');
    vz = get(guiData.sliderHandles.zSlider, 'Value');
    updateViewDisplays(guiData.sliderHandles, vx, vy, vz, cfg);

    % Persist for memory
    guiData.finalLeft = vx; guiData.finalRight = vy; guiData.finalZ = vz;
    set(fig, 'UserData', guiData);

    polys = calculatePolygonsFromView(guiData.width, guiData.height, vx, vy, vz, cfg);
    for i = 1:numel(polygons)
        if isvalid(polygons{i})
            set(polygons{i}, 'Position', squeeze(polys(i,:,:)));
        end
    end
    % Throttle UI refresh for smoother interaction
    drawnow limitrate
end

%% File Operations
function n = getNumRegions(cfg)
    n = cfg.numSquares;
end

function saveCroppedRegions(img, imageName, polygons, outputDir, cfg)
    [~, baseName, extOrig] = fileparts(imageName);
    extOrig = lower(extOrig);
    supported = cfg.output.supportedFormats;

    for i = 1:size(polygons, 1)
        poly = squeeze(polygons(i,:,:));
        cropped = cropImageWithPolygon(img, poly);
        if isempty(cropped)
            warning('concentration:empty_crop', 'Empty crop for region %d in image %s', i-1, imageName);
            continue;
        end
        concDir = fullfile(outputDir, sprintf('%s%d', cfg.concFolderPrefix, i-1));
        if ~(exist(concDir, 'dir') == 7), mkdir(concDir); end
        outExt = determineOutputExtension(extOrig, supported, cfg.output.preserveFormat);
        outPath = fullfile(concDir, sprintf('%s_%s%d%s', baseName, cfg.concFolderPrefix, i-1, outExt));
        saveImageWithFormat(cropped, outPath, outExt, cfg);

        % Write coordinates directly to phone-level coordinates.txt
        if isfield(cfg, 'output') && isfield(cfg.output, 'saveCoordinates') && cfg.output.saveCoordinates
            appendPolygonCoordinates(outputDir, baseName, i-1, poly, cfg);
        end
    end

    fprintf('  >> Saved %d concentration regions\n', size(polygons, 1));
end

function [existingNames, existingNums] = readExistingCoordinates(coordPath, scanFmt, numericCount)
    % Read existing coordinate entries from file

    existingNames = {};
    existingNums = zeros(0, numericCount);

    if ~isfile(coordPath)
        return;
    end

    fid_r = fopen(coordPath, 'rt');
    if fid_r == -1
        warning('concentration:coord_open', 'Cannot open coordinates file for reading: %s', coordPath);
        return;
    end

    % Read first line and check if it's a header
    firstLine = fgetl(fid_r);
    if ischar(firstLine)
        trimmed = strtrim(firstLine);
        expectedPrefix = 'image concentration';
        if ~strncmpi(trimmed, expectedPrefix, numel(expectedPrefix))
            fseek(fid_r, 0, 'bof');
        end
    else
        fseek(fid_r, 0, 'bof');
    end

    % Parse data lines
    data = textscan(fid_r, scanFmt, 'Delimiter', ' ', 'MultipleDelimsAsOne', true, 'CollectOutput', true);
    fclose(fid_r);

    if ~isempty(data)
        if numel(data) >= 1 && ~isempty(data{1})
            existingNames = data{1};
        end
        if numel(data) >= 2 && ~isempty(data{2})
            nums = data{2};
            if size(nums, 2) >= numericCount
                existingNums = nums(:, 1:numericCount);
            else
                % Pad with NaNs if fewer columns
                pad = nan(size(nums, 1), numericCount - size(nums, 2));
                existingNums = [nums, pad];
            end
        end
    end

    % Align name/value arrays if textscan returned mismatched lengths
    if ~isempty(existingNames) && ~isempty(existingNums)
        rows = min(numel(existingNames), size(existingNums, 1));
        if size(existingNums, 1) ~= numel(existingNames)
            existingNames = existingNames(1:rows);
            existingNums = existingNums(1:rows, :);
        end
    end
end

function [filteredNames, filteredNums] = filterConflictingEntries(existingNames, existingNums, newName, concentration)
    % Remove entries matching same image and concentration to allow override

    if isempty(existingNames)
        filteredNames = existingNames;
        filteredNums = existingNums;
        return;
    end

    existingNames = existingNames(:);
    sameImageMask = strcmp(existingNames, newName);
    sameConcentrationMask = sameImageMask & (existingNums(:, 1) == concentration);
    keepMask = ~sameConcentrationMask;

    filteredNames = existingNames(keepMask);
    filteredNums = existingNums(keepMask, :);
end

function atomicWriteCoordinates(coordPath, header, names, nums, writeFmt, coordFolder)
    % Perform atomic write of coordinate data using temp file

    tmpPath = tempname(coordFolder);
    fid_w = fopen(tmpPath, 'wt');
    if fid_w == -1
        warning('concentration:coord_open', 'Cannot open temp coordinates file for writing: %s', tmpPath);
        return;
    end

    fprintf(fid_w, '%s\n', header);

    % Write all rows
    for j = 1:numel(names)
        fprintf(fid_w, writeFmt, names{j}, nums(j, :));
    end

    fclose(fid_w);

    % Atomic move
    [ok, msg, msgid] = movefile(tmpPath, coordPath, 'f');
    if ~ok
        warning('concentration:coord_move', 'Failed to move temp file to coordinates.txt: %s (%s). Attempting fallback copy.', msg, msgid);
        [copied, cmsg, ~] = copyfile(tmpPath, coordPath, 'f');
        if ~copied
            if isfile(tmpPath)
                delete(tmpPath);
            end
            error('concentration:coord_write_fail', 'Cannot write coordinates to %s: movefile failed (%s), copyfile failed (%s). Stopping to prevent data loss.', coordPath, msg, cmsg);
        end
        if isfile(tmpPath)
            delete(tmpPath);
        end
    end
end

function appendPolygonCoordinates(phoneOutputDir, baseName, concentration, polygon, cfg)
    % Append polygon vertex coordinates to phone-level coordinates file with atomic write
    % Overwrites existing entry for same image/concentration combination

    coordFolder = phoneOutputDir;
    coordPath = fullfile(coordFolder, cfg.coordinateFileName);

    % Validate polygon shape and derive numeric column count
    if ~isnumeric(polygon) || size(polygon, 2) ~= 2
        warning('concentration:coord_polygon', 'Polygon must be an Nx2 numeric array. Skipping write for %s.', baseName);
        return;
    end
    nVerts = size(polygon, 1);
    if nVerts ~= 4
        warning('concentration:coord_vertices', 'Expected 4-vertex polygon; got %d. Proceeding with %d vertices may break downstream tools.', nVerts, nVerts);
    end
    numericCount = 1 + 2 * nVerts;

    % Build header dynamically
    headerParts = cell(1, 2 + 2 * nVerts);
    headerParts{1} = 'image';
    headerParts{2} = 'concentration';
    for v = 1:nVerts
        headerParts{2*v+1} = sprintf('x%d', v);
        headerParts{2*v+2} = sprintf('y%d', v);
    end
    header = strjoin(headerParts, ' ');

    % Prepare new row
    P = round(polygon);
    newName = baseName;
    newNums = [concentration, reshape(P.', 1, [])];

    % Build dynamic formats
    scanFmt = ['%s' repmat(' %f', 1, numericCount)];
    writeFmt = ['%s' repmat(' %.0f', 1, numericCount) '\n'];

    % Read existing entries
    [existingNames, existingNums] = readExistingCoordinates(coordPath, scanFmt, numericCount);

    % Filter out conflicting entries
    [existingNames, existingNums] = filterConflictingEntries(existingNames, existingNums, newName, concentration);

    % Combine existing and new data
    allNames = [existingNames; {newName}];
    allNums = [existingNums; newNums];

    % Atomic write
    atomicWriteCoordinates(coordPath, header, allNames, allNums, writeFmt, coordFolder);
end

function croppedImg = cropImageWithPolygon(img, polygonVertices)
    [h, w, c] = size(img);
    mask = poly2mask(polygonVertices(:,1), polygonVertices(:,2), h, w);

    % Compute tight bounding box around mask region
    cols = any(mask, 1);
    rows = any(mask, 2);
    if ~any(cols) || ~any(rows)
        croppedImg = [];
        return;
    end
    xMin = find(cols, 1, 'first');
    xMax = find(cols, 1, 'last');
    yMin = find(rows, 1, 'first');
    yMax = find(rows, 1, 'last');

    croppedImg = img(yMin:yMax, xMin:xMax, :);
    maskCrop = mask(yMin:yMax, xMin:xMax);

    % Vectorized mask application (handles grayscale c=1 correctly)
    mask3 = repmat(maskCrop, 1, 1, c);
    croppedImg(~mask3) = 0;
end

%% Memory System
function memory = initializeMemory()
    memory = struct( ...
        'hasSettings', false, ...     % Whether memory has valid settings
        'polygons', [], ...           % Polygon positions
        'imageSize', [], ...          % Original image dimensions [w h]
        'leftSlider', 0, ...          % Left slider value (width or x)
        'rightSlider', 0, ...         % Right slider value (width or y)
        'zSlider', 0, ...             % Z slider value (height) for perspective mode
        'numRegions', 0, ...          % Number of regions
        'timestamp', [], ...          % When settings were saved
        'sourceImage', '' ...         % Source image name for reference
    );
    
    % Validate initialized values
    validateattributes(memory.hasSettings, {'logical'}, {'scalar'}, 'initializeMemory', 'hasSettings');
    validateattributes(memory.leftSlider, {'numeric'}, {'scalar', 'finite'}, 'initializeMemory', 'leftSlider');
    validateattributes(memory.rightSlider, {'numeric'}, {'scalar', 'finite'}, 'initializeMemory', 'rightSlider');
    validateattributes(memory.zSlider, {'numeric'}, {'scalar', 'finite'}, 'initializeMemory', 'zSlider');
    validateattributes(memory.numRegions, {'numeric'}, {'scalar', 'integer', '>=', 0}, 'initializeMemory', 'numRegions');
end

function memory = updateMemory(oldMemory, polygons, img, sliders)
    % Input validation
    validateattributes(oldMemory, {'struct'}, {'scalar'}, 'updateMemory', 'oldMemory');
    validateattributes(polygons, {'numeric'}, {'nonempty'}, 'updateMemory', 'polygons');
    validateattributes(img, {'uint8', 'double'}, {'nonempty'}, 'updateMemory', 'img');
    validateattributes(sliders, {'struct'}, {'scalar'}, 'updateMemory', 'sliders');
    % Enforce expected dimensions using size() for robustness
    [np, nv, nc] = size(polygons);
    if nc ~= 2 || nv ~= 4
        error('concentration:polygons_dims', 'polygons must be N x 4 x 2 array (got %dx%dx%d).', np, nv, nc);
    end
    imgDims = ndims(img);
    if imgDims < 2 || imgDims > 3
        error('concentration:image_dims', 'img must be a 2-D grayscale or 3-D color image.');
    end
    
    [h, w, ~] = size(img);
    
    memory = oldMemory;  % Preserve any existing data
    memory.hasSettings = true;
    memory.polygons = polygons;
    memory.imageSize = [w, h];
    memory.leftSlider = sliders.left;
    memory.rightSlider = sliders.right;
    if isfield(sliders, 'z')
        memory.zSlider = sliders.z;
    end
    memory.numRegions = size(polygons, 1);
    % Timestamp of when settings were saved
    memory.timestamp = datetime('now');
    
    % Validate memory integrity
    if validateMemoryIntegrity(memory)
        if isfield(memory, 'zSlider') && ~isempty(memory.zSlider)
            fprintf('  >> Memory updated: %d regions, P=(%.2f, %.2f, %.2f)\n', ...
                    memory.numRegions, memory.leftSlider, memory.rightSlider, memory.zSlider);
        else
            fprintf('  >> Memory updated: %d regions, L=%.1f, R=%.1f\n', ...
                    memory.numRegions, memory.leftSlider, memory.rightSlider);
        end
    else
        warning('concentration:memory_validation', 'Memory validation failed, using defaults');
        memory.hasSettings = false;
    end
end

function isValid = validateMemoryIntegrity(memory)
    isValid = true;
    
    % Check required fields
    requiredFields = {'hasSettings', 'polygons', 'imageSize', 'leftSlider', 'rightSlider'};
    for i = 1:numel(requiredFields)
        if ~isfield(memory, requiredFields{i})
            isValid = false;
            return;
        end
    end
    
    % Check data consistency
    if memory.hasSettings
        if isempty(memory.polygons) || isempty(memory.imageSize)
            isValid = false;
            return;
        end
        
        % Ensure numeric values, allow negatives for perspective mode
        if ~isnumeric(memory.leftSlider) || ~isnumeric(memory.rightSlider) || ...
           any(isnan([memory.leftSlider memory.rightSlider]))
            isValid = false;
            return;
        end
        if isfield(memory, 'zSlider') && (~isnumeric(memory.zSlider) || any(isnan(memory.zSlider)))
            isValid = false;
            return;
        end
        
        if size(memory.imageSize, 2) ~= 2
            isValid = false;
            return;
        end
    end
end

function displayMemoryInfo(memory, isFirst)
    if isFirst
        fprintf('  >> First image - establishing baseline settings\n');
    elseif memory.hasSettings
        if isfield(memory, 'zSlider') && ~isempty(memory.zSlider)
            fprintf('  >> Using saved settings: %d regions, P=(%.2f, %.2f, %.2f)\n', ...
                    memory.numRegions, memory.leftSlider, memory.rightSlider, memory.zSlider);
        else
            fprintf('  >> Using saved settings: %d regions, L=%.1f, R=%.1f\n', ...
                    memory.numRegions, memory.leftSlider, memory.rightSlider);
        end
        if isfield(memory, 'timestamp') && ~isempty(memory.timestamp)
            try
                timeStr = string(memory.timestamp, 'HH:mm:ss');
            catch
                timeStr = string(datetime('now'), 'HH:mm:ss');
            end
            fprintf('  >> Settings from: %s\n', timeStr);
        end
    else
        fprintf('  >> No valid memory - using default view settings\n');
    end
end

function guiData = initializeGUIData(guiData, width, height, initialSliders)
    guiData.width = width;
    guiData.height = height;
    guiData.finalLeft = initialSliders.left;
    guiData.finalRight = initialSliders.right;
    if isfield(initialSliders, 'z')
        guiData.finalZ = initialSliders.z;
    end
    guiData.action = '';
    % mode is already set in buildConcentrationEditUI
end

function [img, isValid] = loadImage(imageName)
    isValid = false;
    try
        img = imread_raw(imageName);
        isValid = true;
    catch ME
        warning('concentration:read_fail', 'Cannot read %s: %s', imageName, ME.message);
        img = [];
    end
end

function displayProcessingInfo(imageName, memory, isFirst)
    fprintf('\nImage: %s', imageName);
    
    % Use the memory display function
    displayMemoryInfo(memory, isFirst);
end

function logProcessingResult(success, isFirst)
    if success
        if isFirst
            fprintf('>> Settings saved for next images\n');
        else
            fprintf('>> Applied settings\n');
        end
    else
        fprintf('!! Image skipped\n');
    end
end

%% Polygon Scaling and Bounds Validation
function scaled = scalePolygonsToNewDimensions(polygons, oldSize, newSize)
    % Validate inputs to prevent division by zero
    if length(oldSize) < 2 || length(newSize) < 2
        error('concentration:invalid_size', 'Size arrays must have at least 2 elements');
    end
    if oldSize(1) <= 0 || oldSize(2) <= 0
        error('concentration:invalid_oldsize', 'Old size values must be positive');
    end
    if newSize(1) <= 0 || newSize(2) <= 0
        error('concentration:invalid_newsize', 'New size values must be positive');
    end
    
    scaleX = newSize(1) / oldSize(1);
    scaleY = newSize(2) / oldSize(2);
    scaled = polygons;
    scaled(:,:,1) = polygons(:,:,1) * scaleX;
    scaled(:,:,2) = polygons(:,:,2) * scaleY;
    scaled = constrainPolygonsWithinBounds(scaled, newSize(1), newSize(2));
end

function constrained = constrainPolygonsWithinBounds(polygons, maxW, maxH)
    constrained = polygons;
    constrained(:,:,1) = max(1, min(polygons(:,:,1), maxW));
    constrained(:,:,2) = max(1, min(polygons(:,:,2), maxH));
end



function quad = ensureQuadrilateral(vertices)
    if isempty(vertices)
        quad = [];
        return;
    end
    n = size(vertices,1);
    if n == 4
        quad = vertices;
    elseif n < 4
        quad = repmat(vertices(end,:), 4, 1);  % Pre-allocate with repeated last vertex
        quad(1:n,:) = vertices;                 % Copy original vertices
    else
        quad = vertices(1:4, :);
        warning('concentration:vertex_count', 'Polygon limited to 4 vertices');
    end
end

function enforceQuadrilateral(polygon)
    if ~isvalid(polygon), return; end
    currentPos = polygon.Position;
    if size(currentPos, 1) ~= 4
        polygon.Position = ensureQuadrilateral(currentPos);
    end
end

%% Path and File Utilities
function validatePaths(cfg)
    if ~isfolder(cfg.inputPath)
        error('concentration:input_missing', 'Input directory not found: %s', cfg.inputPath);
    end
    fullOutputPath = fullfile(cfg.projectRoot, cfg.outputPath);
    if ~isfolder(fullOutputPath)
        mkdir(fullOutputPath);
        fprintf('Created output directory: %s\n', fullOutputPath);
    end
end

function outputDir = createOutputDirectory(basePath, phoneName, projectRoot, numConcentrations, concFolderPrefix)
    outputDir = fullfile(projectRoot, basePath, phoneName);
    if ~isfolder(outputDir)
        mkdir(outputDir);
        fprintf('Created output directory: %s\n', outputDir);
    end
    if numConcentrations > 0
        for i = 0:(numConcentrations-1)
            concDir = fullfile(outputDir, sprintf('%s%d', concFolderPrefix, i));
            if ~isfolder(concDir)
                mkdir(concDir);
            end
        end
    end
end

function pathStr = composePathSegments(varargin)
    parts = varargin(~cellfun(@isempty, varargin));
    if isempty(parts)
        pathStr = '';
    else
        pathStr = strjoin(parts, '/');
    end
end

function executeInFolder(folder, func)
    % Execute function in specified folder, restoring original directory afterward
    if isempty(folder)
        folder = '.';
    end
    currentDir = pwd;
    cd(folder);
    cleanup = onCleanup(@() cd(currentDir)); 
    func();
end

function folders = getSubFolders(dirPath)
    items = dir(dirPath);
    isFolder = [items.isdir];
    names = {items(isFolder).name};
    folders = names(~ismember(names, {'.', '..'}));
end

function files = getImageFiles(dirPath, extensions)
    % Collect image files matching any extension pattern

    allFiles = {};
    for i = 1:length(extensions)
        tempFiles = dir(fullfile(dirPath, extensions{i}));
        if ~isempty(tempFiles)
            allFiles = [allFiles, {tempFiles.name}]; %#ok<AGROW>
        end
    end

    if isempty(allFiles)
        files = {};
        return;
    end

    files = sort(unique(allFiles));
end

function saveImageWithFormat(img, outPath, outExt, cfg)
    if any(strcmp(outExt, {'.jpg','.jpeg'}))
        imwrite(img, outPath, 'JPEG', 'Quality', cfg.output.jpegQuality);
    else
        imwrite(img, outPath);
    end
end

function outExt = determineOutputExtension(extOrig, supported, preserveFormat)
    if preserveFormat && any(strcmp(extOrig, supported))
        outExt = extOrig;
    else
        outExt = '.jpeg';
    end
end

function handleError(ME)
    if strcmp(ME.message, 'User stopped execution')
        fprintf('!! Script stopped by user\n');
    else
        fprintf('ERROR: %s\n', ME.message);
        rethrow(ME);
    end
end

%% Dynamic Path Resolution Functions
function cfg = addPathConfiguration(cfg, inputFolder, outputFolder)
    projectRoot = findProjectRoot(inputFolder);
    cfg.projectRoot = projectRoot;
    cfg.inputPath = fullfile(projectRoot, inputFolder);
    cfg.outputPath = outputFolder;
end

function projectRoot = findProjectRoot(inputFolder)
    currentDir = pwd;
    searchDir = currentDir;
    maxLevels = 5;
    
    for level = 1:maxLevels
        [parentDir, ~] = fileparts(searchDir);
        
        % Check if input folder exists at current level
        if isfolder(fullfile(searchDir, inputFolder))
            projectRoot = searchDir;
            return;
        end
        
        % Move up one level
        if strcmp(searchDir, parentDir)
            break; % Reached root directory
        end
        searchDir = parentDir;
    end
    
    % default: use current directory and issue warning
    warning('concentration:path_resolution', ...
            'Could not find input folder "%s". Using current directory as project root.', inputFolder);
    projectRoot = currentDir;
end

function I = imread_raw(fname)
% Read image in on-disk orientation, reversing EXIF 90-degree rotations only

    % Read image (some builds honor AutoOrient=false; some ignore it silently)
    try
        I = imread(fname, 'AutoOrient', false);
    catch
        I = imread(fname);
    end

    % Get EXIF orientation (if present)
    try
        info = imfinfo(fname);
        if ~isfield(info, 'Orientation') || isempty(info.Orientation), return; end
        ori = double(info.Orientation);
    catch
        return; % no EXIF → done
    end

    % Always invert only the 90° EXIF cases
    switch ori
        case 5  % mirror H + rotate -90 (to display upright)
            I = rot90(I, +1); I = fliplr(I);   % invert: +90 then mirror H
        case 6  % rotate +90
            I = rot90(I, -1);                  % invert: -90 (== rot90(...,3))
        case 7  % mirror H + rotate +90
            I = rot90(I, -1); I = fliplr(I);   % invert: -90 then mirror H
        case 8  % rotate +270 (== -90)
            I = rot90(I, +1);                  % invert: +90
        otherwise
            % 1,2,3,4 → leave unchanged (no risk of double-undo)
    end
end



