function cut_concentration_rectangles(varargin)
    %% microPAD Colorimetric Analysis — Concentration Region Cutting Tool
    %% Interactively select and save polygonal concentration regions from strip images
    %% Author: Veysel Y. Yilmaz
    %
    % Inputs (Name-Value pairs):
    % - 'numSquares': number of regions to capture per strip (default: 7)
    % - 'aspectRatio': width/height of the reference strip
    % - 'coverage': fraction of image width to fill (default: 0.995)
    % - 'gapPercent': gap as percent of region width (0..1 or 0..100)
    % - 'inputFolder' | 'outputFolder': override default I/O folders
    % - 'preserveFormat' | 'jpegQuality' | 'saveCoordinates': output behavior
    % - 'useAIDetection': use YOLO for initial polygon placement (default: false)
    % - 'detectionModel': path to YOLOv11 model (default: 'models/yolo11n_micropad_seg.pt')
    % - 'minConfidence': minimum detection confidence (default: 0.6)
    % - 'pythonPath': path to Python executable (default: auto-detected)
    %
    % Outputs/Side effects:
    % - Writes polygon crops to 3_concentration_rectangles/[phone]/con_*/
    % - Writes consolidated coordinates.txt at phone level (atomic, no duplicate rows per image)
    %
    % Behavior:
    % - Shows interactive UI with drawpolygon editing for every image
    % - If useAIDetection=true, attempts AI detection for initial placement
    % - If AI fails or disabled, uses default geometry (aspectRatio, coverage, gapPercent)
    % - User can manually adjust polygons before saving
    % - Cuts N region crops and saves into con_0..con_(N-1) subfolders for each strip
    % - All polygon coordinates written to single phone-level coordinates.txt
    %
    % Examples:
    %   cut_concentration_rectangles('numSquares', 7)
    %   cut_concentration_rectangles('numSquares', 7, 'useAIDetection', true)
    %   cut_concentration_rectangles('useAIDetection', true, 'minConfidence', 0.7)

    %% ========================================================================
    %% EXPERIMENT CONFIGURATION CONSTANTS
    %% ========================================================================
    if mod(length(varargin), 2) ~= 0
        error('concentration:invalid_args', 'Parameters must be provided as name-value pairs');
    end

    % === DATASET AND FOLDER STRUCTURE ===
    INPUT_FOLDER = '2_micropad_papers';
    OUTPUT_FOLDER = '3_concentration_rectangles';

    % === OUTPUT FORMATTING ===
    PRESERVE_FORMAT = true;
    JPEG_QUALITY = 100;
    SAVE_COORDINATES = true;

    % === DEFAULT GEOMETRY / SELECTION ===
    DEFAULT_NUM_SQUARES = 7;
    DEFAULT_ASPECT_RATIO = 7.6;
    DEFAULT_COVERAGE = 0.995;
    DEFAULT_GAP_PERCENT = 0.2;

    % === AI DETECTION DEFAULTS ===
    DEFAULT_USE_AI_DETECTION = true;
    DEFAULT_DETECTION_MODEL = 'models/yolo11n_micropad_seg.pt';
    DEFAULT_MIN_CONFIDENCE = 0.6;
    DEFAULT_PYTHON_PATH = 'C:\Users\veyse\miniconda3\envs\microPAD-python-env\python.exe';

    % === NAMING / FILE CONSTANTS ===
    COORDINATE_FILENAME = 'coordinates.txt';
    SUPPORTED_FORMATS = {'.jpg','.jpeg','.png','.bmp','.tif','.tiff'};
    ALLOWED_IMAGE_EXTENSIONS = {'*.jpg','*.jpeg','*.png','*.bmp','*.tiff','*.tif'};
    CONC_FOLDER_PREFIX = 'con_';

    % === UI CONSTANTS ===
    UI_CONST = struct();
    UI_CONST.fontSize = struct(...
        'title', 16, ...
        'path', 12, ...
        'button', 12, ...
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
        'cutButtonPanel', [0.55 0.02 0.43 0.10], ...
        'previewPanel', [0.25 0.02 0.50 0.10], ...
        'stopButton', [0.02 0.93 0.06 0.05], ...
        'instructions', [0.02 0.125 0.96 0.025], ...
        'previewLeft', [0.02 0.16 0.47 0.72], ...
        'previewRight', [0.51 0.16 0.47 0.72]);
    UI_CONST.polygon = struct(...
        'lineWidth', 3, ...
        'borderWidth', 2);
    UI_CONST.dimFactor = 0.3;

    %% Build configuration
    cfg = createConfiguration(INPUT_FOLDER, OUTPUT_FOLDER, PRESERVE_FORMAT, JPEG_QUALITY, SAVE_COORDINATES, ...
                              DEFAULT_NUM_SQUARES, DEFAULT_ASPECT_RATIO, DEFAULT_COVERAGE, DEFAULT_GAP_PERCENT, ...
                              DEFAULT_USE_AI_DETECTION, DEFAULT_DETECTION_MODEL, DEFAULT_MIN_CONFIDENCE, DEFAULT_PYTHON_PATH, ...
                              COORDINATE_FILENAME, SUPPORTED_FORMATS, ALLOWED_IMAGE_EXTENSIONS, CONC_FOLDER_PREFIX, UI_CONST, varargin{:});

    try
        processAllFolders(cfg);
        fprintf('>> Concentration region cutting completed successfully!\n');
    catch ME
        handleError(ME);
    end
end

%% -------------------------------------------------------------------------
%% Configuration
%% -------------------------------------------------------------------------

function cfg = createConfiguration(inputFolder, outputFolder, preserveFormat, jpegQuality, saveCoordinates, ...
                                   defaultNumSquares, defaultAspectRatio, defaultCoverage, defaultGapPercent, ...
                                   defaultUseAI, defaultDetectionModel, defaultMinConfidence, defaultPythonPath, ...
                                   coordinateFileName, supportedFormats, allowedImageExtensions, concFolderPrefix, UI_CONST, varargin)
    parser = inputParser;
    parser.addParameter('numSquares', defaultNumSquares, @(x) validateattributes(x, {'numeric'}, {'scalar','integer','>=',1,'<=',20}));

    validateFolder = @(s) validateattributes(s, {'char', 'string'}, {'nonempty', 'scalartext'});
    parser.addParameter('inputFolder', inputFolder, validateFolder);
    parser.addParameter('outputFolder', outputFolder, validateFolder);
    parser.addParameter('preserveFormat', preserveFormat, @(x) islogical(x));
    parser.addParameter('jpegQuality', jpegQuality, @(x) validateattributes(x, {'numeric'}, {'scalar','>=',1,'<=',100}));
    parser.addParameter('saveCoordinates', saveCoordinates, @(x) islogical(x));

    parser.addParameter('aspectRatio', defaultAspectRatio, @(x) validateattributes(x, {'numeric'}, {'scalar','>',0}));
    parser.addParameter('coverage', defaultCoverage, @(x) validateattributes(x, {'numeric'}, {'scalar','>',0,'<=',1}));
    parser.addParameter('gapPercent', defaultGapPercent, @(x) isnumeric(x) && isscalar(x) && x>=0);

    parser.addParameter('useAIDetection', defaultUseAI, @islogical);
    parser.addParameter('detectionModel', defaultDetectionModel, @(x) validateattributes(x, {'char', 'string'}, {'nonempty', 'scalartext'}));
    parser.addParameter('minConfidence', defaultMinConfidence, @(x) validateattributes(x, {'numeric'}, {'scalar', '>=', 0, '<=', 1}));
    parser.addParameter('pythonPath', defaultPythonPath, @(x) validateattributes(x, {'char', 'string'}, {'nonempty', 'scalartext'}));

    parser.parse(varargin{:});

    cfg.numSquares = parser.Results.numSquares;

    if cfg.numSquares > 15
        warning('concentration:many_squares', 'Large numSquares (%d) may cause UI layout issues and small regions', cfg.numSquares);
    end

    % Store model path (relative), will be resolved in addPathConfiguration
    cfg.useAIDetection = parser.Results.useAIDetection;
    cfg.detectionModelRelative = parser.Results.detectionModel;
    cfg.minConfidence = parser.Results.minConfidence;
    cfg.pythonPath = parser.Results.pythonPath;

    cfg = addPathConfiguration(cfg, parser.Results.inputFolder, parser.Results.outputFolder);

    cfg.output.preserveFormat = parser.Results.preserveFormat;
    cfg.output.jpegQuality = parser.Results.jpegQuality;
    cfg.output.saveCoordinates = parser.Results.saveCoordinates;
    cfg.output.supportedFormats = supportedFormats;
    cfg.allowedImageExtensions = allowedImageExtensions;

    cfg.coordinateFileName = coordinateFileName;
    cfg.concFolderPrefix = concFolderPrefix;

    % Geometry configuration
    cfg.geometry = struct();
    cfg.geometry.aspectRatio = parser.Results.aspectRatio;
    gp = parser.Results.gapPercent;
    if gp > 1
        gp = gp / 100;
    end
    cfg.geometry.gapPercentWidth = gp;
    cfg.coverage = parser.Results.coverage;

    % UI configuration
    cfg.ui.fontSize = UI_CONST.fontSize;
    cfg.ui.colors = UI_CONST.colors;
    cfg.ui.positions = UI_CONST.positions;
    cfg.ui.polygon = UI_CONST.polygon;
    cfg.dimFactor = UI_CONST.dimFactor;
end

function cfg = addPathConfiguration(cfg, inputFolder, outputFolder)
    projectRoot = find_project_root(inputFolder);

    cfg.projectRoot = projectRoot;
    cfg.inputPath = fullfile(projectRoot, inputFolder);
    cfg.outputPath = fullfile(projectRoot, outputFolder);

    % Resolve model path to absolute path
    cfg.detectionModel = fullfile(projectRoot, cfg.detectionModelRelative);

    validatePaths(cfg);
end

function projectRoot = find_project_root(inputFolder)
    searchPath = pwd;
    maxLevels = 5;

    for level = 1:maxLevels
        candidatePath = fullfile(searchPath, inputFolder);
        if isfolder(candidatePath)
            projectRoot = searchPath;
            return;
        end
        parentPath = fileparts(searchPath);
        if strcmp(parentPath, searchPath)
            break;
        end
        searchPath = parentPath;
    end

    warning('concentration:no_input_folder', ...
        'Could not find input folder "%s" within %d directory levels. Using current directory as project root.', ...
        inputFolder, maxLevels);
    projectRoot = pwd;
end

function validatePaths(cfg)
    if ~isfolder(cfg.inputPath)
        error('concentration:missing_input', 'Input folder not found: %s', cfg.inputPath);
    end
    if ~isfolder(cfg.outputPath)
        mkdir(cfg.outputPath);
    end
end

%% -------------------------------------------------------------------------
%% Main Processing Loop
%% -------------------------------------------------------------------------

function processAllFolders(cfg)
    fprintf('\n=== Starting Concentration Region Cutting ===\n');
    fprintf('Input: %s\n', cfg.inputPath);
    fprintf('Output: %s\n', cfg.outputPath);
    fprintf('AI Detection: %s\n', string(cfg.useAIDetection));
    if cfg.useAIDetection
        fprintf('Detection model: %s\n', cfg.detectionModel);
        fprintf('Min confidence: %.2f\n', cfg.minConfidence);
    end
    fprintf('Regions per strip: %d\n\n', cfg.numSquares);

    executeInFolder(cfg.inputPath, @() processPhones(cfg));
end

function processPhones(cfg)
    phoneFolders = getSubFolders('.');
    if isempty(phoneFolders)
        warning('concentration:no_phones', 'No phone folders found in input directory');
        return;
    end

    for i = 1:numel(phoneFolders)
        processPhone(phoneFolders{i}, cfg);
    end
end

function processPhone(phoneName, cfg)
    fprintf('\n=== Processing Phone: %s ===\n', phoneName);
    executeInFolder(phoneName, @() processImagesInPhone(phoneName, cfg));
end

function processImagesInPhone(phoneName, cfg)
    imageList = getImageFiles('.', cfg.allowedImageExtensions);
    if isempty(imageList)
        warning('concentration:no_images', 'No images found for phone folder: %s', phoneName);
        return;
    end

    fprintf('Found %d images\n', numel(imageList));

    outputDir = createOutputDirectory(cfg.outputPath, phoneName, cfg.numSquares, cfg.concFolderPrefix);

    % Setup Python environment once per phone if AI detection is enabled
    if cfg.useAIDetection
        ensurePythonSetup(cfg.pythonPath);
        validateModelFile(cfg.detectionModel);
        loadYOLOModel(cfg.detectionModel);
    end

    persistentFig = [];

    try
        for idx = 1:numel(imageList)
            if ~isempty(persistentFig) && ~isvalid(persistentFig)
                persistentFig = [];
            end
            [success, persistentFig] = processOneImage(imageList{idx}, outputDir, cfg, persistentFig, phoneName);
            if success
                fprintf('  >> Saved %d concentration regions\n', cfg.numSquares);
            else
                fprintf('  Image skipped by user\n');
            end
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

    fprintf('Completed: %s\n', phoneName);
end

function [success, fig] = processOneImage(imageName, outputDir, cfg, fig, phoneName)
    success = false;

    fprintf('  -> Processing: %s\n', imageName);

    [img, isValid] = loadImage(imageName);
    if ~isValid
        fprintf('  !! Failed to load image\n');
        return;
    end

    % Get initial polygon positions (AI detection or default geometry)
    initialPolygons = getInitialPolygons(img, cfg);

    % Interactive region selection with persistent window
    [polygonParams, fig] = showInteractiveGUI(img, imageName, phoneName, cfg, initialPolygons, fig);

    if ~isempty(polygonParams)
        saveCroppedRegions(img, imageName, polygonParams, outputDir, cfg);
        success = true;
    end
end

function initialPolygons = getInitialPolygons(img, cfg)
    % Get initial polygon positions using AI detection (if enabled) or default geometry
    if cfg.useAIDetection
        try
            [detectedQuads, confidences] = detectQuadsYOLO(img, cfg.minConfidence);

            if ~isempty(detectedQuads) && size(detectedQuads, 1) == cfg.numSquares
                fprintf('  AI detected %d regions (avg confidence: %.2f)\n', ...
                    size(detectedQuads, 1), mean(confidences));
                initialPolygons = detectedQuads;
                return;
            elseif ~isempty(detectedQuads)
                fprintf('  AI detected %d regions but expected %d, using default geometry\n', ...
                    size(detectedQuads, 1), cfg.numSquares);
            else
                fprintf('  No AI detections, using default geometry\n');
            end
        catch ME
            fprintf('  AI detection failed (%s), using default geometry\n', ME.message);
        end
    end

    % Fallback to default geometry
    [imageHeight, imageWidth, ~] = size(img);
    initialPolygons = calculateDefaultPolygons(imageWidth, imageHeight, cfg);
end

function polygons = calculateDefaultPolygons(imageWidth, imageHeight, cfg)
    % Generate default polygon positions using geometry parameters
    n = cfg.numSquares;

    % Build world coordinates
    aspect = cfg.geometry.aspectRatio;
    aspect = max(aspect, eps);
    totalGridWidth = 1.0;
    rectHeightWorld = 1.0 / aspect;

    % Compute gap size
    gp = cfg.geometry.gapPercentWidth;
    denom = n + max(n-1, 0) * gp;
    if denom <= 0
        denom = max(n, 1);
    end
    w = totalGridWidth / denom;
    gapSizeWorld = gp * w;

    % Build world corners
    worldCorners = zeros(n, 4, 2);
    xi = -totalGridWidth / 2;
    yi = -rectHeightWorld / 2;

    for i = 1:n
        worldCorners(i, :, :) = [
            xi,       yi;
            xi + w,   yi;
            xi + w,   yi + rectHeightWorld;
            xi,       yi + rectHeightWorld
        ];
        xi = xi + w + gapSizeWorld;
    end

    % Scale and center to image
    polygons = scaleAndCenterPolygons(worldCorners, imageWidth, imageHeight, cfg);
end

function polygons = scaleAndCenterPolygons(worldCorners, imageWidth, imageHeight, cfg)
    % Scale world coordinates to fit image with coverage factor
    n = size(worldCorners, 1);
    polygons = zeros(n, 4, 2);

    % Find bounding box of all world corners (width only needed for scaling)
    allX = worldCorners(:, :, 1);
    minX = min(allX(:));
    maxX = max(allX(:));

    worldW = maxX - minX;

    % Scale to fit image width with coverage factor
    targetWidth = imageWidth * cfg.coverage;
    scale = targetWidth / worldW;

    % Center in image
    centerX = imageWidth / 2;
    centerY = imageHeight / 2;

    for i = 1:n
        corners = squeeze(worldCorners(i, :, :));
        scaled = corners * scale;
        scaled(:, 1) = scaled(:, 1) + centerX;
        scaled(:, 2) = scaled(:, 2) + centerY;
        polygons(i, :, :) = scaled;
    end
end

%% -------------------------------------------------------------------------
%% Interactive UI
%% -------------------------------------------------------------------------

function [polygonParams, fig] = showInteractiveGUI(img, imageName, phoneName, cfg, initialPolygons, fig)
    % Show interactive GUI with editing and preview modes
    polygonParams = [];

    % Create figure if needed
    if isempty(fig) || ~isvalid(fig)
        fig = createFigure(imageName, phoneName, cfg);
    end

    while true
        % Editing mode
        clearAndRebuildUI(fig, 'editing', img, imageName, phoneName, cfg, initialPolygons);

        [action, userPolygons] = waitForUserAction(fig);

        switch action
            case 'skip'
                return;
            case 'stop'
                close(fig);
                error('User stopped execution');
            case 'accept'
                % Preview mode
                clearAndRebuildUI(fig, 'preview', img, imageName, phoneName, cfg, userPolygons);

                [prevAction, ~] = waitForUserAction(fig);

                switch prevAction
                    case 'accept'
                        polygonParams = userPolygons;
                        return;
                    case {'skip', 'stop'}
                        if strcmp(prevAction, 'stop')
                            close(fig);
                            error('User stopped execution');
                        end
                        return;
                    case 'retry'
                        % Use edited polygons as new initial positions
                        initialPolygons = userPolygons;
                        continue;
                end
        end
    end
end

function clearAndRebuildUI(fig, mode, img, imageName, phoneName, cfg, polygonParams)
    % Modes: 'editing' (interactive polygon adjustment), 'preview' (final confirmation)

    guiData = get(fig, 'UserData');
    clearAllUIElements(fig, guiData);

    switch mode
        case 'editing'
            buildEditingUI(fig, img, imageName, phoneName, cfg, polygonParams);
        case 'preview'
            buildPreviewUI(fig, img, imageName, phoneName, cfg, polygonParams);
    end
end

function clearAllUIElements(fig, guiData)
    % Delete all UI elements
    allObjects = findall(fig);
    if isempty(allObjects)
        set(fig, 'UserData', []);
        return;
    end

    objTypes = get(allObjects, 'Type');
    if ~iscell(objTypes), objTypes = {objTypes}; end

    isControl = strcmp(objTypes, 'uicontrol');
    isPanel = strcmp(objTypes, 'uipanel');
    isAxes = strcmp(objTypes, 'axes');

    toDelete = allObjects(isControl | isPanel | isAxes);

    % Add polygon ROIs from guiData
    if ~isempty(guiData) && isstruct(guiData) && isfield(guiData, 'polygons')
        validPolys = collectValidPolygons(guiData);
        if ~isempty(validPolys)
            toDelete = [toDelete; validPolys];
        end
    end

    % Bulk delete
    if ~isempty(toDelete)
        validMask = arrayfun(@isvalid, toDelete);
        delete(toDelete(validMask));
    end

    % Cleanup remaining ROIs
    rois = findobj(fig, '-isa', 'images.roi.Polygon');
    if ~isempty(rois)
        validRois = rois(arrayfun(@isvalid, rois));
        if ~isempty(validRois)
            delete(validRois);
        end
    end

    set(fig, 'UserData', []);
end

function polys = collectValidPolygons(guiData)
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

function buildEditingUI(fig, img, imageName, phoneName, cfg, initialPolygons)
    % Build UI for polygon editing mode
    set(fig, 'Name', sprintf('Concentration Region Cutter - %s - %s', phoneName, imageName));

    guiData = struct();
    guiData.mode = 'editing';

    % Title and path
    guiData.titleHandle = createTitle(fig, phoneName, imageName, cfg);
    guiData.pathHandle = createPathDisplay(fig, phoneName, imageName, cfg);

    % Image display
    guiData.imgAxes = createImageAxes(fig, img, cfg);

    % Create editable polygons
    guiData.polygons = createPolygons(initialPolygons, cfg);

    % Buttons
    guiData.cutButtonPanel = createEditButtonPanel(fig, cfg);
    guiData.stopButton = createStopButton(fig, cfg);
    guiData.instructionText = createInstructions(fig, cfg);

    guiData.action = '';
    set(fig, 'UserData', guiData);
end

function buildPreviewUI(fig, img, imageName, phoneName, cfg, polygonParams)
    % Build UI for preview mode
    set(fig, 'Name', sprintf('PREVIEW - %s - %s', phoneName, imageName));

    guiData = struct();
    guiData.mode = 'preview';
    guiData.savedPolygonParams = polygonParams;

    % Title and path
    titleText = sprintf('PREVIEW: %s - %s', phoneName, imageName);
    guiData.titleHandle = uicontrol('Parent', fig, 'Style', 'text', 'String', titleText, ...
                                   'Units', 'normalized', 'Position', cfg.ui.positions.title, ...
                                   'FontSize', cfg.ui.fontSize.title, 'FontWeight', 'bold', ...
                                   'ForegroundColor', cfg.ui.colors.foreground, ...
                                   'BackgroundColor', cfg.ui.colors.background, ...
                                   'HorizontalAlignment', 'center');

    pathText = sprintf('PREVIEW - Path: %s | Image: %s', phoneName, imageName);
    guiData.pathHandle = uicontrol('Parent', fig, 'Style', 'text', 'String', pathText, ...
                                  'Units', 'normalized', 'Position', cfg.ui.positions.pathDisplay, ...
                                  'FontSize', cfg.ui.fontSize.path, 'FontWeight', 'normal', ...
                                  'ForegroundColor', cfg.ui.colors.path, ...
                                  'BackgroundColor', cfg.ui.colors.background, ...
                                  'HorizontalAlignment', 'center');

    % Preview axes
    [guiData.leftAxes, guiData.rightAxes] = createPreviewAxes(fig, img, polygonParams, cfg);

    % Buttons
    guiData.stopButton = createStopButton(fig, cfg);
    guiData.buttonPanel = createPreviewButtons(fig, cfg);

    guiData.action = '';
    set(fig, 'UserData', guiData);
end

%% -------------------------------------------------------------------------
%% UI Components
%% -------------------------------------------------------------------------

function fig = createFigure(imageName, phoneName, cfg)
    titleText = sprintf('Concentration Region Cutter - %s - %s', phoneName, imageName);
    fig = figure('Name', titleText, ...
                'Units', 'normalized', 'Position', cfg.ui.positions.figure, ...
                'MenuBar', 'none', 'ToolBar', 'none', ...
                'Color', cfg.ui.colors.background, 'KeyPressFcn', @keyPressHandler);

    drawnow limitrate;
    pause(0.05);
    set(fig, 'WindowState', 'maximized');
    figure(fig);
    drawnow limitrate;
end

function titleHandle = createTitle(fig, phoneName, imageName, cfg)
    titleText = sprintf('%s - %s', phoneName, imageName);
    titleHandle = uicontrol('Parent', fig, 'Style', 'text', 'String', titleText, ...
                           'Units', 'normalized', 'Position', cfg.ui.positions.title, ...
                           'FontSize', cfg.ui.fontSize.title, 'FontWeight', 'bold', ...
                           'ForegroundColor', cfg.ui.colors.foreground, ...
                           'BackgroundColor', cfg.ui.colors.background, ...
                           'HorizontalAlignment', 'center');
end

function pathHandle = createPathDisplay(fig, phoneName, imageName, cfg)
    pathText = sprintf('Path: %s | Image: %s', phoneName, imageName);
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

function polygons = createPolygons(initialPolygons, cfg)
    % Create drawpolygon objects from initial positions
    n = size(initialPolygons, 1);
    polygons = cell(1, n);

    for i = 1:n
        pos = squeeze(initialPolygons(i, :, :));
        polygons{i} = drawpolygon('Position', pos, ...
                                 'Color', cfg.ui.colors.polygon, ...
                                 'LineWidth', cfg.ui.polygon.lineWidth, ...
                                 'MarkerSize', 8, ...
                                 'Selected', false);

        % Add listener for quadrilateral enforcement
        addlistener(polygons{i}, 'ROIMoved', @(~,~) enforceQuadrilateral(polygons{i}));
    end
end

function enforceQuadrilateral(polygon)
    % Ensure polygon remains a quadrilateral
    if ~isvalid(polygon)
        return;
    end

    pos = polygon.Position;
    if size(pos, 1) ~= 4
        warning('concentration:invalid_polygon', 'Polygon must have exactly 4 vertices');
    end
end

function cutButtonPanel = createEditButtonPanel(fig, cfg)
    cutButtonPanel = uipanel('Parent', fig, 'Units', 'normalized', ...
                             'Position', cfg.ui.positions.cutButtonPanel, ...
                             'BackgroundColor', cfg.ui.colors.panel, ...
                             'BorderType', 'etchedin', 'HighlightColor', cfg.ui.colors.foreground);

    % APPLY button
    uicontrol('Parent', cutButtonPanel, 'Style', 'pushbutton', ...
             'String', 'APPLY', 'FontSize', cfg.ui.fontSize.button, 'FontWeight', 'bold', ...
             'Units', 'normalized', 'Position', [0.15 0.35 0.30 0.35], ...
             'BackgroundColor', cfg.ui.colors.apply, 'ForegroundColor', cfg.ui.colors.foreground, ...
             'Callback', @(~,~) setAction(fig, 'accept'));

    % SKIP button
    uicontrol('Parent', cutButtonPanel, 'Style', 'pushbutton', ...
             'String', 'SKIP', 'FontSize', cfg.ui.fontSize.button, 'FontWeight', 'bold', ...
             'Units', 'normalized', 'Position', [0.55 0.35 0.30 0.35], ...
             'BackgroundColor', cfg.ui.colors.skip, 'ForegroundColor', cfg.ui.colors.foreground, ...
             'Callback', @(~,~) setAction(fig, 'skip'));
end

function instructionText = createInstructions(fig, cfg)
    instructionString = 'Mouse = Drag Vertices | APPLY = Save & Continue | SKIP = Skip | STOP = Exit | Space = APPLY | Esc = SKIP';

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

    buttons = {'ACCEPT', 'RETRY', 'SKIP'};
    positions = {[0.05 0.25 0.25 0.50], [0.375 0.25 0.25 0.50], [0.70 0.25 0.25 0.50]};
    colors = {cfg.ui.colors.accept, cfg.ui.colors.retry, cfg.ui.colors.skip};
    actions = {'accept', 'retry', 'skip'};

    for i = 1:numel(buttons)
        uicontrol('Parent', buttonPanel, 'Style', 'pushbutton', ...
                 'String', buttons{i}, 'FontSize', cfg.ui.fontSize.preview, 'FontWeight', 'bold', ...
                 'Units', 'normalized', 'Position', positions{i}, ...
                 'BackgroundColor', colors{i}, 'ForegroundColor', cfg.ui.colors.foreground, ...
                 'Callback', @(~,~) setAction(fig, actions{i}));
    end
end

function [leftAxes, rightAxes] = createPreviewAxes(fig, img, polygonParams, cfg)
    % Left: original with overlays
    leftAxes = axes('Parent', fig, 'Units', 'normalized', 'Position', cfg.ui.positions.previewLeft);
    imshow(img, 'Parent', leftAxes, 'InitialMagnification', 'fit');
    axis(leftAxes, 'image');
    axis(leftAxes, 'tight');
    title(leftAxes, sprintf('Original with %d Concentration Regions', size(polygonParams, 1)), ...
          'Color', cfg.ui.colors.foreground, 'FontSize', cfg.ui.fontSize.button, 'FontWeight', 'bold');
    hold(leftAxes, 'on');

    % Draw polygon overlays
    for i = 1:size(polygonParams, 1)
        poly = squeeze(polygonParams(i,:,:));
        if size(poly, 1) >= 3
            plot(leftAxes, [poly(:,1); poly(1,1)], [poly(:,2); poly(1,2)], ...
                 'Color', cfg.ui.colors.polygon, 'LineWidth', cfg.ui.polygon.lineWidth);

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

    % Right: highlighted regions
    rightAxes = axes('Parent', fig, 'Units', 'normalized', 'Position', cfg.ui.positions.previewRight);
    maskedImg = createMaskedPreview(img, polygonParams, cfg);
    imshow(maskedImg, 'Parent', rightAxes, 'InitialMagnification', 'fit');
    axis(rightAxes, 'image');
    axis(rightAxes, 'tight');
    title(rightAxes, 'Highlighted Concentration Regions', ...
          'Color', cfg.ui.colors.foreground, 'FontSize', cfg.ui.fontSize.button, 'FontWeight', 'bold');
end

function maskedImg = createMaskedPreview(img, polygonParams, cfg)
    [height, width, ~] = size(img);
    totalMask = false(height, width);

    numRegions = size(polygonParams, 1);
    for i = 1:numRegions
        poly = squeeze(polygonParams(i,:,:));
        if size(poly, 1) >= 3
            regionMask = poly2mask(poly(:,1), poly(:,2), height, width);
            totalMask = totalMask | regionMask;
        end
    end

    dimFactor = cfg.dimFactor;
    maskedImg = double(img);
    dimMultiplier = double(totalMask) + (1 - double(totalMask)) * dimFactor;
    maskedImg = maskedImg .* dimMultiplier;
    maskedImg = uint8(maskedImg);
end

%% -------------------------------------------------------------------------
%% User Interaction
%% -------------------------------------------------------------------------

function setAction(fig, action)
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
    switch event.Key
        case 'space'
            setAction(src, 'accept');
        case 'escape'
            setAction(src, 'skip');
    end
end

function [action, polygonParams] = waitForUserAction(fig)
    uiwait(fig);

    action = '';
    polygonParams = [];

    if isvalid(fig)
        guiData = get(fig, 'UserData');
        action = guiData.action;

        if strcmp(action, 'accept')
            if strcmp(guiData.mode, 'preview')
                polygonParams = guiData.savedPolygonParams;
            elseif strcmp(guiData.mode, 'editing')
                polygonParams = extractPolygonParameters(guiData);
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

%% -------------------------------------------------------------------------
%% Image Cropping and Coordinate Saving
%% -------------------------------------------------------------------------

function saveCroppedRegions(img, imageName, polygons, outputDir, cfg)
    [~, baseName, ~] = fileparts(imageName);
    [~, extOrig] = fileparts(imageName);
    outExt = determineOutputExtension(extOrig, cfg.output.supportedFormats, cfg.output.preserveFormat);

    numRegions = size(polygons, 1);

    for concentration = 0:(numRegions - 1)
        polygon = squeeze(polygons(concentration + 1, :, :));

        croppedImg = cropImageWithPolygon(img, polygon);

        concFolder = sprintf('%s%d', cfg.concFolderPrefix, concentration);
        concPath = fullfile(outputDir, concFolder);

        outputName = sprintf('%s_con_%d%s', baseName, concentration, outExt);
        outputPath = fullfile(concPath, outputName);

        saveImageWithFormat(croppedImg, outputPath, outExt, cfg);

        if cfg.output.saveCoordinates
            appendPolygonCoordinates(outputDir, baseName, concentration, polygon, cfg);
        end
    end
end

function croppedImg = cropImageWithPolygon(img, polygonVertices)
    x = polygonVertices(:, 1);
    y = polygonVertices(:, 2);

    minX = floor(min(x));
    maxX = ceil(max(x));
    minY = floor(min(y));
    maxY = ceil(max(y));

    [imgH, imgW, ~] = size(img);
    minX = max(1, minX);
    maxX = min(imgW, maxX);
    minY = max(1, minY);
    maxY = min(imgH, maxY);

    croppedImg = img(minY:maxY, minX:maxX, :);
end

function appendPolygonCoordinates(phoneOutputDir, baseName, concentration, polygon, cfg)
    coordPath = fullfile(phoneOutputDir, cfg.coordinateFileName);

    scanFmt = '%s %d %f %f %f %f %f %f %f %f';
    writeFmt = '%s %d %.6f %.6f %.6f %.6f %.6f %.6f %.6f %.6f\n';
    numericCount = 9;

    [existingNames, existingNums] = readExistingCoordinates(coordPath, scanFmt, numericCount);

    [filteredNames, filteredNums] = filterConflictingEntries(existingNames, existingNums, baseName, concentration);

    newRow = [concentration, polygon(1,:), polygon(2,:), polygon(3,:), polygon(4,:)];
    filteredNames{end+1} = baseName;
    filteredNums = [filteredNums; newRow];

    header = 'image concentration x1 y1 x2 y2 x3 y3 x4 y4';
    atomicWriteCoordinates(coordPath, header, filteredNames, filteredNums, writeFmt, phoneOutputDir);
end

function [existingNames, existingNums] = readExistingCoordinates(coordPath, scanFmt, numericCount)
    existingNames = {};
    existingNums = [];

    if ~isfile(coordPath)
        return;
    end

    fid = fopen(coordPath, 'rt');
    if fid == -1
        warning('concentration:coord_read', 'Could not open coordinates file: %s', coordPath);
        return;
    end

    headerLine = fgetl(fid);
    if headerLine == -1
        fclose(fid);
        return;
    end

    rowData = textscan(fid, scanFmt, 'Delimiter', ' ', 'MultipleDelimsAsOne', true);
    fclose(fid);

    if isempty(rowData) || isempty(rowData{1})
        return;
    end

    existingNames = rowData{1};
    existingNums = zeros(numel(existingNames), numericCount);
    for i = 1:numericCount
        existingNums(:, i) = rowData{i+1};
    end
end

function [filteredNames, filteredNums] = filterConflictingEntries(existingNames, existingNums, newName, concentration)
    if isempty(existingNames)
        filteredNames = {};
        filteredNums = [];
        return;
    end

    matchMask = strcmp(existingNames, newName) & (existingNums(:, 1) == concentration);

    filteredNames = existingNames(~matchMask);
    filteredNums = existingNums(~matchMask, :);
end

function atomicWriteCoordinates(coordPath, header, names, nums, writeFmt, coordFolder)
    tmpPath = tempname(coordFolder);

    fid = fopen(tmpPath, 'wt');
    if fid == -1
        error('concentration:coord_write', 'Cannot open temp file for coordinates: %s', tmpPath);
    end

    fprintf(fid, '%s\n', header);

    for i = 1:numel(names)
        fprintf(fid, writeFmt, names{i}, nums(i, :));
    end

    fclose(fid);

    [ok, msg, msgid] = movefile(tmpPath, coordPath, 'f');
    if ~ok
        error('concentration:coord_move', 'Failed to move temp coordinate file: %s (%s)', msg, msgid);
    end
end

%% -------------------------------------------------------------------------
%% File I/O Utilities
%% -------------------------------------------------------------------------

function [img, isValid] = loadImage(imageName)
    isValid = false;
    img = [];

    if ~isfile(imageName)
        warning('concentration:missing_file', 'Image file not found: %s', imageName);
        return;
    end

    try
        img = imread_raw(imageName);
        isValid = true;
    catch ME
        warning('concentration:read_error', 'Failed to read image %s: %s', imageName, ME.message);
    end
end

function I = imread_raw(fname)
    try
        I = imread(fname, 'AutoOrient', false);
    catch
        I = imread(fname);
    end

    try
        info = imfinfo(fname);
        if ~isfield(info, 'Orientation'), return; end
        ori = double(info.Orientation);
    catch
        return;
    end

    % Invert 90-degree EXIF rotations to preserve raw sensor layout
    switch ori
        case 5, I = rot90(I, +1); I = fliplr(I);
        case 6, I = rot90(I, -1);
        case 7, I = rot90(I, -1); I = fliplr(I);
        case 8, I = rot90(I, +1);
    end
end

function saveImageWithFormat(img, outPath, outExt, cfg)
    if strcmpi(outExt, '.jpg') || strcmpi(outExt, '.jpeg')
        imwrite(img, outPath, 'jpg', 'Quality', cfg.output.jpegQuality);
    else
        imwrite(img, outPath);
    end
end

function outExt = determineOutputExtension(extOrig, supported, preserveFormat)
    if preserveFormat && any(strcmpi(extOrig, supported))
        outExt = lower(extOrig);
    else
        outExt = '.jpg';
    end
end

function outputDir = createOutputDirectory(basePath, phoneName, numConcentrations, concFolderPrefix)
    phoneOutputDir = fullfile(basePath, phoneName);
    if ~isfolder(phoneOutputDir)
        mkdir(phoneOutputDir);
    end

    for i = 0:(numConcentrations - 1)
        concFolder = sprintf('%s%d', concFolderPrefix, i);
        concPath = fullfile(phoneOutputDir, concFolder);
        if ~isfolder(concPath)
            mkdir(concPath);
        end
    end

    outputDir = phoneOutputDir;
end

function folders = getSubFolders(dirPath)
    items = dir(dirPath);
    folders = {items([items.isdir]).name};
    folders = folders(~ismember(folders, {'.', '..'}));
end

function files = getImageFiles(dirPath, extensions)
    % Preallocate cell array to avoid growing in loop
    maxFilesPerExt = 1000;  % Reasonable upper bound
    allFiles = cell(numel(extensions) * maxFilesPerExt, 1);
    fileCount = 0;

    for i = 1:numel(extensions)
        foundFiles = dir(fullfile(dirPath, extensions{i}));
        if ~isempty(foundFiles)
            numFound = numel(foundFiles);
            allFiles(fileCount+1:fileCount+numFound) = {foundFiles.name}';
            fileCount = fileCount + numFound;
        end
    end

    % Trim to actual size and get unique files
    allFiles = allFiles(1:fileCount);
    files = unique(allFiles);
end

function executeInFolder(folder, func)
    origDir = pwd;
    cleanupObj = onCleanup(@() cd(origDir));
    cd(folder);
    func();
end

%% -------------------------------------------------------------------------
%% YOLO Auto-Detection Integration
%% -------------------------------------------------------------------------

function ensurePythonSetup(pythonPath)
    persistent setupComplete
    if ~isempty(setupComplete) && setupComplete
        return;
    end

    % Check environment variable first
    envPath = getenv('MICROPAD_PYTHON');
    if ~isempty(envPath)
        pythonPath = envPath;
    end

    if ~isfile(pythonPath)
        error('concentration:python_missing', ...
            ['Python executable not found at: %s\n', ...
             'Resolution options:\n', ...
             '1. Install miniconda and create microPAD-python-env environment\n', ...
             '2. Set MICROPAD_PYTHON environment variable to your Python executable\n', ...
             '3. Pass pythonPath parameter: cut_concentration_rectangles(''pythonPath'', ''path/to/python'')'], ...
            pythonPath);
    end

    currentPython = pyenv;
    if strcmp(currentPython.Version, '')
        fprintf('Configuring Python environment for auto-detection...\n');
        pyenv('Version', pythonPath);
        fprintf('[OK] Python environment configured\n');
    elseif ~strcmp(currentPython.Executable, pythonPath)
        warning('concentration:python_mismatch', ...
            'Python already initialized with different executable:\n  Current: %s\n  Expected: %s\nRestart MATLAB to switch environments.', ...
            currentPython.Executable, pythonPath);
    end

    try
        py.importlib.import_module('ultralytics');
    catch
        error('concentration:ultralytics_missing', ...
            'Ultralytics package not found in Python environment.\nRun: pip install ultralytics');
    end

    try
        py.importlib.import_module('cv2');
    catch
        error('concentration:opencv_missing', ...
            'OpenCV package not found in Python environment.\nRun: pip install opencv-python');
    end

    fprintf('[OK] Setup complete - ready for auto-detection\n\n');
    setupComplete = true;
end

function validateModelFile(modelPath)
    if ~isfile(modelPath)
        error('concentration:model_missing', ...
            'YOLO model not found: %s\nEnsure model file exists at specified path.', ...
            modelPath);
    end
end

function loadYOLOModel(modelPath)
    [currentModel, currentPath] = modelCache();
    if ~isempty(currentModel) && ~isempty(currentPath) && strcmp(currentPath, modelPath)
        return;
    end

    fprintf('Loading YOLO model: %s\n', modelPath);

    YOLO = py.getattr(py.importlib.import_module('ultralytics'), 'YOLO');
    newModel = YOLO(modelPath);
    modelCache(newModel, modelPath);

    fprintf('[OK] YOLO model loaded and cached\n');
end

function model = getYOLOModel()
    model = modelCache();

    if isempty(model)
        error('concentration:model_not_loaded', ...
            'YOLO model not loaded. Call loadYOLOModel() first.');
    end
end

function [model, modelPath] = modelCache(newModel, newPath)
    persistent cachedModel
    persistent cachedModelPath

    if nargin == 0
        model = cachedModel;
        modelPath = cachedModelPath;
    elseif nargin == 2
        cachedModel = newModel;
        cachedModelPath = newPath;
        model = cachedModel;
        modelPath = cachedModelPath;
    else
        error('concentration:invalid_cache_args', 'modelCache requires 0 or 2 arguments');
    end
end

function [quads, confidences] = detectQuadsYOLO(img, confThreshold)
    INFERENCE_SIZE = int32(640);

    [imgH, imgW, ~] = size(img);

    pyImg = py.numpy.array(img);

    model = getYOLOModel();

    results = model.predict(pyImg, ...
        pyargs('imgsz', INFERENCE_SIZE, ...
               'conf', confThreshold, ...
               'verbose', false));

    result = results{1};

    if isempty(result.masks) || result.masks.data.size(int32(0)) == 0
        quads = [];
        confidences = [];
        return;
    end

    numDetections = int32(result.masks.data.size(int32(0)));

    quads = zeros(double(numDetections), 4, 2);
    confidences = zeros(double(numDetections), 1);

    for i = 1:double(numDetections)
        pyIdx = int32(i - 1);

        maskTensor = py.operator.getitem(result.masks.data, pyIdx);
        maskCPU = maskTensor.cpu();
        if isa(maskCPU, 'py.numpy.ndarray')
            maskNumpy = maskCPU;
        else
            maskNumpy = maskCPU.numpy();
        end
        maskLogical = logical(maskNumpy);

        confTensor = py.operator.getitem(result.boxes.conf, pyIdx);
        confCPU = confTensor.cpu();
        if isa(confCPU, 'py.numpy.ndarray') || isa(confCPU, 'py.float') || isa(confCPU, 'py.int')
            conf = double(confCPU);
        else
            conf = double(confCPU.item());
        end

        [quad, maskQuality] = convertMaskToQuad(maskLogical);

        if isempty(quad)
            continue;
        end

        scaleX = double(imgW) / double(maskNumpy.shape{2});
        scaleY = double(imgH) / double(maskNumpy.shape{1});

        quad(:, 1) = quad(:, 1) * scaleX;
        quad(:, 2) = quad(:, 2) * scaleY;

        quads(i, :, :) = quad;
        confidences(i) = conf * maskQuality;
    end

    validMask = confidences > 0;
    quads = quads(validMask, :, :);
    confidences = confidences(validMask);
end

function [quad, maskQuality] = convertMaskToQuad(mask)
    cv2 = py.importlib.import_module('cv2');

    maskUint8 = py.numpy.array(uint8(mask) * uint8(255));

    contours = cv2.findContours(maskUint8, ...
        int32(cv2.RETR_EXTERNAL), ...
        int32(cv2.CHAIN_APPROX_SIMPLE));

    if isa(contours, 'py.tuple') && length(contours) >= 2
        contours = contours{1};
    end

    if isempty(contours)
        quad = [];
        maskQuality = 0;
        return;
    end

    areas = zeros(1, length(contours));
    for i = 1:length(contours)
        areas(i) = double(cv2.contourArea(contours{i}));
    end

    [~, maxIdx] = max(areas);
    largestContour = contours{maxIdx};

    quad = fitMinAreaRect(largestContour);

    maskArea = double(sum(mask(:)));
    maskQuality = computeQuadConfidence(quad, mask, maskArea);
end

function quad = fitMinAreaRect(contour)
    cv2 = py.importlib.import_module('cv2');

    rotatedRect = cv2.minAreaRect(contour);
    boxPoints = cv2.boxPoints(rotatedRect);

    quad = double(boxPoints);

    quad = orderQuadVertices(quad);
end

function quadOrdered = orderQuadVertices(quad)
    centroid = mean(quad, 1);

    angles = atan2(quad(:, 2) - centroid(2), quad(:, 1) - centroid(1));

    [~, order] = sort(angles);
    quadOrdered = quad(order, :);
end

function conf = computeQuadConfidence(quad, mask, maskArea)
    [maskH, maskW] = size(mask);

    polyMask = poly2mask(quad(:, 1) + 1, quad(:, 2) + 1, maskH, maskW);

    intersection = sum(polyMask(:) & mask(:));
    union = sum(polyMask(:) | mask(:));

    if union == 0
        conf = 0;
        return;
    end

    iou = double(intersection) / double(union);

    quadArea = polyarea(quad(:, 1), quad(:, 2));
    areaRatio = min(quadArea / maskArea, maskArea / quadArea);

    conf = iou * areaRatio;
end

%% -------------------------------------------------------------------------
%% Error Handling
%% -------------------------------------------------------------------------

function handleError(ME)
    if strcmp(ME.message, 'User stopped execution')
        fprintf('\n!! Script stopped by user\n');
        return;
    end

    fprintf('\n!! ERROR: %s\n', ME.message);
    fprintf('Stack trace:\n');
    for i = 1:numel(ME.stack)
        fprintf('  %s (line %d)\n', ME.stack(i).name, ME.stack(i).line);
    end
    rethrow(ME);
end
