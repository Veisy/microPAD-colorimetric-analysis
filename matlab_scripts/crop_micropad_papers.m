function crop_micropad_papers()
    %% microPAD Colorimetric Analysis — Paper Cropping Tool
    %% Interactive batch processor for strip-level paper crops with coordinate logging.
    %% Author: Veysel Y. Yilmaz
    %
    % Inputs:
    % - None; configuration is defined via constants inside this function.
    %
    % Outputs:
    % - Writes rectangular crops to 2_micropad_papers/[phone]/.
    % - Writes coordinates.txt alongside cropped images (image x y width height rotation).
    %
    % Behavior:
    % - Scans 1_dataset/[phone]/ for supported images, launches the crop GUI with
    %   persistent memory across runs, and saves crops while optionally preserving format.
    %
    % Note on Persistent State:
    % - This script uses persistent variables for coordinate file caching to improve performance.
    % - If you need to clear cached state between runs in the same MATLAB session, execute:
    %   clear functions

    %% ========================================================================
    %% EXPERIMENT CONFIGURATION CONSTANTS - Modify here for different setups
    %% ========================================================================
    %
    % All experimental constants are centralized here for easy modification.
    % Change these values to adapt the script for different experimental setups
    % without hunting through the code.
    %
    
    % === DATASET AND FOLDER STRUCTURE ===
    INPUT_FOLDER = '1_dataset';                        % Source dataset folder
    OUTPUT_FOLDER = '2_micropad_papers';        % Output cropped images folder
    
    % === DEFAULT CROP AREA PARAMETERS ===
    DEFAULT_CROP_WIDTH_PERCENT = 53;                   % Default crop width as % of image
    DEFAULT_CROP_HEIGHT_PERCENT = 13;                   % Default crop height as % of image
    DEFAULT_CROP_X_PERCENT = 24.5;                       % Default crop X position as % of image width
    DEFAULT_CROP_Y_PERCENT = 52;                       % Default crop Y position as % of image height
    
    % === ROTATION PARAMETERS ===
    ROTATION_RANGE = [-360, 360];                      % Slider rotation range in degrees
    QUICK_ROTATION_ANGLES = [-90, 0, 90, 180];        % Quick rotation button angles
    PORTRAIT_TO_LANDSCAPE_ANGLE = 90;                  % Auto-rotation for portrait->landscape
    LANDSCAPE_TO_PORTRAIT_ANGLE = -90;                 % Auto-rotation for landscape->portrait
    
    % === IMAGE QUALITY SETTINGS ===
    JPEG_QUALITY = 100;                                % JPEG compression quality (1-100)
    PRESERVE_FORMAT = true;                             % Preserve original image format
    % === NAMING / FILE CONSTANTS ===
    COORDINATE_FILENAME = 'coordinates.txt';            % Coordinates log filename
    SAVE_COORDINATES = true;                            % Save rectangle coordinates to coordinates.txt


    %% ========================================================================
    
    
    % Configuration - All defaults calculated from these base constants
    cfg = createConfiguration(INPUT_FOLDER, OUTPUT_FOLDER, DEFAULT_CROP_WIDTH_PERCENT, DEFAULT_CROP_HEIGHT_PERCENT, ...
                              DEFAULT_CROP_X_PERCENT, DEFAULT_CROP_Y_PERCENT, ROTATION_RANGE, QUICK_ROTATION_ANGLES, ...
                              PORTRAIT_TO_LANDSCAPE_ANGLE, LANDSCAPE_TO_PORTRAIT_ANGLE, ...
                              JPEG_QUALITY, PRESERVE_FORMAT, SAVE_COORDINATES, COORDINATE_FILENAME);
    
    try
        checkUnsupportedFiles(cfg);
        processAllFolders(cfg);
        fprintf('>> Rectangular region cropping completed successfully!\n');
    catch ME
        handleError(ME);
    end
end

%% Configuration
function cfg = createConfiguration(inputFolder, outputFolder, defaultCropWidthPercent, defaultCropHeightPercent, ...
                             defaultCropXPercent, defaultCropYPercent, rotationRange, quickRotationAngles, ...
                             portraitToLandscapeAngle, landscapeToPortraitAngle, ...
                             jpegQuality, preserveFormat, saveCoordinates, coordinateFileName)
% Create configuration structure for rectangular region cropping
%
% Inputs:
%   inputFolder - Source dataset folder path
%   outputFolder - Destination folder for cropped images
%   defaultCropWidthPercent - Default crop width as % of image width
%   defaultCropHeightPercent - Default crop height as % of image height
%   defaultCropXPercent - Default crop X position as % of image width
%   defaultCropYPercent - Default crop Y position as % of image height
%   rotationRange - [min max] rotation slider range in degrees
%   quickRotationAngles - Vector of quick rotation button angles
%   portraitToLandscapeAngle - Auto-rotation angle for portrait->landscape
%   landscapeToPortraitAngle - Auto-rotation angle for landscape->portrait
%   jpegQuality - JPEG compression quality (1-100)
%   preserveFormat - Logical, preserve original image format
%   saveCoordinates - Logical, save coordinates.txt alongside crops
%   coordinateFileName - Name of coordinates file
%
% Outputs:
%   cfg - Configuration struct with fields:
%         .defaultCropArea - Default crop area parameters
%         .rotation - Rotation parameters
%         .output - Image quality and format settings
%         .ui - UI configuration
%         .paths - Path configuration with dynamic resolution

    % Validate inputs
    validateattributes(inputFolder, {'char'}, {'nonempty'}, mfilename, 'inputFolder', 1);
    validateattributes(outputFolder, {'char'}, {'nonempty'}, mfilename, 'outputFolder', 2);
    validateattributes(defaultCropWidthPercent, {'numeric'}, {'scalar', 'positive', 'finite'}, mfilename, 'defaultCropWidthPercent', 3);
    validateattributes(defaultCropHeightPercent, {'numeric'}, {'scalar', 'positive', 'finite'}, mfilename, 'defaultCropHeightPercent', 4);
    validateattributes(defaultCropXPercent, {'numeric'}, {'scalar', 'nonnegative', 'finite'}, mfilename, 'defaultCropXPercent', 5);
    validateattributes(defaultCropYPercent, {'numeric'}, {'scalar', 'nonnegative', 'finite'}, mfilename, 'defaultCropYPercent', 6);
    validateattributes(rotationRange, {'numeric'}, {'numel', 2, 'finite'}, mfilename, 'rotationRange', 7);
    validateattributes(quickRotationAngles, {'numeric'}, {'vector', 'finite'}, mfilename, 'quickRotationAngles', 8);
    validateattributes(portraitToLandscapeAngle, {'numeric'}, {'scalar', 'finite'}, mfilename, 'portraitToLandscapeAngle', 9);
    validateattributes(landscapeToPortraitAngle, {'numeric'}, {'scalar', 'finite'}, mfilename, 'landscapeToPortraitAngle', 10);
    validateattributes(jpegQuality, {'numeric'}, {'scalar', 'integer', '>=', 1, '<=', 100}, mfilename, 'jpegQuality', 11);
    validateattributes(preserveFormat, {'logical'}, {'scalar'}, mfilename, 'preserveFormat', 12);
    validateattributes(saveCoordinates, {'logical'}, {'scalar'}, mfilename, 'saveCoordinates', 13);
    validateattributes(coordinateFileName, {'char'}, {'nonempty'}, mfilename, 'coordinateFileName', 14);

    % Default crop area parameters (using centralized constants)
    cfg.defaultCropArea = struct('widthPercent', defaultCropWidthPercent, 'heightPercent', defaultCropHeightPercent, ...
                                'xPercent', defaultCropXPercent, 'yPercent', defaultCropYPercent);
    
    % Rotation parameters (using centralized constants)
    cfg.rotation = struct('range', rotationRange, 'quickAngles', quickRotationAngles, ...
                         'portraitToLandscapeAngle', portraitToLandscapeAngle, 'landscapeToPortraitAngle', landscapeToPortraitAngle);
    
    % Image quality and format settings (using centralized constants)
    cfg.output = struct('jpegQuality', jpegQuality, ...
                    'preserveFormat', preserveFormat, ...
                    'saveCoordinates', saveCoordinates, ...
                    'coordinateFileName', coordinateFileName);

    % UI configuration
    cfg.ui = createUIConfiguration();

    % Path configuration - Dynamic path resolution
    cfg.paths = createPathConfiguration(inputFolder, outputFolder);
end

function uiCfg = createUIConfiguration()
% Create UI configuration structure with font sizes, colors, and layout
%
% Outputs:
%   uiCfg - UI configuration struct with fields:
%           .fontSize - Font sizes for all UI elements
%           .colors - Color scheme for backgrounds, buttons, and annotations
%           .positions - Normalized positions for major UI panels
%           .layout - Normalized positions for controls within panels
%           .rectangle - Rectangle drawing properties

    % Font sizes and colors
    uiCfg.fontSize = struct('title', 16, 'path', 12, 'button', 12, 'label', 11, 'value', 11, ...
                           'info', 10, 'instruction', 10, 'preview', 14, 'quickButton', 9);
    
    uiCfg.colors = struct('background', 'black', 'foreground', 'white', 'panel', [0.1 0.1 0.1], ...
                         'stop', [0.8 0.2 0.2], 'accept', [0.2 0.7 0.2], 'retry', [0.8 0.8 0.2], ...
                         'skip', [0.7 0.2 0.2], 'rectangle', 'red', 'rotation', 'cyan', 'info', 'yellow', ...
                         'path', [0.7 0.7 0.7], 'apply', [0.2 0.4 0.8]);
    
    % UI element positioning (normalized coordinates) for layout
    uiCfg.positions = struct('figure', [0 0 1 1], 'title', [0.1 0.93 0.8 0.04], ...
                            'pathDisplay', [0.1 0.89 0.8 0.03], ...
                            'image', [0.02 0.18 0.96 0.68], ... % Moved down for more space
                            'rotationPanel', [0.02 0.02 0.50 0.12], ... % Slightly taller
                            'cropButtonPanel', [0.55 0.02 0.43 0.12], ... % For APPLY/SKIP buttons in crop mode
                            'rectInfoDisplay', [0.75 0.89 0.23 0.03], ... % Top right for rect info
                            'previewPanel', [0.25 0.02 0.50 0.12], ... % Preview buttons without text
                            'stopButton', [0.02 0.93 0.06 0.05], ...
                            'instructions', [0.02 0.145 0.96 0.025], ... % Better spacing from image
                            'previewLeft', [0.02 0.18 0.47 0.65], ... % Adjusted for better fit
                            'previewRight', [0.51 0.18 0.47 0.65]); % Adjusted for better fit
    
    % Control panel layout (normalized coordinates)
    uiCfg.layout = struct('rotationLabel', [0.02 0.75 0.96 0.20], ... % Top of panel
                         'rotationSlider', [0.23 0.45 0.50 0.25], ... % Below label
                         'rotationValue', [0.75 0.45 0.23 0.25], ...
                         'applyButton', [0.15 0.35 0.30 0.35], ... % APPLY button position
                         'skipButton', [0.55 0.35 0.30 0.35]); ... % SKIP button position
    
    % Rectangle properties
    uiCfg.rectangle = struct('lineWidth', 3, 'borderWidth', 2);
end

%% File Validation
function checkUnsupportedFiles(cfg)
    fprintf('Scanning for unsupported file formats...\n');
    [supportedFiles, unsupportedFiles] = scanAllImageFiles(cfg.paths.inputPath);

    if ~isempty(unsupportedFiles)
        showUnsupportedFilesDialog(supportedFiles, unsupportedFiles);
    else
        fprintf('All image files are supported.\n');
    end
end

function [supportedFiles, unsupportedFiles] = scanAllImageFiles(inputPath)
    % Constants for array pre-allocation
    MIN_CAPACITY = 50;
    FILES_PER_FOLDER_ESTIMATE = 20;

    supportedExtensions = getSupportedImageExtensions(false);
    allImageExtensions = getAllImageExtensions();

    % Pre-allocate with conservative estimate, use geometric growth
    phoneList = getSubFolders(inputPath);
    initialCapacity = max(MIN_CAPACITY, numel(phoneList) * FILES_PER_FOLDER_ESTIMATE);

    supportedFiles = cell(initialCapacity, 1);
    unsupportedFiles = cell(initialCapacity, 1);
    supportedCount = 0;
    unsupportedCount = 0;

    for phoneIdx = 1:numel(phoneList)
        phonePath = fullfile(inputPath, phoneList{phoneIdx});
        if ~exist(phonePath, 'dir'), continue; end

        fileInfo = dir(phonePath);
        fileInfo([fileInfo.isdir]) = [];
        if isempty(fileInfo), continue; end

        % Combine fileparts and lower into single pass to reduce overhead
        names = {fileInfo.name}.';
        numFiles = numel(names);
        exts = cell(numFiles, 1);
        for i = 1:numFiles
            [~, ~, ext] = fileparts(names{i});
            exts{i} = lower(ext);
        end

        imageMask = ismember(exts, allImageExtensions);
        if ~any(imageMask), continue; end

        names = names(imageMask);
        exts = exts(imageMask);

        numImageFiles = numel(names);
        relPaths = fullfile(repmat(phoneList(phoneIdx), numImageFiles, 1), names);

        supportMask = ismember(exts, supportedExtensions);

        % Efficient insertion with geometric growth
        if any(supportMask)
            count = sum(supportMask);
            newCount = supportedCount + count;

            % Double capacity if needed (geometric growth)
            if newCount > length(supportedFiles)
                newCapacity = max(newCount, length(supportedFiles) * 2);
                supportedFiles = [supportedFiles; cell(newCapacity - length(supportedFiles), 1)];
            end

            supportedFiles(supportedCount+1:newCount) = relPaths(supportMask);
            supportedCount = newCount;
        end

        if any(~supportMask)
            count = sum(~supportMask);
            newCount = unsupportedCount + count;

            % Double capacity if needed (geometric growth)
            if newCount > length(unsupportedFiles)
                newCapacity = max(newCount, length(unsupportedFiles) * 2);
                unsupportedFiles = [unsupportedFiles; cell(newCapacity - length(unsupportedFiles), 1)];
            end

            unsupportedFiles(unsupportedCount+1:newCount) = relPaths(~supportMask);
            unsupportedCount = newCount;
        end
    end

    % Trim to actual size
    supportedFiles = supportedFiles(1:supportedCount);
    unsupportedFiles = unsupportedFiles(1:unsupportedCount);
end
function showUnsupportedFilesDialog(supportedFiles, unsupportedFiles)
    SUPPORTED_FORMATS_DISPLAY = 'JPG, JPEG, PNG, BMP, TIFF';

    [numSupported, numUnsupported] = deal(length(supportedFiles), length(unsupportedFiles));

    % Get unique unsupported extensions
    if numUnsupported > 0
        [~, ~, extCells] = cellfun(@fileparts, unsupportedFiles, 'UniformOutput', false);
        extCells = extCells(:);
        unsupportedExts = unique(cellfun(@upper, extCells, 'UniformOutput', false));
    else
        unsupportedExts = {};
    end

    % Create message and show dialog
    if numSupported > 0
        message = sprintf(['Warning: Found %d unsupported files (%s)\n\n' ...
                          'Supported: %s\n' ...
                          'Found %d supported files.\n\n' ...
                          'Continue with supported files only?'], ...
                          numUnsupported, strjoin(unsupportedExts, ', '), ...
                          SUPPORTED_FORMATS_DISPLAY, numSupported);
        choice = questdlg(message, 'Unsupported File Formats', 'Continue', 'Exit', 'Show Details', 'Continue');
    else
        message = sprintf(['Error: Found %d unsupported files (%s)\n\n' ...
                          'Supported: %s\n' ...
                          'No supported files found.'], ...
                          numUnsupported, strjoin(unsupportedExts, ', '), ...
                          SUPPORTED_FORMATS_DISPLAY);
        choice = questdlg(message, 'No Supported Files', 'Exit', 'Show Details', 'Exit');
    end
    
    switch choice
        case 'Continue'
            fprintf('Continuing with %d supported files...\n', numSupported);
        case 'Show Details'
            showFileDetails(supportedFiles, unsupportedFiles);
            showUnsupportedFilesDialog(supportedFiles, unsupportedFiles);
        case {'Exit', ''}
            error('User stopped execution');
    end
end

function showFileDetails(supportedFiles, unsupportedFiles)
    fig = figure('Name', 'File Format Details', 'Position', [100 100 800 600], ...
                'MenuBar', 'none', 'ToolBar', 'none', 'Resize', 'on');
    
    mainPanel = uipanel('Parent', fig, 'Position', [0 0 1 1], 'BackgroundColor', 'white');
    
    % Create UI elements
    uicontrol('Parent', mainPanel, 'Style', 'text', 'String', 'File Format Scan Results', ...
             'Position', [20 540 760 40], 'FontSize', 16, 'FontWeight', 'bold', ...
             'HorizontalAlignment', 'center', 'BackgroundColor', 'white');
    
    % File lists
    createFileList(mainPanel, 'Supported Files', supportedFiles, [20 260 380 240], [0 0.6 0]);
    createFileList(mainPanel, 'Unsupported Files', unsupportedFiles, [420 260 360 240], [0.8 0 0]);
    
    % Instructions and close button
    uicontrol('Parent', mainPanel, 'Style', 'text', ...
             'String', 'Supported formats can be processed. Consider converting unsupported files to JPG/PNG.', ...
             'Position', [20 180 760 60], 'FontSize', 10, 'HorizontalAlignment', 'left', ...
             'BackgroundColor', 'white', 'ForegroundColor', [0.3 0.3 0.3]);
    
    uicontrol('Parent', mainPanel, 'Style', 'pushbutton', 'String', 'Close', ...
             'Position', [350 30 100 40], 'FontSize', 12, 'Callback', @(~,~) close(fig));
    
    uiwait(fig);
end

function createFileList(parent, title, files, position, color)
    uicontrol('Parent', parent, 'Style', 'text', 'String', sprintf('%s (%d):', title, length(files)), ...
             'Position', [position(1) position(2)+260 position(3) 20], 'FontSize', 12, 'FontWeight', 'bold', ...
             'HorizontalAlignment', 'left', 'BackgroundColor', 'white', 'ForegroundColor', color);
    
    uicontrol('Parent', parent, 'Style', 'listbox', 'String', files, ...
             'Position', position, 'FontSize', 9);
end

%% Main Processing Functions
function processAllFolders(cfg)
    validatePaths(cfg);
    phoneList = getSubFolders(cfg.paths.inputPath);
    
    if isempty(phoneList)
        error('cropRect:no_phones', 'No phone folders found in: %s', cfg.paths.inputPath);
    end
    
    fprintf('Found %d phone(s): %s\n', length(phoneList), strjoin(phoneList, ', '));
    
    executeInFolder(cfg.paths.inputPath, @() processPhoneList(phoneList, cfg));
end

function processPhoneList(phoneList, cfg)
    for phoneIdx = 1:numel(phoneList)
        processPhone(phoneList{phoneIdx}, cfg);
    end
end

function processPhone(phoneName, cfg)
    fprintf('=== Processing Phone: %s ===', phoneName);

    if ~exist(phoneName, 'dir')
        warning('cropRect:missing_phone', 'Phone folder missing: %s', phoneName);
        return;
    end

    executeInFolder(phoneName, @() processImagesInPhone(phoneName, cfg));
end

function processImagesInPhone(phoneName, cfg)
    imageList = getImageFiles('.');

    if isempty(imageList)
        warning('cropRect:no_images', 'No supported images in phone folder: %s', phoneName);
        return;
    end

    fprintf('Found %d supported images\n', numel(imageList));

    outputDir = createOutputDirectory(cfg.paths.outputPath, phoneName, cfg.paths.projectRoot);
    memory = struct('hasSettings', false, 'rectPosition', [], 'rotation', 0, 'isPortrait', [], 'imageSize', [], 'rectProportions', []);

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

%% Image Processing with Persistent Window
function [updatedMemory, success, fig] = processOneImageWithPersistentWindow(imageName, outputDir, cfg, memory, isFirst, fig, phoneName)
    updatedMemory = memory;
    success = false;
    
    [img, ~, isPortrait] = loadAndAnalyzeImage(imageName);
    if isempty(img), return; end
    
    autoRotation = calculateAutoRotation(memory, isPortrait, isFirst, cfg);
    displayOrientationInfo(isPortrait, memory, autoRotation, isFirst);
    
    [rectParams, finalRotation, fig] = showInteractiveGUIWithPersistentWindow(img, imageName, cfg, memory, autoRotation, isPortrait, isFirst, fig, phoneName);
    
    if isempty(rectParams)
        fprintf('  Image skipped by user\n');
        return;
    end
    
    [success, rotatedImageSize] = saveRectangularCroppedImage(img, imageName, rectParams, finalRotation, outputDir, cfg);
    if success
        updatedMemory = updateMemory(memory, rectParams, finalRotation, isPortrait, rotatedImageSize);
    end
end

function [rectParams, finalRotation, fig] = showInteractiveGUIWithPersistentWindow(img, imageName, cfg, memory, autoRotation, isPortrait, isFirst, fig, phoneName)
    rectParams = [];
    finalRotation = autoRotation;
    
    % Create or reuse figure
    if isempty(fig) || ~isvalid(fig)
        fig = createFigure(imageName, cfg, phoneName);
    end
    
    while true
        % Clear and rebuild UI for rectangular region crop mode
        clearAndRebuildUI(fig, 'crop', img, imageName, phoneName, cfg, memory, autoRotation, isPortrait, isFirst);
        
        [action, userRect, userRotation] = waitForUserAction(fig);
        
        switch action
            case 'skip'
                return;
            case 'stop'
                close(fig); 
                error('User stopped execution');
            case 'accept'
                rectParams = userRect;
                finalRotation = userRotation;
                
                % Clear and rebuild UI for preview mode
                clearAndRebuildUI(fig, 'preview', img, imageName, phoneName, cfg, memory, autoRotation, isPortrait, isFirst, rectParams, finalRotation);
                
                [prevAction, ~, ~] = waitForUserAction(fig);
                
                switch prevAction
                    case 'accept'
                        return;
                    case {'skip', 'stop'}
                        if strcmp(prevAction, 'stop')
                            close(fig);
                            error('User stopped execution');
                        end
                        rectParams = []; 
                        return;
                    case 'retry'
                        % Continue loop to redraw crop interface
                        continue;
                end
        end
    end
end

%% Unified UI Clear and Rebuild Function
function clearAndRebuildUI(fig, mode, img, imageName, phoneName, cfg, memory, autoRotation, isPortrait, isFirst, rectParams, finalRotation)
    % Unified function to clear all UI elements and rebuild for specified mode
    
    % Get existing GUI data
    guiData = get(fig, 'UserData');
    
    % Clear ALL UI elements comprehensively
    clearAllUIElements(fig, guiData);
    
    % Rebuild UI based on mode
    switch mode
        case 'crop'
            buildRectangularCropUI(fig, img, imageName, phoneName, cfg, memory, autoRotation, isPortrait, isFirst);
        case 'preview'
            % Pass rectParams and finalRotation to preview mode
            if nargin >= 12 && exist('rectParams', 'var') && exist('finalRotation', 'var')
                buildPreviewUI(fig, img, imageName, phoneName, cfg, rectParams, finalRotation);
            else
                error('cropRect:preview_state', 'Preview mode requires rectParams and finalRotation');
            end
    end
end

function clearAllUIElements(fig, guiData)
    % UI clearing strategy that minimizes graphics object traversals

    % Critical: Delete ROI rectangles first (must be recreated due to parent change)
    if ~isempty(guiData) && isstruct(guiData) && isfield(guiData, 'rect')
        if isvalid(guiData.rect)
            delete(guiData.rect);
        end
    end

    % Single findall pass to get all graphics objects by type
    % This is faster than multiple findall calls
    allChildren = get(fig, 'Children');

    % Pre-allocate arrays for categorizing children by type
    numChildren = length(allChildren);
    axesToDelete = gobjects(numChildren, 1);
    panelsToDelete = gobjects(numChildren, 1);
    controlsToDelete = gobjects(numChildren, 1);
    axesCount = 0;
    panelsCount = 0;
    controlsCount = 0;

    for i = 1:numChildren
        child = allChildren(i);
        if ~isvalid(child), continue; end

        childType = get(child, 'Type');
        switch childType
            case 'axes'
                axesCount = axesCount + 1;
                axesToDelete(axesCount) = child;
            case 'uipanel'
                panelsCount = panelsCount + 1;
                panelsToDelete(panelsCount) = child;
            case 'uicontrol'
                controlsCount = controlsCount + 1;
                controlsToDelete(controlsCount) = child;
        end
    end

    % Trim to actual size and batch delete by type (faster than individual deletes)
    if axesCount > 0
        delete(axesToDelete(1:axesCount));
    end
    if panelsCount > 0
        delete(panelsToDelete(1:panelsCount));
    end
    if controlsCount > 0
        delete(controlsToDelete(1:controlsCount));
    end

    % Final cleanup: any remaining ROI rectangles from panels
    remainingROIs = findall(fig, 'Type', 'images.roi.Rectangle');
    if ~isempty(remainingROIs)
        delete(remainingROIs);
    end

    % Reset UserData
    set(fig, 'UserData', []);
end

function buildRectangularCropUI(fig, img, imageName, phoneName, cfg, memory, autoRotation, isPortrait, isFirst)
    % Build UI for rectangular region crop mode
    
    % Update figure name
    set(fig, 'Name', ['Rectangular Region Crop Tool - ' composePathSegments(phoneName, imageName)]);
    
    % Create new GUI data
    guiData = struct();
    guiData.mode = 'crop';
    
    % Prepare images
    [guiData.baseImg, guiData.currentImg] = prepareImages(img, autoRotation);
    
    % Create UI components
    guiData.titleHandle = createTitle(fig, isPortrait, autoRotation, isFirst, cfg);
    guiData.pathHandle = createPathDisplay(fig, phoneName, imageName, cfg);
    
    % Display image
    guiData.imgAxes = createImageAxes(fig, guiData.currentImg, cfg);
    
    % Create rectangle
    guiData.rect = createInitialRectangle(guiData.currentImg, memory, isFirst, cfg);
    
    % Create rectangle info display at top right
    guiData.rectInfoDisplay = createRectInfoDisplay(fig, guiData.rect, size(guiData.currentImg), cfg);
    
    % Create control panels - rotation panel and rectangular crop button panel
    [guiData.rotSlider, guiData.rotValue] = createRotationPanel(fig, 0, cfg);

    % Create rectangular crop mode buttons (APPLY and SKIP)
    createRectangularCropButtonPanel(fig, cfg);
    
    % Create buttons and instructions
    guiData.stopButton = createStopButton(fig, cfg);
    guiData.instructionText = createInstructions(fig, cfg);
    
    % Initialize GUI data
    guiData = initializeGUIData(guiData, img, autoRotation);
    guiData.action = '';
    
    set(fig, 'UserData', guiData);
    
    % Add listener for rectangle updates
    addlistener(guiData.rect, 'ROIMoved', @(~,~) updateRectangleInfo(fig));
end

function buildPreviewUI(fig, img, imageName, phoneName, cfg, rectParams, rotation)
    % Build UI for preview mode
    
    % Update figure name
    set(fig, 'Name', ['PREVIEW - ' composePathSegments(phoneName, imageName)]);
    
    % Create new GUI data
    guiData = struct();
    guiData.mode = 'preview';
    guiData.savedRectPosition = rectParams;
    guiData.savedRotation = rotation;  % Store rotation for preview mode
    
    % Create preview title and path
    orientationString = getOrientationString(size(img, 1) > size(img, 2));
    titleText = ternary(rotation ~= 0, ...
        sprintf('PREVIEW: %s (rotated %.0f%s)', orientationString, rotation, char(176)), ...
        sprintf('PREVIEW: %s', orientationString));
    
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
    
    % Apply rotation with preserved image content and create preview
    rotatedImage = applyRotation(img, rotation);
    
    % Create preview axes
    [guiData.leftAxes, guiData.rightAxes] = createPreviewAxes(fig, rotatedImage, rectParams, cfg);
    
    % Create buttons
    guiData.stopButton = createStopButton(fig, cfg);
    createPreviewButtons(fig, cfg);
    
    guiData.action = '';
    
    set(fig, 'UserData', guiData);
end

%% GUI Helper Functions
function fig = createFigure(imageName, cfg, phoneName)
    displayName = composePathSegments(phoneName, imageName);
    fig = figure('Name', ['Rectangular Region Crop Tool - ' displayName], ...
                'Units', 'normalized', 'Position', cfg.ui.positions.figure, ...
                'MenuBar', 'none', 'ToolBar', 'none', 'WindowState', 'maximized', ...
                'Color', cfg.ui.colors.background, 'KeyPressFcn', @keyPressHandler);
end

function titleHandle = createTitle(fig, isPortrait, autoRotation, isFirst, cfg)
    orientationString = getOrientationString(isPortrait);
    
    if isFirst
        titleText = sprintf('First Image (%s) - Set rectangular region position and rotation', orientationString);
    elseif autoRotation ~= 0
        titleText = sprintf('Auto-applied %.0f%s (%s) - Additional rotation relative to auto-rotated image', ...
                           autoRotation, char(176), orientationString);
    else
        titleText = sprintf('Same orientation (%s) - Additional rotation relative to current', orientationString);
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

function rectInfoDisplay = createRectInfoDisplay(fig, rect, imageSize, cfg)
    % Create rectangle info display at top right corner
    rectInfoDisplay = uicontrol('Parent', fig, 'Style', 'text', ...
                               'String', generateRectInfoString(rect, imageSize), ...
                               'Units', 'normalized', 'Position', cfg.ui.positions.rectInfoDisplay, ...
                               'FontSize', cfg.ui.fontSize.info, 'FontWeight', 'bold', ...
                               'ForegroundColor', cfg.ui.colors.info, ...
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

function [rotSlider, rotValue] = createRotationPanel(fig, initialRotation, cfg)
    % Rotation panel
    rotationPanel = uipanel('Parent', fig, 'Units', 'normalized', ...
                           'Position', cfg.ui.positions.rotationPanel, ...
                           'BackgroundColor', cfg.ui.colors.panel, ...
                           'BorderType', 'line', 'HighlightColor', cfg.ui.colors.foreground);
    
    % Create "Rotation Panel" label at top
    createLabel(rotationPanel, 'Rotation Panel', cfg.ui.layout.rotationLabel, cfg);
    
    % Create slider and value display
    rotSlider = createSlider(rotationPanel, initialRotation, fig, cfg);
    rotValue = createRotationValue(rotationPanel, initialRotation, cfg);
    
    % Create quick rotation buttons
    createQuickRotationButtons(rotationPanel, fig, cfg);
end

function createRectangularCropButtonPanel(fig, cfg)
    % Create panel for rectangular crop mode buttons
    cropButtonPanel = uipanel('Parent', fig, 'Units', 'normalized', ...
                             'Position', cfg.ui.positions.cropButtonPanel, ...
                             'BackgroundColor', cfg.ui.colors.panel, ...
                             'BorderType', 'line', 'HighlightColor', cfg.ui.colors.foreground);
    
    % APPLY button (blue)
    uicontrol('Parent', cropButtonPanel, 'Style', 'pushbutton', ...
             'String', 'APPLY', 'FontSize', cfg.ui.fontSize.button, 'FontWeight', 'bold', ...
             'Units', 'normalized', 'Position', cfg.ui.layout.applyButton, ...
             'BackgroundColor', cfg.ui.colors.apply, 'ForegroundColor', cfg.ui.colors.foreground, ...
             'Callback', @(~,~) setRectangularCropAction(fig, 'accept'));
    
    % SKIP button (red)
    uicontrol('Parent', cropButtonPanel, 'Style', 'pushbutton', ...
             'String', 'SKIP', 'FontSize', cfg.ui.fontSize.button, 'FontWeight', 'bold', ...
             'Units', 'normalized', 'Position', cfg.ui.layout.skipButton, ...
             'BackgroundColor', cfg.ui.colors.skip, 'ForegroundColor', cfg.ui.colors.foreground, ...
             'Callback', @(~,~) setRectangularCropAction(fig, 'skip'));
end

function setRectangularCropAction(fig, action)
    % Set action for rectangular crop mode buttons
    guiData = get(fig, 'UserData');
    guiData.action = action;
    set(fig, 'UserData', guiData);
    uiresume(fig);
end

function instructionText = createInstructions(fig, cfg)
    instructionString = 'Mouse = Drag Rectangle | Slider/Buttons = Rotate | APPLY = Accept & Save | SKIP = Skip Image | STOP = Exit';
    
    instructionText = uicontrol('Parent', fig, 'Style', 'text', 'String', instructionString, ...
             'Units', 'normalized', 'Position', cfg.ui.positions.instructions, ...
             'FontSize', cfg.ui.fontSize.instruction, 'ForegroundColor', cfg.ui.colors.foreground, ...
             'BackgroundColor', cfg.ui.colors.background, 'HorizontalAlignment', 'center');
end

function createPreviewButtons(fig, cfg)
    % Create simplified preview button panel without instruction text
    buttonPanel = uipanel('Parent', fig, 'Units', 'normalized', ...
                         'Position', cfg.ui.positions.previewPanel, ...
                         'BackgroundColor', cfg.ui.colors.panel, ...
                         'BorderType', 'line', 'HighlightColor', cfg.ui.colors.foreground, ...
                         'BorderWidth', cfg.ui.rectangle.borderWidth);
    
    % Create buttons with adjusted positions (more centered vertically)
    buttons = {'ACCEPT', 'RETRY', 'SKIP'};
    positions = {[0.05 0.25 0.25 0.50], [0.375 0.25 0.25 0.50], [0.70 0.25 0.25 0.50]};
    colors = {cfg.ui.colors.accept, cfg.ui.colors.retry, cfg.ui.colors.skip};
    actions = {'accept', 'retry', 'skip'};
    
    for i = 1:length(buttons)
        createActionButton(buttonPanel, buttons{i}, positions{i}, colors{i}, ...
                          cfg.ui.colors.foreground, actions{i}, fig, cfg);
    end
end

%% Rotation and Rectangle Handling
function rotationSliderCallback(slider, fig, cfg)
    guiData = get(fig, 'UserData');
    if ~strcmp(guiData.mode, 'crop'), return; end
    
    newRotation = get(slider, 'Value');
    rectProportions = convertRectangleToProportions(guiData.rect, size(guiData.currentImg));
    
    guiData.currentRelativeRotation = newRotation;
    guiData.totalRotation = guiData.baseRotation + newRotation;
    set(guiData.rotValue, 'String', sprintf('%.0f%s', newRotation, char(176)));
    
    guiData.currentImg = applyRotation(guiData.baseImg, newRotation);
    updateImageDisplay(guiData, rectProportions, fig, cfg);
end

function setQuickRotation(angle, fig, cfg)
    guiData = get(fig, 'UserData');
    if strcmp(guiData.mode, 'crop')
        set(guiData.rotSlider, 'Value', angle);
        rotationSliderCallback(guiData.rotSlider, fig, cfg);
    end
end

function rect = createInitialRectangle(img, memory, isFirst, cfg)
    [imageHeight, imageWidth, ~] = size(img);
    
    if memory.hasSettings && ~isFirst && ~isempty(memory.rectPosition)
        if isfield(memory, 'rectProportions') && numel(memory.rectProportions) >= 4 && all(isfinite(memory.rectProportions(1:4)))
            position = [ ...
                memory.rectProportions(1) * imageWidth, ...
                memory.rectProportions(2) * imageHeight, ...
                memory.rectProportions(3) * imageWidth, ...
                memory.rectProportions(4) * imageHeight
            ];
        elseif ~isempty(memory.imageSize)
            position = scaleRectangleToNewDimensions(memory.rectPosition, memory.imageSize, [imageHeight, imageWidth]);
        else
            defaultCrop = calculateDefaultCropArea(imageWidth, imageHeight, cfg);
            position = [defaultCrop.x, defaultCrop.y, defaultCrop.width, defaultCrop.height];
        end
    else
        defaultCrop = calculateDefaultCropArea(imageWidth, imageHeight, cfg);
        position = [defaultCrop.x, defaultCrop.y, defaultCrop.width, defaultCrop.height];
    end
    
    constrained = constrainRectangleWithinBounds(struct('x', position(1), 'y', position(2), 'width', position(3), 'height', position(4)), imageWidth, imageHeight);
    
    rect = drawrectangle('Position', [constrained.x, constrained.y, constrained.width, constrained.height], 'Color', cfg.ui.colors.rectangle, ...
                        'LineWidth', cfg.ui.rectangle.lineWidth);
end

function updateRectangleInfo(fig)
    guiData = get(fig, 'UserData');
    if strcmp(guiData.mode, 'crop') && isfield(guiData, 'rectInfoDisplay') && isvalid(guiData.rectInfoDisplay)
        set(guiData.rectInfoDisplay, 'String', generateRectInfoString(guiData.rect, size(guiData.currentImg)));
    end
end

%% User Interaction
function keyPressHandler(src, event)
    guiData = get(src, 'UserData');
    keyMap = containers.Map({'space', 'escape'}, {'accept', 'skip'});
    
    if isKey(keyMap, event.Key)
        guiData.action = keyMap(event.Key);
        set(src, 'UserData', guiData);
        uiresume(src);
    end
end

function [action, rectParams, rotation] = waitForUserAction(fig)
    uiwait(fig);
    
    [action, rectParams, rotation] = deal('', [], 0);
    
    if isvalid(fig)
        guiData = get(fig, 'UserData');
        action = guiData.action;
        
        if strcmp(action, 'accept')
            if strcmp(guiData.mode, 'preview')
                % In preview mode, use saved values
                rectParams = guiData.savedRectPosition;
                rotation = guiData.savedRotation;
            elseif strcmp(guiData.mode, 'crop')
                % In crop mode, extract current values
                rectParams = extractRectangleParameters(guiData);
                if ~isempty(rectParams)
                    rotation = guiData.totalRotation;
                else
                    action = 'skip';
                end
            end
            
            if isempty(rectParams)
                action = 'skip';
            end
        end
    end
end

function stopExecution(fig)
    guiData = get(fig, 'UserData');
    guiData.action = 'stop';
    set(fig, 'UserData', guiData);
    uiresume(fig);
end

function setPreviewAction(fig, action)
    guiData = get(fig, 'UserData');
    guiData.action = action;
    set(fig, 'UserData', guiData);
    uiresume(fig);
end

%% File Operations
function [success, rotatedImageSize] = saveRectangularCroppedImage(img, imageName, rectParams, rotation, outputDir, cfg)
    success = false;
    rotatedImageSize = [];

    try
        % Apply rotation with 'loose' to preserve entire image
        rotatedImage = applyRotation(img, rotation);
        rotatedImageSize = size(rotatedImage);
        croppedImage = cropImageWithParameters(rotatedImage, rectParams);

        [~, fileName, fileExtension] = fileparts(imageName);
        outputPath = fullfile(outputDir, [fileName fileExtension]);

        saveImageWithQuality(croppedImage, outputPath, fileExtension, cfg);

        rotationString = ternary(rotation ~= 0, sprintf(', rotated %.0f%s', rotation, char(176)), '');
        fprintf('  >> Saved: %s (%dx%d%s)\n', [fileName fileExtension], ...
                size(croppedImage, 2), size(croppedImage, 1), rotationString);

        if isfield(cfg, 'output') && isfield(cfg.output, 'saveCoordinates') && cfg.output.saveCoordinates
            appendRectangleCoordinates(outputDir, [fileName fileExtension], rectParams, rotation, rotatedImageSize, cfg);
        end

        success = true;
    catch ME
        warning('cropRect:save_failed', 'Failed to save %s: %s', imageName, ME.message);
    end
end

function appendRectangleCoordinates(outputDir, imageName, rectParams, rotation, rotatedImageSize, cfg)
% Append or update rectangle coordinates in coordinates.txt with persistent caching
%
% Inputs:
%   outputDir - Output directory path
%   imageName - Image filename
%   rectParams - Struct with fields: x, y, width, height (in pixels)
%   rotation - Rotation angle in degrees
%   rotatedImageSize - [height width] of rotated image for normalization
%   cfg - Configuration struct with .output.saveCoordinates flag
%
% Behavior:
%   Uses persistent cache to minimize file I/O. Validates cache via file size
%   and timestamp checks. Writes to temp file then atomic move for safety.
%   Format: 'image x y width height rotation' (header + data rows)

    if ~isfield(cfg, 'output') || ~isfield(cfg.output, 'saveCoordinates') || ~cfg.output.saveCoordinates
        return;
    end
    if nargin < 6 || isempty(outputDir) || isempty(imageName) || isempty(rectParams)
        return;
    end
    if numel(rotatedImageSize) < 2
        warning('cropRect:coord_size', 'Rotated image size missing for %s; skipping coordinates.', imageName);
        return;
    end

    imageHeight = rotatedImageSize(1);
    imageWidth = rotatedImageSize(2);

    if imageHeight <= 0 || imageWidth <= 0
        warning('cropRect:coord_size', 'Invalid rotated image size for %s; skipping coordinates.', imageName);
        return;
    end

    [x1, x2, y1, y2] = computePixelBounds(rectParams, imageWidth, imageHeight);

    width = max(0, x2 - x1 + 1);
    height = max(0, y2 - y1 + 1);
    rotValue = rotation;
    if ~isfinite(rotValue)
        rotValue = 0;
    end
    rotValue = round(rotValue, 6);
    newRow = double([x1, y1, width, height, rotValue]);

    coordPath = fullfile(outputDir, cfg.output.coordinateFileName);
    header = 'image x y width height rotation';

    persistent caches
    if isempty(caches)
        caches = createEmptyCoordinateCacheStruct();
    end

    cacheIdx = find(strcmp({caches.path}, coordPath), 1);
    if isempty(cacheIdx)
        cache = createEmptyCoordinateCache(coordPath);
        caches(end + 1) = cache;
        cacheIdx = numel(caches);
    else
        cache = caches(cacheIdx);
    end

    fileExists = exist(coordPath, 'file') == 2;
    needsReload = false;

    % Caching strategy that validates file size before reload
    % Cache fileInfo result to avoid redundant dir() calls
    cachedFileInfo = [];
    fileStamp = NaN;  % Initialize for later use

    if fileExists
        if isempty(cache.names)
            % Cache empty - initial load required
            needsReload = true;
        else
            % Cache populated - perform cheap validation first
            % Quick check: file size change indicates modification
            cachedFileInfo = dir(coordPath);
            if ~isempty(cachedFileInfo)
                currentSize = cachedFileInfo(1).bytes;
                fileStamp = cachedFileInfo(1).datenum;
                if isfield(cache, 'fileSize') && cache.fileSize ~= currentSize
                    % File size changed - reload required
                    needsReload = true;
                elseif ~isfield(cache, 'fileSize')
                    % Cache without size tracking - use timestamp validation
                    needsReload = ~isfinite(cache.lastModified) || timestampsMismatch(cache.lastModified, fileStamp);
                end
                % If size unchanged and cache has size field, trust cache (fast path)
            else
                % File disappeared - reload to handle gracefully
                needsReload = true;
            end
        end
    elseif ~isempty(cache.names)
        % File deleted externally - clear cache
        cache = createEmptyCoordinateCache(coordPath);
    end

    if fileExists && needsReload
        cache = loadCoordinateCache(coordPath);
        % Reuse cached fileInfo if available, otherwise fetch
        if isempty(cachedFileInfo)
            cachedFileInfo = dir(coordPath);
        end
        if ~isempty(cachedFileInfo)
            cache.lastModified = cachedFileInfo(1).datenum;
            cache.fileSize = cachedFileInfo(1).bytes;  % Track file size for quick validation
            fileStamp = cachedFileInfo(1).datenum;
        end
    end

    [updatedCache, rowChanged] = integrateCoordinateRow(cache, imageName, newRow);

    if ~rowChanged
        if fileExists
            cache.lastModified = fileStamp;
        else
            cache.lastModified = NaN;
        end
        caches(cacheIdx) = cache;
        return;
    end

    [writeOk, newStamp, newSize] = writeCoordinateEntries(coordPath, header, updatedCache.names, updatedCache.values);
    if writeOk
        updatedCache.lastModified = newStamp;
        updatedCache.fileSize = newSize;
        caches(cacheIdx) = updatedCache;
    else
        if fileExists
            fallbackCache = loadCoordinateCache(coordPath);
            fileInfo = dir(coordPath);
            if ~isempty(fileInfo)
                fallbackCache.lastModified = fileInfo(1).datenum;
                fallbackCache.fileSize = fileInfo(1).bytes;
            end
        else
            fallbackCache = createEmptyCoordinateCache(coordPath);
        end
        caches(cacheIdx) = fallbackCache;
    end
end


function caches = createEmptyCoordinateCacheStruct()
    caches = struct('path', {}, 'names', {}, 'values', {}, 'fullLower', {}, 'baseLower', {}, 'lastModified', {}, 'fileSize', {});
end

function cache = createEmptyCoordinateCache(coordPath)
    cache = struct('path', coordPath, 'names', [], 'values', zeros(0, 5), ...
                   'fullLower', [], 'baseLower', [], 'lastModified', NaN, 'fileSize', 0);
    cache.names = cell(0, 1);
    cache.fullLower = cell(0, 1);
    cache.baseLower = cell(0, 1);
end

function cache = loadCoordinateCache(coordPath)
    cache = createEmptyCoordinateCache(coordPath);
    if exist(coordPath, 'file') ~= 2
        return;
    end

    [names, values] = readCoordinateEntries(coordPath);
    if isempty(names)
        return;
    end

    cache.names = names;
    cache.values = values;
    cache.fullLower = cellfun(@lower, names, 'UniformOutput', false);
    [~, bases, ~] = cellfun(@fileparts, names, 'UniformOutput', false);
    cache.baseLower = cellfun(@lower, bases, 'UniformOutput', false);
end

function [names, values] = readCoordinateEntries(coordPath)
% Read coordinate entries from coordinates.txt file
%
% Inputs:
%   coordPath - Full path to coordinates.txt file
%
% Outputs:
%   names - Cell array of image filenames
%   values - Nx5 matrix [x y width height rotation] in pixels/degrees
%
% Behavior:
%   Skips header line. Filters invalid entries (non-finite x/y/width/height).
%   Sets rotation to 0 for missing/invalid rotation values.

    names = cell(0, 1);
    values = zeros(0, 5);

    fid = fopen(coordPath, 'rt');
    if fid == -1
        warning('cropRect:coord_open', 'Cannot open coordinates file for reading: %s', coordPath);
        return;
    end
    cleaner = onCleanup(@() fclose(fid));

    % Read entire file, skipping header line
    % Header format: 'image x y width height rotation'
    raw = textscan(fid, '%s%f%f%f%f%f', 'HeaderLines', 1, 'Delimiter', sprintf(' \t'), ...
                   'MultipleDelimsAsOne', true, 'CollectOutput', true);
    if isempty(raw) || isempty(raw{1})
        return;
    end

    names = raw{1};
    numericVals = raw{2};
    if isempty(numericVals)
        names = cell(0, 1);
        values = zeros(0, 5);
        return;
    end
    if size(numericVals, 2) < 5
        numericVals(:, 5) = 0;
    end

    validMask = all(isfinite(numericVals(:, 1:4)), 2);
    if all(~validMask)  % If all entries invalid
        names = cell(0, 1);
        values = zeros(0, 5);
        return;
    end

    names = names(validMask);
    numericVals = numericVals(validMask, :);
    numericVals(~isfinite(numericVals(:, 5)), 5) = 0;
    values = numericVals;
end

function [cache, rowChanged] = integrateCoordinateRow(cache, imageName, newRow)
    newFullLower = lower(imageName);
    [~, newBase, ~] = fileparts(imageName);
    newBaseLower = lower(newBase);

    if isempty(cache.fullLower)
        matchMask = false(0, 1);
    else
        matchMask = strcmp(cache.fullLower, newFullLower) | strcmp(cache.baseLower, newBaseLower);
    end

    rowChanged = true;
    if any(matchMask)
        if sum(matchMask) == 1
            prevIdx = find(matchMask, 1, 'first');
            prevName = cache.names{prevIdx};
            prevValues = cache.values(prevIdx, :);
            sameName = strcmp(prevName, imageName);
            sameValues = isequal(prevValues, newRow);
            if sameName && sameValues
                rowChanged = false;
            end
        end
        if rowChanged
            cache.names(matchMask) = [];
            cache.values(matchMask, :) = [];
            cache.fullLower(matchMask) = [];
            cache.baseLower(matchMask) = [];
        end
    end

    if rowChanged
        cache.names{end+1, 1} = imageName;
        cache.values(end+1, :) = newRow;
        cache.fullLower{end+1, 1} = newFullLower;
        cache.baseLower{end+1, 1} = newBaseLower;
    end
end

function [ok, newStamp, newSize] = writeCoordinateEntries(coordPath, header, names, values)
    ok = false;
    newStamp = NaN;
    newSize = 0;

    coordDir = fileparts(coordPath);
    if isempty(coordDir)
        coordDir = pwd;
    end
    tmpPath = tempname(coordDir);

    fid_w = fopen(tmpPath, 'wt');
    if fid_w == -1
        warning('cropRect:coord_open', 'Cannot open temp coordinates file for writing: %s', tmpPath);
        return;
    end

    writeFailed = false;
    try
        fprintf(fid_w, '%s\n', header);
        for i = 1:numel(names)
            row = values(i, :);
            fprintf(fid_w, '%s %.0f %.0f %.0f %.0f %.6f\n', names{i}, row(1), row(2), row(3), row(4), row(5));
        end
    catch ME
        writeFailed = true;
        warning('cropRect:coord_write', 'Failed to write coordinates: %s', ME.message);
    end
    fclose(fid_w);

    if writeFailed
        if exist(tmpPath, 'file')
            delete(tmpPath);
        end
        return;
    end

    [ok, msg, msgid] = movefile(tmpPath, coordPath, 'f');
    if ~ok
        warning('cropRect:coord_move', 'Failed to move temp file to coordinates.txt: %s (%s). Attempting fallback copy.', msg, msgid);
        [copied, cmsg, cmsgid] = copyfile(tmpPath, coordPath, 'f');
        if ~copied
            warning('cropRect:coord_copy', 'Fallback copy failed: %s (%s). Coordinates may be stale.', cmsg, cmsgid);
            if exist(tmpPath, 'file')
                delete(tmpPath);
            end
            return;
        end
        ok = copied;
    end

    if exist(tmpPath, 'file')
        delete(tmpPath);
    end

    fileInfo = dir(coordPath);
    if ~isempty(fileInfo)
        newStamp = fileInfo(1).datenum;
        newSize = fileInfo(1).bytes;
    else
        newStamp = NaN;
        newSize = 0;
    end
end

function saveImageWithQuality(img, outputPath, fileExtension, cfg)
    narginchk(4, 4);
    validateattributes(img, {'numeric', 'logical'}, {'nonempty'}, mfilename, 'img', 1);
    validateattributes(outputPath, {'char'}, {'nonempty'}, mfilename, 'outputPath', 2);
    validateattributes(fileExtension, {'char'}, {'nonempty'}, mfilename, 'fileExtension', 3);

    if cfg.output.preserveFormat && any(strcmpi(fileExtension, {'.jpg', '.jpeg'}))
        imwrite(img, outputPath, 'JPEG', 'Quality', cfg.output.jpegQuality);
    else
        imwrite(img, outputPath);
    end
end

%% Utility Functions
function memory = updateMemory(memory, rectParams, rotation, isPortrait, rotatedImageSize)
    if nargin < 5 || isempty(rotatedImageSize)
        rotatedImageSize = [rectParams.height, rectParams.width];
    end
    
    if numel(rotatedImageSize) < 2 || any(rotatedImageSize(1:2) <= 0)
        rotatedImageSize = [rectParams.height, rectParams.width];
    end
    
    rotatedImageSize = double(rotatedImageSize);
    rotatedImageHeight = rotatedImageSize(1);
    rotatedImageWidth = rotatedImageSize(2);
    
    memory.hasSettings = true;
    memory.rectPosition = [rectParams.x, rectParams.y, rectParams.width, rectParams.height];
    memory.rotation = rotation;
    memory.isPortrait = isPortrait;
    memory.imageSize = [rotatedImageHeight, rotatedImageWidth];
    if rotatedImageHeight > 0 && rotatedImageWidth > 0
        memory.rectProportions = [ ...
            rectParams.x / rotatedImageWidth, ...
            rectParams.y / rotatedImageHeight, ...
            rectParams.width / rotatedImageWidth, ...
            rectParams.height / rotatedImageHeight
        ];
    else
        memory.rectProportions = [];
    end
end

function guiData = initializeGUIData(guiData, originalImg, autoRotation)
    guiData.originalImg = originalImg;
    guiData.baseRotation = autoRotation;
    guiData.currentRelativeRotation = 0;
    guiData.totalRotation = autoRotation;
    guiData.action = '';
    % mode is already set in buildRectangularCropUI
end

function [img, dimensions, isPortrait] = loadAndAnalyzeImage(imageName)
    try
        img = imread_raw(imageName);
        [imageHeight, imageWidth, ~] = size(img);
        dimensions = [imageHeight, imageWidth];
        isPortrait = imageHeight > imageWidth;
    catch ME
        warning('cropRect:read_failed', 'Cannot read %s: %s', imageName, ME.message);
        [img, dimensions, isPortrait] = deal([], [], []);
    end
end

function [baseImg, currentImg] = prepareImages(img, autoRotation)
    % Apply auto-rotation with preserved aspect ratio (loose mode)
    baseImg = applyRotation(img, autoRotation);
    currentImg = baseImg;
    if autoRotation ~= 0
        fprintf('    Applied auto-rotation: %.0f%s\n', autoRotation, char(176));
    end
end

function rotatedImg = applyRotation(img, rotation)
    % Use 'loose' to preserve entire image and avoid cropping
    % Handle exact 90-degree multiples using lossless rot90

    if rotation == 0
        rotatedImg = img;
        return;
    end

    % For exact 90-degree multiples, use lossless rot90
    % Lossless transform avoids interpolation overhead
    if abs(mod(rotation, 90)) < 1e-6  % Handle floating-point precision
        numRotations = mod(round(rotation / 90), 4);
        if numRotations == 0
            rotatedImg = img;
        else
            % rot90 rotates counter-clockwise, imrotate rotates clockwise
            % Adjust direction to match imrotate behavior
            rotatedImg = rot90(img, -numRotations);
        end
    else
        % Fallback to bilinear interpolation for non-90-degree angles
        rotatedImg = imrotate(img, rotation, 'bilinear', 'loose');
    end
end

function croppedImg = cropImageWithParameters(img, rectParams)
    [imageHeight, imageWidth, ~] = size(img);
    [x1, x2, y1, y2] = computePixelBounds(rectParams, imageWidth, imageHeight);
    croppedImg = img(y1:y2, x1:x2, :);
end


function [x1, x2, y1, y2] = computePixelBounds(rectParams, imageWidth, imageHeight)
    x = rectParams.x;
    y = rectParams.y;
    width = rectParams.width;
    height = rectParams.height;

    if ~isfinite(x), x = 1; end
    if ~isfinite(y), y = 1; end
    if ~isfinite(width) || width <= 0, width = 1; end
    if ~isfinite(height) || height <= 0, height = 1; end

    x1 = round(max(1, x));
    y1 = round(max(1, y));
    x2 = round(min(imageWidth, x + width - 1));
    y2 = round(min(imageHeight, y + height - 1));

    x2 = max(x1, x2);
    y2 = max(y1, y2);
end

function rectParams = extractRectangleParameters(guiData)
    if strcmp(guiData.mode, 'preview') && isfield(guiData, 'savedRectPosition')
        rectParams = guiData.savedRectPosition;
    elseif isfield(guiData, 'rect') && isvalid(guiData.rect)
        position = guiData.rect.Position;
        rectParams = struct('x', position(1), 'y', position(2), ...
                           'width', position(3), 'height', position(4));
    else
        rectParams = [];
    end
end

function updateImageDisplay(guiData, rectProportions, fig, cfg)
    cla(guiData.imgAxes);
    imshow(guiData.currentImg, 'Parent', guiData.imgAxes, 'InitialMagnification', 'fit');
    axis(guiData.imgAxes, 'image');
    axis(guiData.imgAxes, 'tight');
    hold(guiData.imgAxes, 'on');
    
    delete(guiData.rect);
    [imageHeight, imageWidth, ~] = size(guiData.currentImg);
    
    newPosition = [
        rectProportions(1) * imageWidth,   ... x
        rectProportions(2) * imageHeight,  ... y
        rectProportions(3) * imageWidth,   ... width
        rectProportions(4) * imageHeight   ... height
    ];
    
    constrainedParams = constrainRectangleWithinBounds(...
        struct('x', newPosition(1), 'y', newPosition(2), 'width', newPosition(3), 'height', newPosition(4)), ...
        imageWidth, imageHeight);
    
    finalPosition = [constrainedParams.x, constrainedParams.y, constrainedParams.width, constrainedParams.height];
    
    guiData.rect = drawrectangle('Position', finalPosition, ...
                                'Color', cfg.ui.colors.rectangle, ...
                                'LineWidth', cfg.ui.rectangle.lineWidth);
    
    % Update rect info display instead of rectInfo
    if isfield(guiData, 'rectInfoDisplay') && isvalid(guiData.rectInfoDisplay)
        set(guiData.rectInfoDisplay, 'String', generateRectInfoString(guiData.rect, [imageHeight, imageWidth]));
    end
    
    addlistener(guiData.rect, 'ROIMoved', @(~,~) updateRectangleInfo(fig));
    
    set(fig, 'UserData', guiData);
end

function [leftAxes, rightAxes] = createPreviewAxes(fig, rotatedImg, rectParams, cfg)
    % Original with overlay
    leftAxes = axes('Parent', fig, 'Units', 'normalized', 'Position', cfg.ui.positions.previewLeft);
    imshow(rotatedImg, 'Parent', leftAxes);
    axis(leftAxes, 'image');
    axis(leftAxes, 'tight');
    title(leftAxes, 'Original with Rectangular Crop Area', ...
          'Color', cfg.ui.colors.foreground, 'FontSize', cfg.ui.fontSize.button, 'FontWeight', 'bold');
    hold(leftAxes, 'on');
    rectangle('Parent', leftAxes, ...
             'Position', [rectParams.x, rectParams.y, rectParams.width, rectParams.height], ...
             'EdgeColor', cfg.ui.colors.rectangle, 'LineWidth', cfg.ui.rectangle.lineWidth);
    
    % Cropped result
    croppedImg = cropImageWithParameters(rotatedImg, rectParams);
    rightAxes = axes('Parent', fig, 'Units', 'normalized', 'Position', cfg.ui.positions.previewRight);
    imshow(croppedImg, 'Parent', rightAxes);
    axis(rightAxes, 'image');
    axis(rightAxes, 'tight');
    title(rightAxes, sprintf('Rectangular Cropped Result (%dx%d pixels)', size(croppedImg, 2), size(croppedImg, 1)), ...
          'Color', cfg.ui.colors.foreground, 'FontSize', cfg.ui.fontSize.button, 'FontWeight', 'bold');
end

function displayOrientationInfo(isPortrait, memory, autoRotation, isFirst)
    orientationString = getOrientationString(isPortrait);
    
    if isFirst
        fprintf('  %s - first image\n', orientationString);
    elseif memory.hasSettings && (isPortrait ~= memory.isPortrait)
        previousOrientationString = getOrientationString(memory.isPortrait);
        fprintf('  %s - changed from %s, auto-rotating %.0f%s\n', ...
                orientationString, previousOrientationString, autoRotation, char(176));
    else
        fprintf('  %s - applying saved rotation %.0f%s\n', orientationString, autoRotation, char(176));
    end
end

function logProcessingResult(success, isFirst)
    if success
        fprintf(ternary(isFirst, '>> Settings saved for next images\n', '>> Applied saved settings\n'));
    else
        fprintf('!! Image skipped\n');
    end
end

function infoString = generateRectInfoString(rect, imageSize)
    position = rect.Position;
    [imageHeight, imageWidth] = deal(imageSize(1), imageSize(2));
    
    widthPercent = (position(3) / imageWidth) * 100;
    heightPercent = (position(4) / imageHeight) * 100;
    
    infoString = sprintf('Rect: [%.0f,%.0f] %.0fx%.0f (%.1f%% x %.1f%%)', ...
                        position(1), position(2), position(3), position(4), ...
                        widthPercent, heightPercent);
end

function orientationString = getOrientationString(isPortrait)
    orientationString = ternary(isPortrait, 'Portrait', 'Landscape');
end

%% Crop Area Calculation Functions
function cropParams = calculateDefaultCropArea(imageWidth, imageHeight, cfg)
    cropWidth = imageWidth * (cfg.defaultCropArea.widthPercent / 100);
    cropHeight = imageHeight * (cfg.defaultCropArea.heightPercent / 100);
    cropX = imageWidth * (cfg.defaultCropArea.xPercent / 100);
    cropY = imageHeight * (cfg.defaultCropArea.yPercent / 100);
    
    cropParams = struct('x', cropX, 'y', cropY, 'width', cropWidth, 'height', cropHeight);
end

function scaledParams = scaleRectangleToNewDimensions(oldParams, oldDimensions, newDimensions)
    invalidOldDims = numel(oldDimensions) < 2 || any(oldDimensions(1:2) <= 0);
    invalidNewDims = numel(newDimensions) < 2 || any(newDimensions(1:min(2, numel(newDimensions))) <= 0);
    if invalidOldDims || invalidNewDims
        widthCap = oldParams(3);
        heightCap = oldParams(4);
        if numel(newDimensions) >= 2 && isfinite(newDimensions(2)) && newDimensions(2) > 0
            widthCap = min(widthCap, newDimensions(2));
        end
        if numel(newDimensions) >= 1 && isfinite(newDimensions(1)) && newDimensions(1) > 0
            heightCap = min(heightCap, newDimensions(1));
        end
        scaledParams = [1, 1, widthCap, heightCap];
        return;
    end

    scaleY = newDimensions(1) / oldDimensions(1);  % Height scaling
    scaleX = newDimensions(2) / oldDimensions(2);  % Width scaling

    newWidth = max(oldParams(3) * scaleX, 1);
    newHeight = max(oldParams(4) * scaleY, 1);

    maxX = max(1, newDimensions(2) - newWidth + 1);
    maxY = max(1, newDimensions(1) - newHeight + 1);

    newX = max(1, min(oldParams(1) * scaleX, maxX));
    newY = max(1, min(oldParams(2) * scaleY, maxY));

    newWidth = min(newWidth, newDimensions(2) - newX + 1);
    newHeight = min(newHeight, newDimensions(1) - newY + 1);

    scaledParams = [newX, newY, newWidth, newHeight];
end


function constrainedParams = constrainRectangleWithinBounds(params, maxWidth, maxHeight)
    if ~isfield(params, 'x') || ~isfinite(params.x), params.x = 1; end
    if ~isfield(params, 'y') || ~isfinite(params.y), params.y = 1; end
    if ~isfield(params, 'width') || ~isfinite(params.width) || params.width <= 0, params.width = maxWidth; end
    if ~isfield(params, 'height') || ~isfinite(params.height) || params.height <= 0, params.height = maxHeight; end

    availableX = max(1, maxWidth - params.width + 1);
    availableY = max(1, maxHeight - params.height + 1);

    x = max(1, min(params.x, availableX));
    y = max(1, min(params.y, availableY));

    width = max(1, min(params.width, maxWidth - x + 1));
    height = max(1, min(params.height, maxHeight - y + 1));

    constrainedParams = struct('x', x, 'y', y, 'width', width, 'height', height);
end


function rotationAngle = calculateAutoRotation(memory, isPortrait, isFirst, cfg)
    if isFirst || ~memory.hasSettings
        rotationAngle = 0;
    elseif isPortrait ~= memory.isPortrait
        rotationAngle = ternary(isPortrait, cfg.rotation.portraitToLandscapeAngle, cfg.rotation.landscapeToPortraitAngle);
    else
        rotationAngle = memory.rotation;
    end
end

function proportions = convertRectangleToProportions(rect, imageSize)
    position = rect.Position;
    [imageHeight, imageWidth] = deal(imageSize(1), imageSize(2));
    
    proportions = [position(1) / imageWidth, position(2) / imageHeight, ...
                  position(3) / imageWidth, position(4) / imageHeight];
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

function slider = createSlider(parent, initialValue, fig, cfg)
    slider = uicontrol('Parent', parent, 'Style', 'slider', ...
                      'Min', cfg.rotation.range(1), 'Max', cfg.rotation.range(2), ...
                      'Value', initialValue, 'Units', 'normalized', ...
                      'Position', cfg.ui.layout.rotationSlider, ...
                      'Callback', @(src,~) rotationSliderCallback(src, fig, cfg));
end

function rotValue = createRotationValue(parent, initialRotation, cfg)
    rotValue = uicontrol('Parent', parent, 'Style', 'text', ...
                        'String', sprintf('%.0f%s', initialRotation, char(176)), ...
                        'Units', 'normalized', 'Position', cfg.ui.layout.rotationValue, ...
                        'FontSize', cfg.ui.fontSize.value, 'FontWeight', 'bold', ...
                        'ForegroundColor', cfg.ui.colors.rotation, ...
                        'BackgroundColor', cfg.ui.colors.panel);
end

function createQuickRotationButtons(parent, fig, cfg)
    angles = cfg.rotation.quickAngles;
    numButtons = length(angles);
    
    % Calculate evenly distributed button positions
    totalWidth = 0.96;  % Total available width (leaving 2% margin on each side)
    buttonWidth = 0.15;  % Individual button width
    totalButtonWidth = buttonWidth * numButtons;
    spacing = (totalWidth - totalButtonWidth) / (numButtons + 1);  % Even spacing between buttons
    
    buttonHeight = 0.30;
    buttonY = 0.10;  % Bottom of panel
    
    for i = 1:numButtons
        angle = angles(i);
        % Calculate x position with even distribution
        buttonX = spacing + (i-1) * (buttonWidth + spacing);
        
        uicontrol('Parent', parent, 'Style', 'pushbutton', ...
                 'String', sprintf('%d%s', angle, char(176)), 'Units', 'normalized', ...
                 'Position', [buttonX, buttonY, buttonWidth, buttonHeight], ...
                 'FontSize', cfg.ui.fontSize.quickButton, ...
                 'Callback', @(~,~) setQuickRotation(angle, fig, cfg));
    end
end

function btn = createActionButton(parent, text, position, bgColor, fgColor, action, fig, cfg)
    btn = uicontrol('Parent', parent, 'Style', 'pushbutton', ...
                   'String', text, 'FontSize', cfg.ui.fontSize.preview, 'FontWeight', 'bold', ...
                   'Units', 'normalized', 'Position', position, ...
                   'BackgroundColor', bgColor, 'ForegroundColor', fgColor, ...
                   'Callback', @(~,~) setPreviewAction(fig, action));
end

%% Path and File Utilities
function validatePaths(cfg)
    if ~exist(cfg.paths.inputPath, 'dir')
        error('cropRect:input_missing', 'Input directory not found: %s', cfg.paths.inputPath);
    end
    
    fullOutputPath = fullfile(cfg.paths.projectRoot, cfg.paths.outputPath);
    if ~exist(fullOutputPath, 'dir')
        mkdir(fullOutputPath);
        fprintf('Created output directory: %s\n', fullOutputPath);
    end
end

function outputDir = createOutputDirectory(basePath, phoneName, projectRoot)
    outputDir = fullfile(projectRoot, basePath, phoneName);

    if ~exist(outputDir, 'dir')
        mkdir(outputDir);
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
    if isempty(folder)
        folder = '.';
    end
    currentDir = pwd;
    cd(folder);
    try
        func();
    catch ME
        cd(currentDir);
        rethrow(ME);
    end
    cd(currentDir);
end

function folders = getSubFolders(path)
    items = dir(path);
    isFolder = [items.isdir];
    names = {items(isFolder).name};
    folders = names(~ismember(names, {'.', '..'}));
end

function files = getImageFiles(path)
    supportedExtensions = getSupportedImageExtensions(false);

    fileInfo = dir(path);
    fileInfo([fileInfo.isdir]) = [];
    if isempty(fileInfo)
        files = {};
        return;
    end

    names = {fileInfo.name};
    [~, ~, exts] = cellfun(@fileparts, names, 'UniformOutput', false);
    exts = cellfun(@lower, exts, 'UniformOutput', false);
    isSupported = ismember(exts, supportedExtensions);

    files = names(isSupported);
end

function extensions = getSupportedImageExtensions(asPatterns)
    baseExtensions = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.tif'};
    if nargin >= 1 && asPatterns
        extensions = cellfun(@(ext) ['*' ext], baseExtensions, 'UniformOutput', false);
    else
        extensions = baseExtensions;
    end
end

function extensions = getAllImageExtensions()
    extensions = [getSupportedImageExtensions(false), {'.heic', '.heif', '.webp', '.raw', '.cr2', '.nef', '.arw'}];
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
function pathConfig = createPathConfiguration(inputFolder, outputFolder)
% Create path configuration with dynamic project root resolution
%
% Inputs:
%   inputFolder - Relative path to input dataset folder
%   outputFolder - Relative path to output folder
%
% Outputs:
%   pathConfig - Struct with fields:
%                .projectRoot - Resolved absolute path to project root
%                .inputPath - Absolute path to input folder
%                .outputPath - Relative path to output folder
%
% Behavior:
%   Searches up directory tree (max 5 levels) to find inputFolder.
%   Falls back to current directory if not found.

    % Create path configuration with dynamic resolution
    projectRoot = findProjectRoot(inputFolder);
    pathConfig = struct('projectRoot', projectRoot, ...
                       'inputPath', fullfile(projectRoot, inputFolder), ...
                       'outputPath', outputFolder);
end

function projectRoot = findProjectRoot(inputFolder)
    % Find project root by searching for the input folder
    currentDir = pwd;
    searchDir = currentDir;
    maxLevels = 5; % Safety limit to prevent infinite loops
    
    for level = 1:maxLevels
        [parentDir, ~] = fileparts(searchDir);
        
        % Check if input folder exists at current level
        if exist(fullfile(searchDir, inputFolder), 'dir')
            projectRoot = searchDir;
            return;
        end
        
        % Move up one level
        if strcmp(searchDir, parentDir)
            break; % Reached root directory
        end
        searchDir = parentDir;
    end
    
    % Fallback: use current directory and issue warning
    warning('cropRect:path_resolution', 'Could not find input folder "%s". Using current directory as project root.', inputFolder);
    projectRoot = currentDir;
end

%% Helper Functions
function result = ternary(condition, trueValue, falseValue)
% Ternary conditional operator
%
% Inputs:
%   condition - Logical scalar or expression
%   trueValue - Value returned if condition is true
%   falseValue - Value returned if condition is false
%
% Outputs:
%   result - trueValue or falseValue based on condition

    if condition
        result = trueValue;
    else
        result = falseValue;
    end
end

function I = imread_raw(fname)
% Read image with EXIF orientation correction for 90-degree rotations only
%
% Inputs:
%   fname - Image file path (string)
%
% Outputs:
%   I - Image array with EXIF orientation 5/6/7/8 inverted to preserve raw layout
%
% Behavior:
%   Inverts EXIF 90-degree rotation tags (5/6/7/8) to preserve on-disk pixel layout.
%   This prevents double-rotation when imread honors EXIF orientation.
%   EXIF tags 1-4 (flips/180) are not modified.
%   Rationale: User expects to see and rotate raw sensor orientation, not camera's
%   corrected orientation.

    % Read (some builds honor AutoOrient=false; some ignore it silently)
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

