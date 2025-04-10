function lgraph = createUNet( ...
    inputSize, encoderDepth, initialEncoderNumChannels, outputNumChannels)
% createUNet creates a fully convolutional U-net network with a specified depth.
%
%  This function constructs a fully convolutional U-net network.
%  The U-net architecture includes an encoder and a decoder with a bridging layer in between.
%  Users can specify the network depth, initial number of channels in the encoder, and the number of output channels.
%
%  Input Parameters:
%    inputSize - Size of the network input image, given as a three-element vector [Height Width Channels].
%    encoderDepth - Depth of the encoder, determining the number of downsampling stages in the network.
%    initialEncoderNumChannels - Number of channels in the first layer of the encoder.
%    outputNumChannels - Number of channels in the output layer.
%
%  Output Parameters:
%    lgraph - Layer graph of the constructed U-net network.

% Adjustable parameters:
isSingleRow = inputSize(1) == 1;
isSingleCol = inputSize(2) == 1;
convFilterSize = [3-2*isSingleRow 3-2*isSingleCol]; % Convolution filter size
upConvFilterSize = [2-isSingleRow 2-isSingleCol]; % Transposed convolution filter size (corresponding to upsampling factor)
poolSize = [2-isSingleRow 2-isSingleCol]; % Pooling area size (corresponding to downsampling factor)

% Create image input layer:
inputlayer = imageInputLayer(inputSize,'Name','ImageInputLayer', ...
    'Normalization','none');

% Create encoder:
[encoder, finalNumChannels] = iCreateEncoder(encoderDepth, ...
    convFilterSize, poolSize, initialEncoderNumChannels, 'same');

% Create the bridge between encoder and decoder:
firstConv = iCreateAndInitializeConvLayer(convFilterSize, ...
    2*finalNumChannels, 'Bridge-Conv-1', 'same');
firstReLU = reluLayer('Name','Bridge-ReLU-1');

secondConv = iCreateAndInitializeConvLayer(convFilterSize, ...
    2*finalNumChannels, 'Bridge-Conv-2', 'same');
secondReLU = reluLayer('Name','Bridge-ReLU-2');

encoderDecoderBridge = [firstConv; firstReLU; secondConv; secondReLU];

dropOutLayer = dropoutLayer(0.5,'Name','Bridge-DropOut');
encoderDecoderBridge = [encoderDecoderBridge; dropOutLayer];

% Create decoder:
initialDecoderNumChannels = finalNumChannels;
decoder = iCreateDecoder(encoderDepth, upConvFilterSize, ...
    convFilterSize, initialDecoderNumChannels, 'same');

% Connect input layer, encoder, bridge, and decoder:
layers = [inputlayer; encoder; encoderDecoderBridge; decoder];

finalConv = convolution2dLayer(1, outputNumChannels, ...
    'BiasL2Factor', 0, ...
    'Name','Final-ConvolutionLayer', 'Padding', 'same', ...
    'WeightsInitializer', 'he', 'BiasInitializer','zeros');

layers = [layers; finalConv];

lgraph = layerGraph(layers);

lgraph = iConnectLgraph(lgraph, encoderDepth);
end


% Helper Function 1: Create the encoder.
function [encoder, finalNumChannels] = iCreateEncoder(encoderDepth, ...
    convFilterSize, poolSize, initialEncoderNumChannels, convolutionPadding)
encoder = [];
for stage = 1:encoderDepth
    % Double the number of channels at each stage of the encoder:
    encoderNumChannels = initialEncoderNumChannels * 2^(stage-1);

    firstConv = iCreateAndInitializeConvLayer(convFilterSize, ...
        encoderNumChannels, ['Encoder-Stage-' num2str(stage) ...
        '-Conv-1'], convolutionPadding);

    firstReLU = reluLayer('Name',['Encoder-Stage-' ...
        num2str(stage) '-ReLU-1']);

    secondConv = iCreateAndInitializeConvLayer(convFilterSize,...
        encoderNumChannels, ['Encoder-Stage-' num2str(stage) ...
        '-Conv-2'], convolutionPadding);
    secondReLU = reluLayer('Name',['Encoder-Stage-' ...
        num2str(stage) '-ReLU-2']);

    encoder = [encoder; firstConv; firstReLU; secondConv; secondReLU]; %#ok<*AGROW>

    if stage == encoderDepth
        dropOutLayer = dropoutLayer(0.5,'Name',...
            ['Encoder-Stage-' num2str(stage) '-DropOut']);
        encoder = [encoder; dropOutLayer];
    end

    maxPoolLayer = maxPooling2dLayer(poolSize, 'Stride', 2, 'Name',...
        ['Encoder-Stage-' num2str(stage) '-MaxPool']);

    encoder = [encoder; maxPoolLayer];
end
finalNumChannels = encoderNumChannels;
end


% Helper Function 2: Create the decoder.
function [decoder, finalDecoderNumChannels] = iCreateDecoder(...
    encoderDepth, upConvFilterSize, convFilterSize,...
    initialDecoderNumChannels, convolutionPadding)

decoder = [];
for stage = 1:encoderDepth
    % Halve the number of channels at each stage of the decoder:
    decoderNumChannels = initialDecoderNumChannels / 2^(stage-1);

    upConv = iCreateAndInitializeUpConvLayer(upConvFilterSize, ...
        decoderNumChannels, ['Decoder-Stage-' num2str(stage) '-UpConv']);
    upReLU = reluLayer('Name',['Decoder-Stage-' num2str(stage) '-UpReLU']);

    % In the decoder, input feature channels are skip-connected with transposed convolution feature channels:
    depthConcatLayer = depthConcatenationLayer(2, 'Name', ...
        ['Decoder-Stage-' num2str(stage) '-DepthConcatenation']);

    firstConv = iCreateAndInitializeConvLayer(convFilterSize, ...
        decoderNumChannels, ['Decoder-Stage-' num2str(stage) ...
        '-Conv-1'], convolutionPadding);
    firstReLU = reluLayer('Name',['Decoder-Stage-' ...
        num2str(stage) '-ReLU-1']);

    secondConv = iCreateAndInitializeConvLayer(convFilterSize,...
        decoderNumChannels, ['Decoder-Stage-' num2str(stage)...
        '-Conv-2'], convolutionPadding);
    secondReLU = reluLayer('Name',['Decoder-Stage-' num2str(stage) ...
        '-ReLU-2']);

    decoder = [decoder; upConv; upReLU; depthConcatLayer;...
        firstConv; firstReLU; secondConv; secondReLU];
end
finalDecoderNumChannels = decoderNumChannels;
end


% Helper Function 3: Create and initialize convolution layer.
function convLayer = iCreateAndInitializeConvLayer(convFilterSize,...
    outputNumChannels, layerName, convolutionPadding)

convLayer = convolution2dLayer(convFilterSize,outputNumChannels,...
    'Padding', convolutionPadding ,'BiasL2Factor',0, ...
    'WeightsInitializer','he',...
    'BiasInitializer','zeros', ...
    'Name',layerName);
end


% Helper Function 4: Create and initialize transposed convolution layer.
function upConvLayer = iCreateAndInitializeUpConvLayer(UpconvFilterSize,...
    outputNumChannels, layerName)

upConvLayer = transposedConv2dLayer(UpconvFilterSize, outputNumChannels,...
    'Stride',2, 'BiasL2Factor', 0, 'WeightsInitializer', 'he', ...
    'BiasInitializer', 'zeros', 'Name', layerName);

upConvLayer.BiasLearnRateFactor = 2;
end


% Helper Function 5: Create skip connections.
function lgraph = iConnectLgraph(lgraph, encoderDepth)
for depth = 1:encoderDepth
    startLayer = sprintf('Encoder-Stage-%d-ReLU-2',depth);
    endLayer = sprintf('Decoder-Stage-%d-DepthConcatenation/in2',...
        encoderDepth-depth + 1);
    lgraph = connectLayers(lgraph,startLayer, endLayer);
end
end