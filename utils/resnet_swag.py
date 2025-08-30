import torch

def get_features(image, model, layers=None):

    if layers is None:
        layers1 = {'0': 'conv1_0',
                    '1': 'conv1_1',
                    '2': 'conv1_2'
                    }
        layers2 = {'0': 'conv2_0',
                    '1': 'conv2_1',
                    '2': 'conv2_2',
                    '3': 'conv2_3'
                    }
        layers3 = {'0': 'conv3_0',
                    '1': 'conv3_1',
                    '2': 'conv3_2',
                    '3': 'conv3_3',
                    '4': 'conv3_4',
                    '5': 'conv3_5'
                    }

    alpha = 0.001
    features = {}
    x = image
    x = model.conv1(x)

    x = model.bn1(x)
    x = model.relu(x)
    features['conv0_0'] = x
    x = model.maxpool(x)
    features['conv0_1'] = x

    for name, layer in enumerate(model.layer1):
        x = layer(x)
        if str(name) in layers1:
            features[layers1[str(name)]] = x
    for name, layer in enumerate(model.layer2):
        x = layer(x)
        if str(name) in layers2:
            features[layers2[str(name)]] = x
    for name, layer in enumerate(model.layer3):
        x = layer(x)
        if str(name) in layers3:
            features[layers3[str(name)]] = alpha * x
            
    return features

def content_loss(content, target, model, content_layer_weights = 'conv3_5'):
    # torch.Size([1, 3, 512, 512])
    content_features = get_features(content, model)
    # print(content_features)
    target_features = get_features(target, model)
    
    content_loss = torch.mean((target_features[content_layer_weights] -
                                       content_features[content_layer_weights]) ** 2)
    return content_loss