import torch
import torch.nn as nn
import torch.nn.functional as F


class ConvNet(nn.Module):
    def __init__(self):
        super(ConvNet, self).__init__()
        self.conv1 = nn.Conv2d(3, 16, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1)
        self.conv3 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1)
        self.fc = nn.Linear(64 * 84 * 84, 64)  # Flatten and reduce to 64 dimensions

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = x.view(x.size(0), -1)  # Flatten the output
        x = self.fc(x)
        return x


class AttentionMechanism(nn.Module):
    def __init__(self, feature_dim=64):
        super(AttentionMechanism, self).__init__()
        self.attn_layer = nn.Linear(feature_dim, 1)

    def forward(self, features):
        # Calculate attention scores
        attn_scores = torch.cat([self.attn_layer(f).unsqueeze(0) for f in features], dim=0)
        attn_scores = F.softmax(attn_scores, dim=0)  # Normalize attention scores
        # Weighted sum of the features based on attention scores
        fused_features = sum(score * feature for score, feature in zip(attn_scores.squeeze(), features))
        return fused_features


class ImageFusionModel(nn.Module):
    def __init__(self):
        super(ImageFusionModel, self).__init__()
        self.conv_net = ConvNet()  # Feature extraction network
        self.attn_mechanism = AttentionMechanism()  # Attention-based fusion

    def forward(self, img1, img2, img3):
        # Extract features for each image
        feat1 = self.conv_net(img1)
        feat2 = self.conv_net(img2)
        feat3 = self.conv_net(img3)

        # Apply attention mechanism to the extracted features
        fused_features = self.attn_mechanism([feat1, feat2, feat3])
        return fused_features


# 示例代码
if __name__ == "__main__":
    # 创建模型
    model = ImageFusionModel()

    # 假设有三张84x84x3的图像
    img1 = torch.randn(1, 3, 84, 84)  # Batch size = 1, 3 channels, 84x84 image
    img2 = torch.randn(1, 3, 84, 84)
    img3 = torch.randn(1, 3, 84, 84)

    # 获取融合后的特征向量
    fused_features = model(img1, img2, img3)
    print(fused_features.shape)  # 应该是 [1, 64]
