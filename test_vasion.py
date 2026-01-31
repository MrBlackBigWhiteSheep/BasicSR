import os
import cv2
import torch
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE  # 需要安装 scikit-learn
import matplotlib.patches as mpatches

# ==============================================================================
# 1. 导入你的模型定义
# ==============================================================================
# 假设你的模型文件在 basicsr/archs/mambawater_arch.py
# 如果路径不同，请修改这里的 import
try:
    from basicsr.archs.mambawater_arch import MambaWater
except ImportError:
    print("错误: 找不到 MambaWater 定义。请确保 basicsr.archs.mambawater_arch 在 Python 路径中，或者修改 import 路径。")
    exit(1)

# ==============================================================================
# 2. 工具类：Hook 管理器 (负责提取中间层)
# ==============================================================================
class FeatureExtractor:
    """
    修正版：不再强制要求 __init__ 传入 model，兼容旧代码
    """
    def __init__(self):  # <--- 改回了这里，不需要传参数了
        self.hooks = []
        self.features = {}

    def _hook_fn(self, name):
        def hook(model, input, output):
            # 兼容性处理：无论输出是 Tensor 还是 Tuple，都只取 Tensor
            if isinstance(output, torch.Tensor):
                out = output
            elif isinstance(output, tuple):
                out = output[0] 
            else:
                # 遇到无法处理的类型，跳过
                return

            # detach 并转到 CPU，防止显存爆炸
            self.features[name] = out.detach().cpu()
        return hook

    def register(self, layer_object, layer_name):
        """
        注册 Hook 到指定层
        """
        handle = layer_object.register_forward_hook(self._hook_fn(layer_name))
        self.hooks.append(handle)
        # print(f"[System] Hook 已注册到: {layer_name}")

    def get(self, name):
        if name not in self.features:
            raise KeyError(f"Feature '{name}' not found. 请确保模型已运行且 Hook 名称正确。")
        return self.features[name]

    def remove_all(self):
        for handle in self.hooks:
            handle.remove()
        self.hooks = []
        self.features = {}
        # print("[System] 所有 Hook 已移除。")

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.remove_all()

# ==============================================================================
# 3. 工具函数：可视化绘图 (Overlay & t-SNE)
# ==============================================================================
def plot_overlay_heatmap(img_rgb, group_indices, num_groups, save_path=None):
    """
    绘制半透明叠加图，显示退化分组的空间分布。
    """
    H, W = group_indices.shape
    
    # 使用区分度高的色图
    cmap = plt.get_cmap('nipy_spectral', num_groups)
    
    # 归一化索引并映射颜色
    mask_color = cmap(group_indices / (num_groups - 1)) # [H, W, 4] RGBA
    mask_rgb = mask_color[..., :3] # 取 RGB
    
    # Alpha Blending (原图 60% + Mask 40%)
    alpha = 0.4
    overlay = (1 - alpha) * img_rgb + alpha * mask_rgb
    overlay = np.clip(overlay, 0, 1)

    plt.figure(figsize=(12, 6))
    
    # 子图 1: 原图
    plt.subplot(1, 2, 1)
    plt.imshow(img_rgb)
    plt.title("Input Image")
    plt.axis('off')

    # 子图 2: 叠加图
    ax = plt.subplot(1, 2, 2)  #以此获取当前的 ax 对象
    plt.imshow(overlay)
    plt.title(f"Degradation Grouping Overlay\n(Num Groups: {num_groups})")
    plt.axis('off')
    
    # 添加一个 Colorbar 指示 Group ID
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=plt.Normalize(vmin=0, vmax=num_groups-1))
    sm.set_array([])
    
    # 【修复点】：添加 ax=ax 参数，告诉 plt 从当前子图偷空间
    plt.colorbar(sm, ax=ax, label='Group Index', fraction=0.046, pad=0.04)

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"[Output] 叠加图已保存至: {save_path}")
    plt.show()

def plot_discrete_clusters(img_rgb, group_indices, num_groups, save_path=None):
    plt.figure(figsize=(12, 6))
    
    # 左侧：原图
    plt.subplot(1, 2, 1)
    plt.imshow(img_rgb)
    plt.title("Input Image")
    plt.axis('off')

    # 右侧：纯离散色块
    plt.subplot(1, 2, 2)
    # 使用 tab20 或 nipy_spectral，强制使用最近邻插值展示边界
    plt.imshow(group_indices, cmap='tab20', interpolation='nearest')
    plt.title(f"Degradation Groups (Discrete)")
    plt.axis('off')
    
    if save_path:
        plt.savefig(save_path.replace('.png', '_discrete.png'), dpi=300, bbox_inches='tight')
    plt.show()

def plot_contour_groups(img_rgb, group_indices, num_groups, save_path=None):
    plt.figure(figsize=(10, 10))
    plt.imshow(img_rgb)
    
    # 为每个组画出边缘
    # levels 指定分组的界限
    plt.contour(group_indices, levels=np.arange(num_groups), colors='white', linewidths=0.5, alpha=0.8)
    
    # 可选：用极低透明度填充颜色
    plt.imshow(group_indices, cmap='nipy_spectral', alpha=0.2)
    
    plt.title("Grouping Boundaries on Original Image")
    plt.axis('off')
    
    if save_path:
        plt.savefig(save_path.replace('.png', '_contour.png'), dpi=300, bbox_inches='tight')
    plt.show()

def plot_top_groups(img_rgb, group_indices, num_tokens, save_path=None):
    # 统计出现频率最高的 3 个组
    unique, counts = np.unique(group_indices, return_counts=True)
    top_groups = unique[np.argsort(counts)[-3:]] 

    fig, axes = plt.subplots(1, 4, figsize=(20, 5))
    axes[0].imshow(img_rgb)
    axes[0].set_title("Input Image")
    
    for i, g_id in enumerate(top_groups):
        # 创建一个遮罩，属于该组的显示原色，不属于的变暗或变灰
        mask = (group_indices == g_id).astype(float)[..., np.newaxis]
        highlight = img_rgb * mask + (img_rgb * 0.2 * (1 - mask)) # 非选定区域变暗
        
        axes[i+1].imshow(highlight.astype(np.uint8) if highlight.max() > 1 else highlight)
        axes[i+1].set_title(f"Degradation Pattern {i+1}")
        
    for ax in axes: ax.axis('off')
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path.replace('.png', '_patterns.png'), dpi=300)
    plt.show()

def plot_tsne_clusters(features, group_indices, max_points=3000, save_path=None):
    """
    绘制 t-SNE 散点图，显示退化特征的流形分布。
    features: [N, C]
    group_indices: [N]
    """
    print("[System] 正在运行 t-SNE 降维 (可能需要几秒钟)...")
    N, C = features.shape
    
    # 如果点太多，进行随机采样以加快速度
    if N > max_points:
        idx = np.random.choice(N, max_points, replace=False)
        features_sample = features[idx]
        groups_sample = group_indices[idx]
    else:
        features_sample = features
        groups_sample = group_indices
    
    # 运行 t-SNE
    tsne = TSNE(n_components=2, perplexity=30, init='pca', random_state=42, learning_rate='auto')
    features_2d = tsne.fit_transform(features_sample)
    
    plt.figure(figsize=(8, 8))
    scatter = plt.scatter(
        features_2d[:, 0], features_2d[:, 1], 
        c=groups_sample, 
        cmap='nipy_spectral', 
        s=15, 
        alpha=0.7, 
        edgecolors='none'
    )
    
    plt.colorbar(scatter, label='Group ID')
    plt.title("t-SNE Visualization of Degradation Features")
    plt.xlabel("Dim 1")
    plt.ylabel("Dim 2")
    
    # 去除顶部和右侧边框
    plt.gca().spines['top'].set_visible(False)
    plt.gca().spines['right'].set_visible(False)
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"[Output] t-SNE图已保存至: {save_path}")
    plt.show()

def plot_pure_degradation_map(deg_feature, save_path=None):
    """
    展示提取出的退化特征场。
    deg_feature: [C, H, W]
    """
    # 使用 PCA 将多通道特征降维到 1，最能体现浓淡变化
    feat_np = deg_feature.numpy().transpose(1, 2, 0)
    h, w, c = feat_np.shape
    from sklearn.decomposition import PCA
    pca = PCA(n_components=1)
    flat_feat = pca.fit_transform(feat_np.reshape(-1, c))
    pca_map = flat_feat.reshape(h, w)
    
    plt.figure(figsize=(6, 5))
    plt.imshow(pca_map, cmap='magma') # magma/viridis 非常适合表现物理场
    plt.title("Extracted Degradation Field (Standardized)")
    plt.axis('off')
    plt.colorbar(fraction=0.046, pad=0.04)
    if save_path: plt.savefig(save_path)

def plot_semantic_grouping(img_rgb, group_indices, num_actual_groups=4, save_path=None):
    """
    对 64 个 Token 进行二次聚类，展示大尺度的物理退化分区
    """
    from sklearn.cluster import KMeans
    h, w = group_indices.shape
    # 对 Group ID 进行聚类（或者对 Token 权重聚类更准）
    # 这里为了演示，简单处理
    flat_indices = group_indices.reshape(-1, 1)
    kmeans = KMeans(n_clusters=num_actual_groups, random_state=42).fit(flat_indices)
    new_labels = kmeans.labels_.reshape(h, w)

    fig, ax = plt.subplots(1, 1, figsize=(8, 8))
    # 用简单的离散色卡
    ax.imshow(img_rgb)
    # 用 Contourf 绘制平滑的填充区域
    ax.contourf(new_labels, levels=num_actual_groups, alpha=0.3, cmap='Set1')
    ax.set_title(f"Clustered Degradation Zones (k={num_actual_groups})")
    ax.axis('off')
    if save_path: plt.savefig(save_path)

def plot_mamba_path(group_indices, save_path=None):
    """
    可视化扫描路径。如果排序正确，路径应该顺着相同的退化区域走，而不是乱跳。
    """
    h, w = group_indices.shape
    # 获取排序后的坐标（简化逻辑）
    flat_indices = group_indices.flatten()
    sort_idx = np.argsort(flat_indices)
    
    # 取前 100 个像素点，画出连线
    y, x = np.unravel_index(sort_idx[:100], (h, w))
    
    plt.figure(figsize=(8, 8))
    plt.scatter(x, y, c=np.arange(len(x)), cmap='jet', s=20)
    plt.plot(x, y, color='black', alpha=0.3, linewidth=1)
    plt.title("Mamba Scanning Path (Top 100 Pixels)")
    plt.gca().invert_yaxis()
    if save_path: plt.savefig(save_path)

def plot_mamba_path_v2(img_rgb, group_indices, save_path=None):
    """
    分段可视化：展示 Mamba 在不同时间段是如何聚焦于不同语义区域的。
    """
    H, W = group_indices.shape
    flat_groups = group_indices.flatten()
    
    # 1. 获取完整的扫描顺序
    scan_order = np.argsort(flat_groups, kind='stable') 
    total_pixels = len(scan_order)
    
    plt.figure(figsize=(12, 12))
    plt.imshow(img_rgb, alpha=0.6) # 背景调淡，突出路径
    
    # ==========================================================
    # 定义我们要观察的三个时间段 (Segments)
    # 既然是按 Group 排序，那么：
    # - 早期 (Early): 对应 Group ID 较小的区域 (e.g., 背景/深水)
    # - 中期 (Mid):   对应 Group ID 中间的区域
    # - 晚期 (Late):  对应 Group ID 较大的区域 (e.g., 近景/高光)
    # ==========================================================
    
    # 我们取 3 段，每段连续取 150 个像素，这样能看到密集的跳跃
    segment_len = 150
    
    # 为了避免取到只有1-2个像素的边缘组，我们取百分比位置
    starts = [
        int(total_pixels * 0.10), # 10% 处 (早期)
        int(total_pixels * 0.50), # 50% 处 (中期)
        int(total_pixels * 0.90)  # 90% 处 (晚期)
    ]
    
    # 定义三段的颜色和标签
    configs = [
        {'color': 'cyan',   'label': 'Stage 1: Deep Water (Early)'},
        {'color': 'yellow', 'label': 'Stage 2: Mid-Range (Middle)'},
        {'color': 'red',    'label': 'Stage 3: Highlights (Late)'}
    ]

    for i, start_idx in enumerate(starts):
        # 取出一小段连续的序列
        end_idx = start_idx + segment_len
        segment_pixels = scan_order[start_idx:end_idx]
        
        # 映射回 2D
        y_coords = segment_pixels // W
        x_coords = segment_pixels % W
        
        # 绘图配置
        c = configs[i]['color']
        lbl = configs[i]['label']
        
        # 1. 画散点
        plt.scatter(x_coords, y_coords, color=c, s=40, edgecolors='white', zorder=5, label=lbl)
        
        # 2. 画连线 (展示它是如何跳跃的)
        # alpha 设置得高一点，让人看清这是连续的动作
        plt.plot(x_coords, y_coords, color=c, linewidth=1.5, alpha=0.8, zorder=4)
        
        # 3. 标注起点，方便读者理解方向
        plt.text(x_coords[0], y_coords[0], f"Start-{i+1}", color='white', 
                 fontweight='bold', bbox=dict(facecolor=c, alpha=0.8, edgecolor='none'))

    plt.legend(loc='lower right', fontsize=12)
    plt.title("Mamba Scanning Dynamics: Grouping by Degradation Similarity\n(Note how the path clusters spatially disjoint but semantically similar regions)", fontsize=14)
    plt.axis('off')
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()
# ==============================================================================
# 4. 辅助函数：模型加载与预处理
# ==============================================================================
def load_basicsr_checkpoint(model, ckpt_path, device):
    """加载 BasicSR 格式的权重，处理 DDP 前缀"""
    checkpoint = torch.load(ckpt_path, map_location=device)
    # BasicSR 通常把权重放在 'params' 键下
    if 'params' in checkpoint:
        state_dict = checkpoint['params']
    else:
        state_dict = checkpoint

    # 移除 'module.' 前缀
    new_state_dict = {}
    for k, v in state_dict.items():
        name = k.replace('module.', '')
        new_state_dict[name] = v
    
    model.load_state_dict(new_state_dict, strict=True)
    model.to(device)
    model.eval()
    print(f"[System] 权重已加载: {ckpt_path}")
    return model

def preprocess_image(img_path, window_size, device):
    """读取、归一化、Padding"""
    img = cv2.imread(img_path)
    if img is None:
        raise ValueError(f"无法读取图片: {img_path}")
    
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    h_orig, w_orig = img.shape[:2]
    
    # 归一化 [0, 1] 并转 Tensor
    img_tensor = torch.from_numpy(img.transpose(2, 0, 1)).float() / 255.0
    img_tensor = img_tensor.unsqueeze(0).to(device) # [1, 3, H, W]
    
    # Padding 确保能被 window_size 整除
    pad_h = (window_size - h_orig % window_size) % window_size
    pad_w = (window_size - w_orig % window_size) % window_size
    img_padded = F.pad(img_tensor, (0, pad_w, 0, pad_h), mode='reflect')
    
    return img, img_padded, (h_orig, w_orig)

# ==============================================================================
# 5. 主程序入口
# ==============================================================================
def main():
    # ------------------------------------------------------------------
    # 配置区域 (请在此处修改参数)
    # ------------------------------------------------------------------
    CKPT_PATH = '/data/wa/work/BasicSR/experiments/MambaWater_UIEB_enhance0102_5/models/net_g_84200.pth' # 权重路径
    INPUT_IMG = '/data/wa/dataset/UIEB_09/reference-800/6_img_.png'                   # 测试图片路径
    OUTPUT_DIR = '/data/wa/work/BasicSR/vis_groups/3'                            # 结果保存目录
    
    # 模型参数 (必须与训练配置一致)
    MODEL_CONFIG = dict(
        img_size=128,
        embed_dim=48,
        d_state=16,
        depths=[2, 4, 4, 2],
        num_heads=[4, 4, 4, 4],
        window_size=16,
        inner_rank=32,
        num_tokens=64,           # 分组数量
        convffn_kernel_size=3,
        mlp_ratio=1.5,
        upsampler=None,          # 图像增强通常为None
        resi_connection='1conv'
    )
    
    # 想要观察哪一层? (Stage 0-3, Block index)
    # 建议选 Stage 1 或 2，语义更丰富
    TARGET_STAGE = 1
    TARGET_BLOCK = 0
    # ------------------------------------------------------------------

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # 1. 初始化模型
    print("[System] 初始化模型...")
    model = MambaWater(**MODEL_CONFIG)
    model = load_basicsr_checkpoint(model, CKPT_PATH, device)

    # 2. 预处理图片
    raw_img_rgb, input_tensor, (orig_h, orig_w) = preprocess_image(
        INPUT_IMG, MODEL_CONFIG['window_size'], device
    )

    # 3. 自动遍历所有层级进行可视化
    print(f"[System] 开始自动遍历所有层级...")
    
    for s_idx, stage in enumerate(model.layers):
        # 根据你的模型结构，确认 blocks 的位置
        # 假设路径是 stage.residual_group.layers
        blocks = stage.residual_group.layers
        
        for b_idx, block in enumerate(blocks):
            print(f"\n>>> 处理 Stage {s_idx}, Block {b_idx}...")
            
            # 使用 FeatureExtractor 提取当前 Block 的数据
            with FeatureExtractor() as extractor:
                target_assm = block.assm
                
                # 注册当前 Block 的探针
                extractor.register(target_assm.vis_probe_deg, 'deg_feature') 
                extractor.register(target_assm.vis_probe_cls, 'cluster_policy')

                # 推理一次
                with torch.no_grad():
                    model(input_tensor)

                # 获取数据
                policy = extractor.get('cluster_policy') # [B, N, num_tokens]
                feats = extractor.get('deg_feature')     # [B, C, H_pad, W_pad]

            # --- 数据后处理 ---
            group_indices_flat = torch.argmax(policy, dim=-1).squeeze(0).numpy()
            _, _, H_pad, W_pad = input_tensor.shape
            group_map = group_indices_flat.reshape(H_pad, W_pad)
            group_map_cropped = group_map[:orig_h, :orig_w]

            C = feats.shape[1]
            feats_2d = feats.permute(0, 2, 3, 1).squeeze(0).numpy()
            feats_cropped = feats_2d[:orig_h, :orig_w, :]
            feats_ready_for_tsne = feats_cropped.reshape(-1, C)
            groups_ready_for_tsne = group_map_cropped.reshape(-1)

            # --- 自动保存 ---
            base_name = os.path.splitext(os.path.basename(INPUT_IMG))[0]
            tag = f"S{s_idx}_B{b_idx}"
            
            # # 画叠加图
            # plot_overlay_heatmap(
            #     raw_img_rgb, 
            #     group_map_cropped, 
            #     num_groups=MODEL_CONFIG['num_tokens'],
            #     save_path=os.path.join(OUTPUT_DIR, f"{base_name}_{tag}_overlay.png")
            # )
            
            # plot_discrete_clusters(
            #     raw_img_rgb, 
            #     group_map_cropped, 
            #     num_groups=MODEL_CONFIG['num_tokens'],
            #     save_path=os.path.join(OUTPUT_DIR, f"{base_name}_{tag}_discrete.png")
            # )

            # plot_contour_groups(
            #     raw_img_rgb, 
            #     group_map_cropped, 
            #     num_groups=MODEL_CONFIG['num_tokens'],
            #     save_path=os.path.join(OUTPUT_DIR, f"{base_name}_{tag}_contour.png")
            # )

            # plot_top_groups(
            #     raw_img_rgb, 
            #     group_map_cropped, 
            #     num_tokens=MODEL_CONFIG['num_tokens'],
            #     save_path=os.path.join(OUTPUT_DIR, f"{base_name}_{tag}_patterns.png")
            # )

            # # 画 t-SNE 图 (可选，如果层数太多，t-SNE 会比较耗时，可以根据需要注释掉)
            # plot_tsne_clusters(
            #     feats_ready_for_tsne,
            #     groups_ready_for_tsne,
            #     max_points=2000, 
            #     save_path=os.path.join(OUTPUT_DIR, f"{base_name}_{tag}_tsne.png")
            # )
            # plot_pure_degradation_map(
            #     torch.from_numpy(feats_cropped.transpose(2, 0, 1)), # [C, H, W]
            #     save_path=os.path.join(OUTPUT_DIR, f"{base_name}_{tag}_degmap.png")
            # )
            # plot_semantic_grouping(
            #     raw_img_rgb, 
            #     group_map_cropped, 
            #     num_actual_groups=4,
            #     save_path=os.path.join(OUTPUT_DIR, f"{base_name}_{tag}_semantic.png")
            # )
            # plot_mamba_path(
            #     group_map_cropped,
            #     save_path=os.path.join(OUTPUT_DIR, f"{base_name}_{tag}_mambapath.png")
            # )
            # plot_mamba_path_v2(
            #     raw_img_rgb, 
            #     group_map_cropped,
            #     save_path=os.path.join(OUTPUT_DIR, f"{base_name}_{tag}_mambapath_v2.png")
            #     )

    print("\n[System] 所有层级可视化完成！请前往输出目录查看。")

if __name__ == '__main__':
    main()