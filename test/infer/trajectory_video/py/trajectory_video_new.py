import json
import os
import re
import shutil
import numpy as np
import cv2
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.patches import FancyArrowPatch
from mpl_toolkits.mplot3d import proj3d

class Arrow3D(FancyArrowPatch):
    def __init__(self, xs, ys, zs, *args, **kwargs):
        FancyArrowPatch.__init__(self, (0, 0), (0, 0), *args, **kwargs)
        self._verts3d = xs, ys, zs

    def do_3d_projection(self, renderer=None):
        xs3d, ys3d, zs3d = self._verts3d
        xs, ys, zs = proj3d.proj_transform(xs3d, ys3d, zs3d, self.axes.M)
        self.set_positions((xs[0], ys[0]), (xs[1], ys[1]))
        return np.min(zs)

def draw_3d_arrow_on_image(base_dir, data_point, output_path):
    """
    在图像上绘制3D箭头和坐标轴。

    :param base_dir: JSON文件所在的基础目录。
    :param data_point: 包含state, actions和wrist_image_path的字典。
    :param output_path: 保存结果图像的路径。
    """
    # 1. 提取数据
    state = np.array(data_point['state'][:3])
    actions = np.array(data_point['actions'][:3])
    relative_image_path = data_point['wrist_image_path']
    image_path = os.path.join(base_dir, relative_image_path)

    # 2. 计算方向向量
    direction_vector = actions - state

    # 3. 加载背景图片
    if not os.path.exists(image_path):
        print(f"错误: 图像文件未找到 at {image_path}")
        return
    img = cv2.imread(image_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img_height, img_width, _ = img.shape

    # 4. 创建一个透明的3D绘图
    fig = plt.figure(figsize=(img_width / 100, img_height / 100), dpi=100)
    # 关键修改：确保figure本身是透明的
    fig.patch.set_alpha(0)

    ax = fig.add_subplot(111, projection='3d')
    # 关键修改：确保axes是透明的
    ax.patch.set_alpha(0)
    
    # 隐藏原始坐标轴平面和网格
    ax.grid(False)
    ax.xaxis.pane.fill = False
    ax.yaxis.pane.fill = False
    ax.zaxis.pane.fill = False
    ax.xaxis.line.set_color((1.0, 1.0, 1.0, 0.0))
    ax.yaxis.line.set_color((1.0, 1.0, 1.0, 0.0))
    ax.zaxis.line.set_color((1.0, 1.0, 1.0, 0.0))
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_zticks([])
    ax.set_axis_off()


    # 5. 绘制坐标轴 (Y轴朝右, Z轴朝上, X轴朝里)
    axis_length = np.linalg.norm(direction_vector) * 1.0 if np.linalg.norm(direction_vector) > 0 else 1.0
    # X轴 (黄色, 沿matplotlib的Y轴绘制, 指向屏幕内)
    ax.plot([0, 0], [0, axis_length], [0, 0], color='yellow', linewidth=2)
    ax.text(0, axis_length, 0, 'X', color='yellow')
    # Y轴 (绿色, 沿matplotlib的X轴绘制, 指向右)
    ax.plot([0, axis_length], [0, 0], [0, 0], color='g', linewidth=2)
    ax.text(axis_length, 0, 0, 'Y', color='g')
    # Z轴 (蓝色, 沿matplotlib的Z轴绘制, 指向上)
    ax.plot([0, 0], [0, 0], [0, axis_length], color='b', linewidth=2)
    ax.text(0, 0, axis_length, 'Z', color='b')

    # 6. 绘制动作箭头 (交换X和Y以匹配坐标系)
    if actions is not None and state is not None:
        direction_vector = actions - state
        arrow_scale_factor = 1  # 用于增长箭头的比例因子
        arrow = Arrow3D(
            [0, direction_vector[1] * arrow_scale_factor],  # Y-coord on matplotlib's X-axis
            [0, direction_vector[0] * arrow_scale_factor],  # X-coord on matplotlib's Y-axis
            [0, direction_vector[2] * arrow_scale_factor],  # Z-coord on matplotlib's Z-axis
            mutation_scale=15,
            lw=2,
            arrowstyle="-|>",
            color="red"
        )
        ax.add_artist(arrow)

    # 调整视角以匹配要求：Y向右, Z向上, X向里
    ax.view_init(elev=5, azim=-90)
    
    # 将3D绘图渲染到numpy数组
    # 关键修改：使用 savefig 获取更可靠的带透明通道的图像
    from io import BytesIO
    buf = BytesIO()
    fig.savefig(buf, format='png', transparent=True, bbox_inches='tight', pad_inches=0)
    buf.seek(0)
    arrow_img_rgba = cv2.imdecode(np.frombuffer(buf.getvalue(), np.uint8), cv2.IMREAD_UNCHANGED)
    buf.close()

    # 关闭matplotlib图像以释放内存
    plt.close(fig)

    # 7. 合成图像
    # 调整箭头/坐标轴图像大小以匹配背景
    arrow_img_resized = cv2.resize(arrow_img_rgba, (img_width, img_height))
    
    # 分离 BGR 和 Alpha 通道
    arrow_bgr = arrow_img_resized[:, :, :3]
    alpha_channel = arrow_img_resized[:, :, 3]

    # 将 Alpha 通道作为蒙版
    # 为了进行混合，我们需要将 alpha 蒙版扩展到3个通道
    alpha_mask = cv2.merge([alpha_channel, alpha_channel, alpha_channel]) / 255.0
    
    # 将原始图像从 RGB 转换为 BGR 以便与 OpenCV 兼容
    img_bgr = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)

    # 使用 Alpha 混合将箭头叠加到背景上
    # 公式: output = foreground * alpha + background * (1 - alpha)
    combined_img = (arrow_bgr * alpha_mask + img_bgr * (1.0 - alpha_mask)).astype(np.uint8)

    # 8. 保存结果
    cv2.imwrite(output_path, combined_img)
    print(f"图像已成功保存到: {output_path}")


def create_video_from_images(image_folder, video_name, fps=30):
    """
    从文件夹中的图像序列创建一个视频，并在每帧上显示编号。

    :param image_folder: 包含图像的文件夹路径。
    :param video_name: 输出视频文件的路径 (e.g., 'output.mp4')。
    :param fps: 视频的帧率。
    """
    all_files = [f for f in os.listdir(image_folder) if f.endswith((".png", ".jpg", ".jpeg"))]
    
    # 自定义排序函数，按文件名中的数字排序
    def sort_key(filename):
        # 假设文件名格式为 '..._frame_NUM_with_arrow.png'
        try:
            # 提取文件名中的数字部分
            parts = filename.split('_')
            # 找到 'frame' 后面的数字
            frame_index = parts.index('frame')
            return int(parts[frame_index + 1])
        except (ValueError, IndexError):
            # 如果格式不匹配，返回一个默认值以便排序
            return -1

    images = sorted(all_files, key=sort_key)

    if not images:
        print(f"在目录 {image_folder} 中没有找到图像文件。")
        return

    # 读取第一张图片以获取视频尺寸
    frame = cv2.imread(os.path.join(image_folder, images[0]))
    if frame is None:
        print(f"无法读取第一张图片: {images[0]}")
        return
    height, width, layers = frame.shape

    # 定义视频编码器和创建VideoWriter对象
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    video = cv2.VideoWriter(video_name, fourcc, fps, (width, height))

    print(f"正在创建视频，共 {len(images)} 帧...")
    # 遍历所有图片并写入视频
    for i, image in enumerate(images):
        img_path = os.path.join(image_folder, image)
        current_frame = cv2.imread(img_path)

        # 检查图像是否成功加载
        if current_frame is None:
            print(f"警告: 无法读取帧 {i} 的图片: {img_path}。跳过此帧。")
            continue # 跳过损坏或无法读取的帧
        
        frame = current_frame

        # 在帧上添加文本（帧编号）
        frame_number_text = f"Frame: {i}"
        font = cv2.FONT_HERSHEY_SIMPLEX
        position = (10, 50)  # 文本位置 (x, y)
        font_scale = 1.5  # 字体大小
        font_color = (255, 255, 255)  # 颜色 (BGR: 白色)
        thickness = 2  # 线条粗细
        cv2.putText(frame, frame_number_text, position, font, font_scale, font_color, thickness, cv2.LINE_AA)

        video.write(frame)

    # 释放VideoWriter对象
    video.release()
    print(f"视频已成功保存到: {video_name}")


def generate_video_from_trajectory(json_path):
    """
    从给定的轨迹JSON文件生成一个视频。

    :param json_path: 输入的 inferred_trajectory JSON 文件的路径。
    """
    # 定义文件和目录路径
    base_directory = '/home/adminroot/lxx/openpi/code/openpi/test/infer/trajectory_video/video'
    
    # 从json文件名动态生成输出目录和视频文件名
    episode_match = re.search(r'_(\d+)\.json$', os.path.basename(json_path))
    if episode_match:
        episode_num = episode_match.group(1)
        output_dir_name = f'image_ep{episode_num}_with_arrow'
        video_filename = f'trajectory_video_ep{episode_num}.mp4'
    else:
        # 如果没有匹配到数字，则使用默认名称
        output_dir_name = 'image_ep_with_arrow'
        video_filename = 'trajectory_video.mp4'

    output_dir = os.path.join(base_directory, output_dir_name)
    
    # 清理并创建输出目录
    if os.path.exists(output_dir):
        shutil.rmtree(output_dir)
    os.makedirs(output_dir)

    # 确保JSON文件存在
    if not os.path.exists(json_path):
        print(f"错误: JSON文件未找到 at {json_path}")
        return
    
    # 读取JSON数据
    with open(json_path, 'r') as f:
        episode_data = json.load(f)

    # 遍历所有数据点
    if episode_data:
        for i, frame_data in enumerate(episode_data):
            print(f"正在处理第 {i+1}/{len(episode_data)} 帧...")
            
            if frame_data.get('actions') is None:
                print(f"警告: 第 {i+1} 帧数据中缺少 'actions'。跳过箭头绘制。")

            original_relative_path = frame_data.get('wrist_image_path')
            if not original_relative_path:
                print(f"警告: 第 {i+1} 帧数据中缺少 'wrist_image_path'。")
                continue

            base_name, ext = os.path.splitext(os.path.basename(original_relative_path))
            output_filename = f"{base_name}_with_arrow{ext}"
            output_filepath = os.path.join(output_dir, output_filename)

            draw_3d_arrow_on_image(base_directory, frame_data, output_filepath)

        # 所有图像处理完毕后，创建视频
        video_output_path = os.path.join(base_directory, video_filename)
        create_video_from_images(output_dir, video_output_path, fps=10)
    else:
        print("错误: JSON文件为空。")


if __name__ == '__main__':
    # 为了独立执行，使用一个默认的JSON文件路径
    default_json_path = '/home/adminroot/lxx/openpi/code/openpi/test/infer/trajectory_video/inferred_trajectory.json'
    generate_video_from_trajectory(default_json_path)
