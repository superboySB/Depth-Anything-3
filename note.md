Depth Anything 3 项目笔记

## 运行环境检查（保持现状）
- python3 --version：3.10.12；pip：25.3。
- 已装（保持不动）：torch 2.5.1+cu118、torchvision 0.20.1+cu118、torchaudio 2.5.0+cu118、isaaclab 0.36.21、isaaclab-rl 0.1.4、isaaclab-tasks 0.10.31、fastapi 0.110.0、uvicorn 0.29.0、einops 0.8.1、huggingface-hub 0.23.4、imageio 2.22.2、numpy 1.26.4、opencv-python 4.11.0.86、pillow 11.0.0、omegaconf 2.3.0、trimesh 4.10.0、safetensors 0.7.0 等。
- 缺失（只补缺，不改已有版本）：moviepy==1.0.3、addict、plyfile、pycolmap、evo、e3nn（我已验证只需要这些包即可跑 `test.py` 的深度/GLB 导出，其他依赖保持原状）。

## 安装原则
- 不触碰现有的 torch/torchvision/torchaudio/pillow 版本，也不升级 isaaclab 系列包；如果 pip 提示将要卸载/升级这些包，立刻 Ctrl+C 退出。
- 不再往 `pyproject.toml` 添加新依赖，也**不要运行** `pip install -e .`；只补齐下面列出的少量三方包。
- 任何联网安装前可选择先备份一次包列表（方便回滚）：
  ```bash
  ${ISAACLAB_PATH}/isaaclab.sh -p -m pip list --format=freeze > /tmp/da3-freeze-before.txt
  ```
- 之后的安装命令一律加 `--upgrade-strategy only-if-needed`，避免 pip 主动升降已有包。

## 安装流程（增量补齐，不破坏基础环境）
1. **进入仓库**：`cd /workspace/Depth-Anything-3`。
2. **一次性补齐必需三方包**（联网执行，若看到 pip 计划操作 torch/pillow/isaaclab 立即终止）：
   ```bash
   ${ISAACLAB_PATH}/isaaclab.sh -p -m pip install --upgrade-strategy only-if-needed \
       moviepy==1.0.3 addict plyfile pycolmap evo e3nn
   ```
3. **校验未被动升级**（可选）：
   ```bash
   ${ISAACLAB_PATH}/isaaclab.sh -p -m pip show torch torchvision torchaudio pillow isaaclab | grep -E "^(Name|Version)"
   ${ISAACLAB_PATH}/isaaclab.sh -p -m pip check
   ```
   - 若 `pip check` 仅报告 isaaclab 与 pillow/torch 的“版本不匹配”提示，说明旧版本仍然在；一旦真的被升级，可用备份文件快速回滚：
     ```bash
     ${ISAACLAB_PATH}/isaaclab.sh -p -m pip install --no-deps -r /tmp/da3-freeze-before.txt
     ```

## 可选功能（确保基础环境稳定后再决定是否联网）
- Gradio 前端：`pip install -e "[app]"`
- 高斯分支导出 `gsplat`：`pip install --no-build-isolation git+https://github.com/nerfstudio-project/gsplat.git@0b4dddf04cb687367602c01196913cde6a743d70`
- 如遇 HF 访问问题，可先设置镜像：`export HF_ENDPOINT=https://hf-mirror.com`。

## Notebook / test.py 快速联调
- 仿照 `notebooks/da3.ipynb` 制作了 `test.py`，默认跑 `assets/examples/SOH/000.png` 与 `010.png` 并导出 GLB + 深度可视化。可用 `--export-format none` 纯跑内存，或改 `--images` 指定自定义输入。
- 运行示例：
  ```bash
  ${ISAACLAB_PATH}/isaaclab.sh -p -m python test.py \
      --model-name depth-anything/DA3NESTED-GIANT-LARGE \
      --device cuda \
      --export-dir workspace/test_output
  ```
- 该脚本等价于 notebook 里的步骤：加载模型 → 推理 → 打印 depth/pose 形状 → 调用 `visualize_depth` 保存 PNG。可通过 `--feat-layers 0,5,10` 同时测试 `feat_vis` 导出链路。

## 文档要点速记（docs/*.md）
- **API.md**
  - `DepthAnything3` 支持 `model_name` 选择多种预设（giant/large/base/small/metric/mono/nested）；`inference` 接受 `image`（路径/PIL/numpy）、可选外参/内参、`align_to_input_ext_scale`、`infer_gs`、`use_ray_pose`、`ref_view_strategy` 等高级参数。
  - 导出链路：`export_format` 可用 `-` 组合（`mini_npz/npz/glb/depth_vis/feat_vis/gs_ply/gs_video` 等），GLB 关联 `conf_thresh_percentile/num_max_points/show_cameras`，GS 系列需 `infer_gs=True` 且模型含 GS 分支（da3-giant 或 nested）。
  - `export_kwargs` 允许对单个导出器（如 `gs_ply`, `gs_video`）单独透传参数；`render_exts/ixts/hw` 控制 gs_video 的新视角。
  - `Prediction` 对象字段：`depth/conf/extrinsics/intrinsics/processed_images/aux`（含 `feat_layer_{idx}`、`gaussians` 等）。多图场景默认运行参考视角选择（详见 ref_view 文档）。
- **CLI.md**
  - `da3` 子命令：`auto/image/images/video/colmap/backend/gradio/gallery`。所有模式都可通过 `--export-dir`（默认 `debug`）、`--export-format`、`--process-res`、`--export-feat`、`--use-backend`、`--ref-view-strategy`、GLB/feat_vis 相关参数定制。
  - `auto` 会自动判断输入类型（单图/目录/视频/COLMAP）并派发；`images` 可用 `--image-extensions` 控制后缀，`video` 支持 `--fps` 抽帧。
  - `colmap` 支持 `--sparse-subdir`、`--align-to-input-ext-scale` 做度量对齐；`backend` 持久驻留模型并提供 `/status` `/dashboard` `/gallery`；`gradio`/`gallery` 用于前端交互与查看结果，记得 `--workspace-dir` 与 `--gallery-dir`。
  - 常用工作流：`da3 backend` → 多场景循环 `da3 auto ... --use-backend --auto-cleanup` → `da3 gallery` 浏览；或结合 `--export-format mini_npz-glb-feat_vis` 生成多种资产。
- **funcs/ref_view_strategy.md**
  - 参考视角仅当视图数 ≥3 时触发。`saddle_balanced` 默认兼顾相似度/模长/方差；`saddle_sim_range` 偏好信息量大的“鞍点”；`middle` 适合视频顺序帧；`first` 仅在已人工排序或调试时使用。
  - `da3 video --ref-view-strategy middle` 能让长序列更稳定，`da3 images --ref-view-strategy saddle_sim_range` 适合大基线多视图。

## 代码结构与关键模块
- `src/depth_anything_3/api.py`：核心 API `DepthAnything3`，加载 YAML 配置构建模型、调用输入预处理、模型推理、姿态对齐、结果导出。`inference` 支持多视图/可选外参与尺度对齐、`use_ray_pose`、`infer_gs`、多种导出格式。
- `src/depth_anything_3/cli.py`：Typer 命令行入口 `da3`，子命令 `auto/image/images/video/colmap/backend/gradio/gallery`，默认模型 `depth-anything/DA3NESTED-GIANT-LARGE`，支持后台复用、特征导出、GLB 点数/置信度阈值调整等。
- `src/depth_anything_3/services/`：
  - `backend.py`：FastAPI 后端，常驻 GPU 模型、任务队列、显存估计/清理、REST API，并可挂载 gallery。
  - `inference_service.py`：统一本地/远端推理封装，供 CLI 调用。
  - `input_handlers.py`：根据输入类型（单图/目录/视频/COLMAP）调度处理；`gallery.py` 为静态 3D 浏览服务器。
- `src/depth_anything_3/utils/`：
  - `io/input_processor.py` 并行读取/resize/裁剪并 ImageNet 归一化，`output_processor.py` 将模型输出转为 numpy `Prediction`。
  - `export/` 下实现 `mini_npz`、`npz`、`glb`（点云+相机）、`depth_vis`、`feat_vis`、`gs_ply`、`gs_video`、`colmap` 等导出。
  - 其他：几何/对齐工具（`geometry.py`、`alignment.py`、`pose_align.py`、`ray_utils.py`）、注册表、日志、GPU 内存工具等。
- `src/depth_anything_3/model/`：
  - `da3.py` 定义 `DepthAnything3Net`（DINOv2 主干 + DualDPT 深度头 + 可选相机 encoder/decoder + 可选 GS head）以及 `NestedDepthAnything3Net`（任意视角分支 + 度量分支，对天空掩膜与尺度对齐）。
  - `configs/*.yaml`：模型预设（da3-giant/large/base/small、da3metric-large、da3mono-large、da3nested-giant-large）。
  - `dinov2/` Transformer 实现（RoPE、qk norm、alt_start），`dpt/dualdpt` 深度解码，`cam_enc`/`cam_dec` 姿态估计，`gsdpt`/`gs_adapter` 高斯分支，`reference_view_selector` 选择多视图参考帧。
- 其他资源：`docs/CLI.md`、`docs/API.md`、`docs/funcs/ref_view_strategy.md`，示例笔记本 `notebooks/da3.ipynb`，示例数据 `assets/examples`（含多张 SOH 图片与 robot_unitree.mp4）。

## 基本用法速记
- Python API 示例：
  ```python
  from depth_anything_3 import DepthAnything3
  model = DepthAnything3.from_pretrained("depth-anything/DA3NESTED-GIANT-LARGE").to("cuda")
  pred = model.inference(
      images,  # list of paths / PIL / numpy
      export_dir="out",
      export_format="glb",
      use_ray_pose=False,
      infer_gs=False,
  )
  # 返回 depth/conf/extrinsics/intrinsics/processed_images，若开启 infer_gs 还会给 gaussians
- 导出 `gs_ply/gs_video` 时必须 `infer_gs=True` 且使用带 GS 分支的权重（如 da3-giant 或 nested）。
- `align_to_input_ext_scale=True` 会用输入外参尺度重标定深度；`ref_view_strategy` 参见 docs（默认 `saddle_balanced`，视频序列可改 `middle`）。
- CLI 常用命令（可先 `export MODEL_DIR=depth-anything/DA3NESTED-GIANT-LARGE`、`export GALLERY_DIR=workspace/gallery`）：
- 启动后台缓存：`da3 backend --model-dir $MODEL_DIR --gallery-dir $GALLERY_DIR`
- 自动识别输入：`da3 auto assets/examples/SOH --export-format glb --export-dir ${GALLERY_DIR}/SOH --use-backend`
- 视频：`da3 video assets/examples/robot_unitree.mp4 --fps 15 --use-backend --export-format glb-feat_vis`
- COLMAP：`da3 colmap path/to/colmap --sparse-subdir 0 --align-to-input-ext-scale`
- Gradio 前端：`da3 gradio --model-dir $MODEL_DIR --workspace-dir workspace/gradio --gallery-dir $GALLERY_DIR --share`
- 独立画廊：`da3 gallery --gallery-dir $GALLERY_DIR --open-browser`
- 导出目录默认 `workspace/gallery/scene`，可用 `--auto-cleanup` 覆盖旧结果；GLB 导出用 `--conf-thresh-percentile`、`--num-max-points` 控制点云密度/过滤。

## 模型与资源
- 模型（Hugging Face）：`DA3NESTED-GIANT-LARGE`（嵌套+度量尺度）、`DA3-GIANT`（含 GS）、`DA3-LARGE/BASE/SMALL`、`DA3METRIC-LARGE`、`DA3MONO-LARGE`。
- 许可证：giant/large 等多为 CC BY-NC 4.0，base/small/metric/mono 为 Apache 2.0（以 README 表格为准）。
- 示例输出：导出目录可包含 `scene.glb`、`scene.jpg`、`depth_vis/`、`gs_video/`、`gs_ply/` 等，Prediction 对象也包含深度/置信度/相机参数。

## 当前缺口与下一步
- 需先安装 pip 和项目依赖，否则 CLI/API 无法运行。
- 拉取模型前确认 GPU/CUDA 版本兼容并准备显存，必要时设置 `HF_ENDPOINT` 镜像；gs 输出需额外安装 `gsplat`。
