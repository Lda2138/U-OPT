# U-OPT 代码说明

本仓库发布已核查的七组chair组合消融实现、训练与断点恢复、真实损失曲线、
统一评测和逐对象数值。基于[UNICORN](https://github.com/monniert/unicorn)，
保留MIT许可和原作者署名。

G是几何残差MLP，T是纹理残差MLP，S是残差融合系数的延迟线性调度。
七组为U/G/T/GT/GS/TS/GTS；无残差时S没有调度对象，不另设重复的仅S组。
交替优化、渐进条件化、邻居一致性和模板连接性继承自UNICORN。

## 使用流程

1. 按根目录`environment.yml`准备Linux/CUDA环境。
2. 自行取得ShapeNet NMR chair数据与完整VGG16权重，遵守各自条款。
3. 根目录README中的`train.py --prepare`接收数据、VGG和个人输出路径。
4. 先执行生成的`smoke.sbatch`，通过后再提交`train_array.sbatch`。
5. 七组训练完成后用`evaluate.py --prepare`创建评测阶段，先pilot后全量。
6. `scripts/summarize_results.py`核对报告并生成均值与逐对象CSV。

详细命令见[复现说明](REPRODUCING.md)。所有运行输出写入指定workspace。
仓库不分发数据集图片、点云和大权重；目前没有公开任务权重下载。

## 结果边界

公开结果是chair、单种子132422、每组238轮。GTS图像PSNR/SSIM最好，
GS的ICP后几何最好。完整GTS相对基线的CD仅降低0.245%，F-score和法向
略退步；不能宣称三个组件各自有效或所有指标全面提升。

真实曲线和数值保留改善与退步。[结果说明](RESULTS.md)区分ICP前后、
训练损失、验证监控和最终测试。公开启动器改成了可配置路径，需在新环境
执行GPU验收；不宣称本仓库能复现原稿中所有历史类别和表格。
