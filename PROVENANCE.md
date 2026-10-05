# 发布说明与结果溯源

整理日期：2026-10-06。

## 来源与处理

课程代码来自 biai-mlp-0.1.0 课程包，原仓库为 https://github.com/VeriTas-arch/brain-inspired-ai 。本仓库保留课程 Notebook、所需辅助模块、README 与 MIT 许可证；没有把课程代码声称为补充实验作者的原创。

`mlp.ipynb` 保留报告使用的已执行输出，SHA-256 为：

```text
155d17a674af259738ddc2e040c2d6ccc7b38ba7ed3547e668f49b9a58ac5347
```

`gha-decay/run_experiment.py` 与报告实验实际使用的脚本一致。`gha-decay/results/` 保留原实验完整精度指标、模型参数和运行方案。仅将 `protocol.json` 及 `results.json` 中的 `source` 字段由本机绝对路径替换为仓库相对路径 `biai-mlp-0.1.0/mlp.ipynb`；因此这两个 JSON 的文件字节哈希会变化，但实验数值、数据哈希、模型哈希和源 Notebook 哈希未修改。

`tables/` 与 `figures/` 来自这些已保存的实验输出。未上传依赖本机绝对目录的报告生成脚本、课程报告草稿、缓存和 MNIST 原始数据。表格 03 是本地提取图片的索引，因对应提取目录不随仓库发布而省略。

## 复现与核对

数据由 torchvision 的 MNIST 下载功能获取，或从原课程包复制。原实验四个解压后的 MNIST 文件哈希保存在 `gha-decay/results/protocol.json`。网络下载会额外留下压缩文件，新方案中的文件列表可能因此增加，不表示四个原始数据文件发生变化。

新运行生成自己的时间、运行耗时及源路径，这些元数据不应要求与原结果完全相同。`.pt` 是 PyTorch 参数字典，读取时使用 `torch.load(..., weights_only=True)`；文件序列化哈希和模型数值哈希也应区分。不同软件版本、硬件和运算实现可能导致数值差异。

基础实验主要是单次运行；补充实验采用三个配对随机种子。结果只支持已记录任务、模型、学习率与训练预算下的结论，不能据此认定某学习算法普遍更优。输入没有中心化，GHA 在这里也不宜直接称为严格的协方差 PCA。
