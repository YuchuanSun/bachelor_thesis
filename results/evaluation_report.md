# 藻类识别模型评估报告

## 评估时间
2026-03-25 20:30:15

## 数据集信息
- 训练集: 1738 张图像
- 验证集: 367 张图像
- 测试集: 379 张图像
- 总图像数: 2484 张图像

## 模型性能指标
- 准确率: 0.6306
- 精确率: 0.9633
- 召回率: 0.6306
- F1分数: 0.6711

## 预测结果
- 测试样本数: 379
- 正确预测: 239
- 错误预测: 140

## 可视化图表

1. 数据划分可视化: results/dataset_split_visualization.png
2. 性能指标柱状图: results/model_performance_metrics.png
3. 混淆矩阵: results/confusion_matrix.png
4. 性能雷达图: results/performance_radar.png
5. 各类别准确率: results/class_accuracy.png
6. 预测概率分布: results/probability_distribution.png
7. 错误分析: results/error_analysis.png
8. 数据分布: results/data_distribution.png

## 结论

模型在独立测试集上的表现为：
- 准确率: 63.06%
- F1分数: 67.11%

这些结果真实反映了模型的泛化能力，为论文提供了可靠的性能评估数据。
