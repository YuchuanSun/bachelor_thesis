# 项目精简计划 - 只保留必要的文件

## [x] 任务 1: 删除缓存目录
- **Priority**: P0
- **Depends On**: None
- **Description**:
  - 删除所有 `__pycache__` 目录，这些是Python编译生成的缓存文件，不影响项目运行
- **Success Criteria**:
  - 所有 `__pycache__` 目录被删除
- **Test Requirements**:
  - `programmatic` TR-1.1: 执行删除命令后，使用 `ls` 命令确认 `__pycache__` 目录不存在
- **Notes**: 这些目录会在Python运行时自动生成，删除后不影响功能

## [x] 任务 2: 删除已完成任务的脚本
- **Priority**: P0
- **Depends On**: 任务 1
- **Description**:
  - 删除已经完成任务的脚本文件，包括：
    - `scripts/extract_docx.py` - 已完成提取文档任务
    - `scripts/convert_algae_descriptions.py` - 已完成转换藻类描述任务
    - `scripts/merge_algae_descriptions.py` - 已完成合并藻类描述任务
    - `scripts/verify_bert_usage.py` - 已完成验证BERT使用任务
    - `scripts/verify_config_load.py` - 已完成验证配置加载任务
- **Success Criteria**:
  - 上述脚本文件被删除
- **Test Requirements**:
  - `programmatic` TR-2.1: 执行删除命令后，使用 `ls` 命令确认这些脚本文件不存在
- **Notes**: 这些脚本已经完成了它们的任务，不再需要保留

## [x] 任务 3: 删除冗余的可视化和评估脚本
- **Priority**: P1
- **Depends On**: 任务 2
- **Description**:
  - 删除冗余的可视化和评估脚本，包括：
    - `scripts/comprehensive_verification.py` - 综合验证脚本
    - `scripts/comprehensive_visualization.py` - 综合可视化脚本
    - `scripts/enhanced_visualization.py` - 增强可视化脚本
    - `scripts/improved_evaluation.py` - 改进评估脚本
    - `scripts/run_visualization.py` - 运行可视化脚本
    - `scripts/ablation_study.py` - 消融研究脚本
- **Success Criteria**:
  - 上述脚本文件被删除
- **Test Requirements**:
  - `programmatic` TR-3.1: 执行删除命令后，使用 `ls` 命令确认这些脚本文件不存在
- **Notes**: 保留核心的 `src/visualization.py` 和 `scripts/model_evaluator.py` 即可

## [x] 任务 4: 验证必要文件的完整性
- **Priority**: P1
- **Depends On**: 任务 3
- **Description**:
  - 验证核心文件和目录是否完整，确保项目能够正常运行
  - 检查以下必要文件和目录：
    - `configs/default_config.py`
    - `data/text_descriptions.json`
    - `data/images/`
    - `src/models/multimodal.py`
    - `src/dataset.py`
    - `src/train_multimodal.py`
    - `src/visualization.py`
    - `src/loss.py`
    - `scripts/dataset_splitter.py`
    - `scripts/model_evaluator.py`
- **Success Criteria**:
  - 所有必要的文件和目录都存在
- **Test Requirements**:
  - `programmatic` TR-4.1: 使用 `ls` 命令确认所有必要文件和目录存在
  - `programmatic` TR-4.2: 运行 `python scripts/verify_config_load.py` 验证配置加载正常
- **Notes**: 确保核心功能不受影响

## [x] 任务 5: 清理 data 目录中的冗余文件
- **Priority**: P2
- **Depends On**: 任务 4
- **Description**:
  - 检查 `data` 目录中是否有冗余文件，如临时文件、备份文件等
  - 保留 `text_descriptions.json` 和 `images/` 目录，删除其他可能的冗余文件
- **Success Criteria**:
  - `data` 目录中只保留必要的文件和目录
- **Test Requirements**:
  - `programmatic` TR-5.1: 使用 `ls` 命令确认 `data` 目录中只保留必要的文件和目录
- **Notes**: 注意保留 `dataset_split.json` 文件，它可能用于数据集分割