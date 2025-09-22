# CoinPilot 预测服务

基于机器学习的用户转换预测系统，包含FastAPI后端服务和Streamlit前端界面。

## 🚀 功能特性

- **机器学习管道**: 使用Stacking Classifier (XGBoost + AdaBoost)进行预测
- **FastAPI服务**: RESTful API提供预测服务
- **Streamlit客户端**: 用户友好的Web界面
- **离线模式**: API不可用时自动切换到本地模型
- **实时预测**: 即时显示预测结果和置信度

## 📁 文件结构

```
├── Li Jinxuan_assignment2.ipynb    # 主要notebook文件
├── coinpilot_data.csv              # 训练数据
├── coinpilot_prediction_pipeline.joblib  # 保存的机器学习管道
├── fastapi_app.py                  # FastAPI应用
├── streamlit_app.py                # Streamlit客户端
├── test_prediction.py              # 测试脚本
└── README.md                       # 说明文档
```

## 🛠️ 安装依赖

```bash
pip install numpy pandas scikit-learn xgboost matplotlib fastapi "pydantic<2" uvicorn streamlit joblib requests
```

## 🚀 快速开始

### 1. 创建机器学习管道

首先运行notebook中的管道创建代码：

```python
# 在notebook中运行
# 这会创建并保存 coinpilot_prediction_pipeline.joblib
```

### 2. 启动FastAPI服务

```bash
python fastapi_app.py
```

或者使用uvicorn：

```bash
uvicorn fastapi_app:app --host 0.0.0.0 --port 8000 --reload
```

服务将在 http://localhost:8000 启动

### 3. 启动Streamlit客户端

```bash
streamlit run streamlit_app.py
```

客户端将在 http://localhost:8501 启动

### 4. 运行测试

```bash
python test_prediction.py
```

## 📋 API端点

### 健康检查
```
GET /health
```

### 服务信息
```
GET /info
```

### 预测转换
```
POST /predict
```

**请求体示例:**
```json
{
  "age": 30,
  "tenure_months": 12,
  "income_monthly": 5000.0,
  "savings_rate": 0.3,
  "risk_score": 50.0,
  "app_opens_7d": 10,
  "sessions_7d": 15,
  "avg_session_min": 5.0,
  "alerts_opt_in": 1,
  "auto_invest": 1,
  "country": "SG",
  "equity_pct": 40.0,
  "bond_pct": 20.0,
  "cash_pct": 30.0,
  "crypto_pct": 10.0
}
```

**响应示例:**
```json
{
  "prediction": 1,
  "probability": 0.75,
  "confidence": "中",
  "timestamp": "2024-01-01T12:00:00"
}
```

## 🎯 使用说明

### Streamlit界面

1. **配置设置**: 在侧边栏设置API地址和模式
2. **输入用户信息**: 填写所有必要的用户特征
3. **开始预测**: 点击"开始预测"按钮
4. **查看结果**: 查看预测结果、概率和置信度

### 特征说明

- **基本信息**: 年龄、使用月数、月收入、储蓄率、风险评分
- **使用行为**: 应用打开次数、会话次数、平均会话时长、提醒设置、自动投资
- **投资组合**: 股票、债券、现金、加密货币投资比例
- **地理位置**: 国家代码 (SG, MY, TH, ID, PH, VN)

## 🔧 技术架构

### 机器学习管道
- **预处理**: StandardScaler标准化连续特征
- **模型**: Stacking Classifier (XGBoost + AdaBoost)
- **元学习器**: Logistic Regression
- **特征**: 20个特征，包括连续、二进制和分类特征

### 服务架构
```
用户输入 → Streamlit界面 → FastAPI服务 → 机器学习管道 → 预测结果
                ↓ (如果API不可用)
            本地管道 → 预测结果
```

## 🧪 测试

运行测试脚本验证系统功能：

```bash
python test_prediction.py
```

测试包括：
- 离线预测功能
- API端点测试
- 端到端预测流程

## 📊 性能指标

基于测试数据的模型性能：
- **准确率**: 69.7%
- **精确率**: 72.4%
- **召回率**: 85.6%
- **ROC AUC**: 72.4%

## 🚨 故障排除

### 常见问题

1. **模型文件不存在**
   - 确保先运行notebook中的管道创建代码
   - 检查 `coinpilot_prediction_pipeline.joblib` 文件是否存在

2. **API连接失败**
   - 确保FastAPI服务正在运行
   - 检查端口8000是否被占用
   - 使用离线模式作为备选方案

3. **依赖包问题**
   - 确保安装了所有必要的Python包
   - 检查Python版本兼容性

### 日志信息

- FastAPI服务启动时会显示模型加载状态
- Streamlit界面会显示连接状态和错误信息
- 测试脚本会提供详细的测试结果

## 📝 开发说明

### 添加新特征
1. 更新 `UserInput` 模型
2. 修改 `preprocess_input` 函数
3. 更新Streamlit界面
4. 重新训练和保存管道

### 修改模型
1. 在notebook中修改模型配置
2. 重新训练管道
3. 保存新的管道文件
4. 重启服务

## 📄 许可证

本项目仅用于学术目的。

## 👥 作者

Li Jinxuan - IS5126 Individual Assignment 2


