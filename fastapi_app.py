# FastAPI应用文件
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from typing import List
import joblib
import pandas as pd
from datetime import datetime
import uvicorn

# 定义Pydantic模型
class UserInput(BaseModel):
    """用户输入数据模型"""
    age: int = Field(..., ge=18, le=100, description="用户年龄")
    tenure_months: int = Field(..., ge=1, le=60, description="用户使用月数")
    income_monthly: float = Field(..., gt=0, description="月收入")
    savings_rate: float = Field(..., ge=0, le=1, description="储蓄率")
    risk_score: float = Field(..., ge=0, le=100, description="风险评分")
    app_opens_7d: int = Field(..., ge=0, description="7天内应用打开次数")
    sessions_7d: int = Field(..., ge=0, description="7天内会话次数")
    avg_session_min: float = Field(..., ge=0, description="平均会话时长(分钟)")
    alerts_opt_in: int = Field(..., ge=0, le=1, description="是否选择接收提醒")
    auto_invest: int = Field(..., ge=0, le=1, description="是否自动投资")
    country: str = Field(..., description="国家代码")
    equity_pct: float = Field(..., ge=0, le=100, description="股票投资比例")
    bond_pct: float = Field(..., ge=0, le=100, description="债券投资比例")
    cash_pct: float = Field(..., ge=0, le=100, description="现金投资比例")
    crypto_pct: float = Field(..., ge=0, le=100, description="加密货币投资比例")

class PredictionResponse(BaseModel):
    """预测响应模型"""
    prediction: int = Field(..., description="预测结果 (0=不转换, 1=转换)")
    probability: float = Field(..., description="转换概率")
    confidence: str = Field(..., description="置信度等级")
    timestamp: str = Field(..., description="预测时间戳")

class HealthResponse(BaseModel):
    """健康检查响应模型"""
    status: str = Field(..., description="服务状态")
    timestamp: str = Field(..., description="检查时间")
    model_loaded: bool = Field(..., description="模型是否已加载")

class InfoResponse(BaseModel):
    """服务信息响应模型"""
    service_name: str = Field(..., description="服务名称")
    version: str = Field(..., description="版本号")
    model_type: str = Field(..., description="模型类型")
    features: List[str] = Field(..., description="特征列表")
    created_at: str = Field(..., description="创建时间")

# 创建FastAPI应用
app = FastAPI(
    title="CoinPilot Premium Conversion Prediction API",
    description="预测用户是否会转换为高级用户的机器学习API服务",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc"
)

# 全局变量存储加载的模型
ml_pipeline = None

def load_model():
    """加载机器学习模型"""
    global ml_pipeline
    try:
        ml_pipeline = joblib.load('coinpilot_prediction_pipeline.joblib')
        print("✅ 模型加载成功!")
        return True
    except Exception as e:
        print(f"❌ 模型加载失败: {e}")
        return False

def preprocess_input(user_input: UserInput) -> pd.DataFrame:
    """预处理用户输入数据"""
    # 创建国家独热编码
    country_columns = ['country_ID', 'country_MY', 'country_PH', 'country_SG', 'country_TH', 'country_VN']
    country_encoded = {col: 1 if col == f'country_{user_input.country}' else 0 for col in country_columns}
    
    # 构建特征字典
    features = {
        'age': user_input.age,
        'tenure_months': user_input.tenure_months,
        'income_monthly': user_input.income_monthly,
        'savings_rate': user_input.savings_rate,
        'risk_score': user_input.risk_score,
        'app_opens_7d': user_input.app_opens_7d,
        'sessions_7d': user_input.sessions_7d,
        'avg_session_min': user_input.avg_session_min,
        'alerts_opt_in': user_input.alerts_opt_in,
        'auto_invest': user_input.auto_invest,
        'equity_pct': user_input.equity_pct,
        'bond_pct': user_input.bond_pct,
        'cash_pct': user_input.cash_pct,
        'crypto_pct': user_input.crypto_pct,
        **country_encoded
    }
    
    # 转换为DataFrame
    df = pd.DataFrame([features])
    return df

def get_confidence_level(probability: float) -> str:
    """根据概率确定置信度等级"""
    if probability >= 0.8:
        return "高"
    elif probability >= 0.6:
        return "中"
    else:
        return "低"

# API端点
@app.get("/health", response_model=HealthResponse)
async def health_check():
    """健康检查端点"""
    return HealthResponse(
        status="healthy" if ml_pipeline is not None else "unhealthy",
        timestamp=datetime.now().isoformat(),
        model_loaded=ml_pipeline is not None
    )

@app.get("/info", response_model=InfoResponse)
async def get_info():
    """服务信息端点"""
    feature_list = [
        'age', 'tenure_months', 'income_monthly', 'savings_rate', 'risk_score',
        'app_opens_7d', 'sessions_7d', 'avg_session_min', 'alerts_opt_in', 'auto_invest',
        'equity_pct', 'bond_pct', 'cash_pct', 'crypto_pct',
        'country_ID', 'country_MY', 'country_PH', 'country_SG', 'country_TH', 'country_VN'
    ]
    
    return InfoResponse(
        service_name="CoinPilot Premium Conversion Prediction API",
        version="1.0.0",
        model_type="Stacking Classifier (XGBoost + AdaBoost)",
        features=feature_list,
        created_at=datetime.now().isoformat()
    )

@app.post("/predict", response_model=PredictionResponse)
async def predict_conversion(user_input: UserInput):
    """预测用户转换概率"""
    if ml_pipeline is None:
        raise HTTPException(status_code=503, detail="模型未加载，请稍后重试")
    
    try:
        # 预处理输入数据
        processed_data = preprocess_input(user_input)
        
        # 进行预测
        prediction = ml_pipeline.predict(processed_data)[0]
        probability = ml_pipeline.predict_proba(processed_data)[0][1]
        
        # 确定置信度
        confidence = get_confidence_level(probability)
        
        return PredictionResponse(
            prediction=int(prediction),
            probability=float(probability),
            confidence=confidence,
            timestamp=datetime.now().isoformat()
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"预测失败: {str(e)}")

# 启动事件
@app.on_event("startup")
async def startup_event():
    """应用启动时加载模型"""
    print("🚀 启动FastAPI服务...")
    success = load_model()
    if success:
        print("✅ 服务启动成功!")
    else:
        print("❌ 服务启动失败!")

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)


