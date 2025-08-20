# backend/main.py
from fastapi import FastAPI, HTTPException, Response, Body
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Optional, List, Dict, Union
from uuid import uuid4
from pathlib import Path
import json
import torch
import pandas as pd
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from starlette.requests import Request

import utils.simulate_all_drugs_fast as sim_mod
import utils.get_simulation_data as data_mod

from PIL import Image as PIL_Image 
sim_mod.Image = PIL_Image

# 정적파일/유틸
from fastapi.staticfiles import StaticFiles
from urllib.parse import quote
import os
import shutil
import contextlib
import random

# 모델 클래스 임포트
try:
    from model_def import CanSimModel
except Exception:
    from cansim_model import CanSimModel

# FastAPI 앱
app = FastAPI(title="CanSim Backend", version="2.1.0-slim")

# CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://127.0.0.1:5173", "http://localhost:5173"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
    expose_headers=["X-Report-Id"],
)

# 정적 이미지 폴더 마운트
IMG_DIR = Path("temp_images")
IMG_DIR.mkdir(exist_ok=True, parents=True)
app.mount("/temp_images", StaticFiles(directory=str(IMG_DIR)), name="temp_images")

# 원본 이미지 폴더
from PIL import Image as PIL_Image
BASE_IMAGE_DIR = Path("data/base_images")
BASE_IMAGE_DIR.mkdir(exist_ok=True, parents=True)
ALLOWED_EXTS = {".png", ".jpg", ".jpeg", ".bmp"}

def _safe_image_path(filename: str) -> Path:
    fname = os.path.basename(filename) 
    path = BASE_IMAGE_DIR / fname
    if path.suffix.lower() not in ALLOWED_EXTS:
        raise HTTPException(status_code=400, detail="허용되지 않는 이미지 확장자입니다.")
    if not path.exists():
        raise HTTPException(status_code=404, detail=f"이미지 파일을 찾을 수 없습니다: {fname}")
    return path

CANCER_FILE_PREFIX = {
    "폐암": "ct_lung_patient",
    "대장암": "ct_colon_patient",
    "위암": "ct_stomach_patient",
    "뇌교종": "ct_brain_patient",
    "유방암": "ct_breast_patient",
    "전립선암": "mri_prostate_patient",
}

def _pick_random_base_image(cancer_type_kor: str) -> Path:
    """암 종류(한국어)에 따라 정해진 prefix로 시작하는 파일들 중 랜덤 1개 선택"""
    prefix = CANCER_FILE_PREFIX.get(cancer_type_kor)
    if not prefix:
        raise HTTPException(status_code=400, detail=f"암 종류 매핑이 정의되지 않았습니다: {cancer_type_kor}")

    candidates: List[Path] = []
    for p in BASE_IMAGE_DIR.iterdir():
        if not p.is_file():
            continue
        if p.suffix.lower() not in ALLOWED_EXTS:
            continue
        if p.name.lower().startswith(prefix.lower()):
            candidates.append(p)

    if not candidates:
        raise HTTPException(status_code=404, detail=f"{cancer_type_kor}({prefix}) 관련 원본 이미지를 찾을 수 없습니다.")
    return random.choice(candidates)

# 예외 로깅
@app.exception_handler(RequestValidationError)
async def validation_exception_handler(request: Request, exc: RequestValidationError):
    try:
        body = await request.body()
        print("[422] Request body raw:", body.decode("utf-8", errors="ignore"))
    except Exception:
        pass
    print("[422] errors:", exc.errors())
    return JSONResponse(status_code=422, content={"detail": exc.errors()})

# 데이터 스키마
class PatientData(BaseModel):
    name: Optional[str] = None
    age: Optional[int] = None
    gender: Optional[str] = None
    cancerType: Optional[str] = None 
    stage: Optional[str] = None
    chemo: Optional[str] = None
    temperature: Optional[str] = None
    bloodPressure: Optional[str] = None
    pulse: Optional[Union[int, str]] = None
    extra: Optional[str] = None
    baseImage: Optional[str] = None   

class DrugPlan(BaseModel):
    name: str
    probability: float
    recommended: bool
    schedule: List[str]

class ReportPayload(BaseModel):
    report_id: str
    patient: PatientData
    plans: List[DrugPlan]

# 메모리 저장소
REPORT_STORE: Dict[str, ReportPayload] = {}

# 결과 저장 디렉토리
RESULT_DIR = Path("results")
RESULT_DIR.mkdir(exist_ok=True)

# CSV 기반 약제 라벨 매핑
CSV_PATH = Path("cancer_drug_effect.csv")
if not CSV_PATH.exists():
    raise FileNotFoundError(f"{CSV_PATH} 파일이 존재하지 않습니다!")

df = pd.read_csv(CSV_PATH)

# 암별 drug_type → drug_name 매핑
label2drug_map: Dict[str, Dict[int, str]] = {}
for cancer, group in df.groupby("cancer_name"):
    mapping = dict(zip(group["drug_type"], group["drug_name"]))
    label2drug_map[cancer] = mapping

# 유틸 호환: cancer_type(int) 열 준비
if "cancer_type" not in df.columns:
    cancer_name_to_id = {name: idx for idx, name in enumerate(sorted(df["cancer_name"].unique()))}
    df["cancer_type"] = df["cancer_name"].map(cancer_name_to_id)
else:
    cancer_name_to_id = {name: int(cid) for name, cid in df[["cancer_name","cancer_type"]].drop_duplicates().itertuples(index=False)}

# (cancer_type_int, drug_idx) → drug_name 맵
drug_map: Dict[tuple, str] = {}
for _, row in df.iterrows():
    ct = int(row["cancer_type"])
    di = int(row["drug_type"])
    dn = str(row["drug_name"])
    drug_map[(ct, di)] = dn

# num_drugs: 모델 출력 차원
num_drugs = int(df["drug_type"].max()) + 1

# 모델 로드
MODEL_PATH = Path("models/best_model.pth")
if not MODEL_PATH.exists():
    raise FileNotFoundError(f"{MODEL_PATH} 파일이 존재하지 않습니다!")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 안전하게 모델 생성
def _build_model_safely(num_drugs: int):
    try:
        m = CanSimModel(tabular_dim=3, hidden_dim=64, num_drugs=num_drugs).to(device)
        return m
    except Exception:
        import torchvision.models as _tv
        _orig_resnet18 = _tv.resnet18
        def _safe_resnet18(*args, **kwargs):
            if "weights" in kwargs:
                kwargs["weights"] = None
            else:
                kwargs["pretrained"] = False
            return _orig_resnet18(*args, **kwargs)
        _tv.resnet18 = _safe_resnet18
        try:
            m = CanSimModel(tabular_dim=3, hidden_dim=64, num_drugs=num_drugs).to(device)
            return m
        finally:
            _tv.resnet18 = _orig_resnet18

model = _build_model_safely(num_drugs)

state = torch.load(MODEL_PATH, map_location=device)
if isinstance(state, dict) and "state_dict" in state and isinstance(state["state_dict"], dict):
    state = state["state_dict"]
if not isinstance(state, dict):
    raise RuntimeError("Unexpected checkpoint format (expected dict or dict['state_dict']).")

def clean_keys(sd: dict) -> dict:
    cleaned = {}
    for k, v in sd.items():
        if k.startswith("_orig_mod."):
            k = k[len("_orig_mod."):]
        if k.startswith("module."):
            k = k[len("module."):]
        cleaned[k] = v
    return cleaned

state = clean_keys(state)
incompat = model.load_state_dict(state, strict=False)
print(f"[Model] load_state_dict: missing={len(incompat.missing_keys)}, unexpected={len(incompat.unexpected_keys)}")
model.eval()

# 유틸 기반 시뮬레이션
from torch import amp

_orig_autocast = amp.autocast
def _safe_autocast(device_type="cuda", dtype=None):
    if device_type != "cuda":
        return contextlib.nullcontext()
    return _orig_autocast(device_type=device_type, dtype=dtype)
amp.autocast = _safe_autocast 

# 유틸 모듈
from utils import simulate_all_drugs_fast as sim_mod
from utils import get_simulation_data as data_mod

def _cancer_name_to_id(name: str) -> int:
    if name in cancer_name_to_id:
        return int(cancer_name_to_id[name])
    if "cancer_type" in df.columns and (df["cancer_name"] == name).any():
        return int(df[df["cancer_name"] == name]["cancer_type"].iloc[0])
    return 0

def _as_float(v, default):
    try:
        if v is None or v == "":
            return float(default)
        return float(v)
    except Exception:
        try:
            return float(str(v))
        except Exception:
            return float(default)

def _as_int(v, default):
    try:
        if v is None or v == "":
            return int(default)
        return int(float(v))
    except Exception:
        try:
            return int(float(str(v)))
        except Exception:
            return int(default)

@app.get("/")
def root():
    return {"status": "ok", "message": "CanSim backend is running"}

@app.get("/api/ping")
def ping():
    return {"message": "pong"}

@app.post("/api/simulate_viz")
def simulate_viz(patient: PatientData, response: Response, request: Request):
    try:
        if not patient.cancerType or patient.cancerType not in label2drug_map:
            raise HTTPException(status_code=400, detail="지원되지 않는 암 종류입니다.")

        # base image 결정
        if patient.baseImage:
            img_path = _safe_image_path(patient.baseImage)
        else:
            img_path = _pick_random_base_image(patient.cancerType)

        base_image = PIL_Image.open(img_path).convert("L")

        # 탭 데이터
        tab = torch.tensor([
            _as_float(patient.age, 0),
            _as_float(patient.temperature, 36.5),
            float(_as_int(patient.pulse, 70)),
        ], dtype=torch.float32, device=device)

        cancer_type_int = _cancer_name_to_id(patient.cancerType)

        # 유틸 시뮬레이션 실행
        results, best = sim_mod.simulate_all_drugs_fast(
            model=model,
            base_image=base_image,
            tabular_tensor=tab,
            device=device,
            cancer_type=cancer_type_int,
            drug_map=drug_map,
            drug_df=df,
            num_weeks=12,
        )

        # report_id 디렉토리 준비
        report_id = uuid4().hex
        response.headers["X-Report-Id"] = report_id
        per_req_dir = IMG_DIR / report_id
        per_req_dir.mkdir(parents=True, exist_ok=True)

        # 유틸의 get_simulation_data 호출 (임시 파일 생성됨)
        data = data_mod.get_simulation_data(
            results=results,
            best_drug=best,
            detailed=True,
        )

        # temp_images/ 루트에 저장된 파일을 report 전용 폴더로 이동 + URL 생성
        base_url = str(request.base_url).rstrip("/")
        for drug in data:
            for wk in drug["weeks"]:
                if "image_path" in wk:
                    src = Path(wk["image_path"])  
                    if not src.is_absolute():
                        src = Path.cwd() / src
                    if not src.exists():
                        continue

                    dst_name = f"{report_id}_{src.name}"
                    dst = per_req_dir / dst_name
                    try:
                        shutil.move(str(src), str(dst))
                    except Exception:
                        shutil.copy2(str(src), str(dst))
                        try:
                            os.remove(str(src))
                        except Exception:
                            pass

                    wk["image_url"] = f"{base_url}/temp_images/{quote(report_id)}/{quote(dst_name)}"

        # 약물별 최종 확률(%) 요약 (마지막 주 score 사용)
        probabilities = []
        best_name = best.get("drug") if isinstance(best, dict) else None
        for r in results:
            if not r.get("scores"):
                continue
            last = float(r["scores"][-1]) * 100.0
            probabilities.append({
                "drug": r["drug"],
                "percent": round(last, 1),
                "recommended": (r["drug"] == best_name)
            })

        # Summary와 호환되는 ReportPayload 저장
        plans = [
            DrugPlan(
                name=prob["drug"],
                probability=prob["percent"],
                recommended=prob["recommended"],
                schedule=[f"{j}주차" for j in range(1, 13)],
            )
            for prob in probabilities
        ]
        payload = ReportPayload(
            report_id=report_id,
            patient=patient,
            plans=plans
        )
        REPORT_STORE[report_id] = payload

        # 최종 응답
        return {
            "report_id": report_id,
            "patient": {
                "name": patient.name,
                "age": patient.age,
                "gender": patient.gender,
                "cancerType": patient.cancerType,
                "baseImage": os.path.basename(str(img_path)),
            },
            "best": best_name,
            "probabilities": probabilities,
            "drugs": data,   # 약물별 1~12주 이미지/점수
        }

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/report/{report_id}", response_model=ReportPayload)
def get_report_by_id(report_id: str):
    payload = REPORT_STORE.get(report_id)
    if not payload:
        raise HTTPException(status_code=404, detail="Report not found.")
    return payload

@app.post("/api/save-result")
def save_result(data: dict = Body(...)):
    try:
        from datetime import datetime
        report_id = data.get("report_id", "unknown")
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        file_path = RESULT_DIR / f"result_{report_id}_{timestamp}.json"
        with open(file_path, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
        return {"ok": True, "path": str(file_path)}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="127.0.0.1", port=8000, reload=True)


#.\.venv\Scripts\Activate.ps1

#uvicorn main:app --reload --port 8000

#npm run dev